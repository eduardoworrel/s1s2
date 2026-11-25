#!/usr/bin/env node
// Node CPU parity of the decoder questions (language, term) against the torch references of decoder_refs.py.
//   node recipes/marimba-audio/decoder-parity.mjs --ref <dir with dec_ref.jsonl and af/> --weights <dir holding marimba-audio/>
//        [--mode features|e2e] [--limit N] [--only id,id] [--out result.json] [--device cpu]
// mode features: the decoder reads torch's `audio_features` (isolates the decoder build and the JS beam search).
// mode e2e:      the JS mel and the ONNX tower make the features from the left-aligned window (adds the tower's difference).
// Tolerances are those of spec/MULTIMODAL.md section 9.4; the script prints which hold. Heavy: tower 0.75 GB + decoder 0.66 GB.
import fs from "node:fs";
import path from "node:path";
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
const { loadMedia } = await import(path.join(ROOT, "dist/index.js"));
const args = process.argv.slice(2);
const opt = (k, d = null) => { const i = args.indexOf(k); return i < 0 ? d : args[i + 1]; };
const refDir = path.resolve(opt("--ref")), wdir = path.resolve(opt("--weights"));
const mode = opt("--mode", "features"), limit = Number(opt("--limit", 1e9)), only = opt("--only")?.split(","), outFile = opt("--out");
const rows = fs.readFileSync(path.join(refDir, "dec_ref.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((r) => !only || only.includes(r.id)).slice(0, limit);
const f32 = (f) => { const b = fs.readFileSync(f); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };

const model = await loadMedia({ manifest: path.join(ROOT, "manifests/marimba-audio.json"), device: opt("--device", "cpu"), localDirs: { "local/marimba-audio-web": path.join(wdir, "marimba-audio") }, onStatus: (m) => console.log("[load]", m) });
console.log("loaded; tasks", model.tasks.join(", "), "| windows", rows.length, "| mode", mode);

const softmax = (o) => { const ks = Object.keys(o), m = Math.max(...ks.map((k) => o[k])), e = ks.map((k) => Math.exp(o[k] - m)), s = e.reduce((a, b) => a + b, 0); return Object.fromEntries(ks.map((k, i) => [k, e[i] / s])); };
const out = [];
const S = { windows: 0, langArgmax: 0, langMaxDp: 0, langTopDp: 0, langMaxDScore: 0, pairs: 0, fracEq: 0, fracWithin: 0, decEq: 0, greedyEq: 0, beamTextEq: 0, beamTexts: 0, featMax: 0, ms: { open: [], language: [], term: [], total: [] } };
for (const r of rows) {
  const af = f32(path.join(refDir, "af", r.id + ".f32"));
  const terms = Object.keys(r.terms);
  const qs = [{ task: "language" }, ...terms.map((t) => ({ task: "term", term: t }))];
  const timing = {}, t0 = performance.now();
  let ans, featDiff = null;
  if (mode === "features") ans = await model.decodeFeatures(af, qs, timing);
  else {
    const win = f32(path.join(refDir, "af", r.id + ".win.f32"));
    const n = r.n, enc = await model.encode({ samples: win.subarray(0, n).slice(), sampleRate: 16000 });
    const left = enc.nValid >= enc.win ? enc.taps.audio_features : (await model.read(enc, [{ task: "language" }], timing), (await enc.left).audio_features);
    featDiff = 0; for (let i = 0; i < af.length; i++) featDiff = Math.max(featDiff, Math.abs(left[i] - af[i]));
    S.featMax = Math.max(S.featMax, featDiff);
    ans = await model.read(enc, qs, timing);
  }
  const ms = performance.now() - t0;
  const lang = ans[0], refP = softmax(r.language);
  const mine = Object.fromEntries(lang.probabilities.map((x) => [x.value, x.probability]));
  const topRef = Object.keys(refP).reduce((a, b) => (refP[a] >= refP[b] ? a : b));
  S.windows++; S.langArgmax += lang.choice === topRef ? 1 : 0;
  const dp = Math.max(...Object.keys(refP).map((k) => Math.abs(refP[k] - mine[k])));
  S.langMaxDp = Math.max(S.langMaxDp, dp); S.langTopDp = Math.max(S.langTopDp, Math.abs(refP[topRef] - mine[topRef]));
  const mineScore = Object.fromEntries(lang.probabilities.map((x, i) => [x.value, lang.raw[i]]));
  const dsc = Math.max(...Object.keys(r.language).map((k) => Math.abs(r.language[k] - mineScore[k])));
  S.langMaxDScore = Math.max(S.langMaxDScore, dsc);
  const hyps = ans[1].hypotheses;
  S.greedyEq += hyps[0] === r.hyps[0] ? 1 : 0;
  for (let k = 1; k < 5; k++) { S.beamTexts++; S.beamTextEq += hyps[k] === r.hyps[k] ? 1 : 0; }
  const pairs = [];
  terms.forEach((t, i) => {
    const a = ans[1 + i], p = a.probability, q = r.terms[t];
    S.pairs++; S.fracEq += Math.abs(p - q) < 1e-9 ? 1 : 0; S.fracWithin += Math.abs(p - q) <= 0.2 + 1e-9 ? 1 : 0; S.decEq += (p >= 0.5) === (q >= 0.5) ? 1 : 0;
    pairs.push([t, q, p]);
  });
  S.ms.open.push(timing.decoderOpen ?? 0); S.ms.language.push(timing.language ?? 0); S.ms.term.push(timing.term ?? 0); S.ms.total.push(ms);
  out.push({ id: r.id, lang: [topRef, lang.choice], dp, dScore: dsc, featDiff, hyps, refHyps: r.hyps, pairs, ms, timing });
  console.log(r.id, `lang ${topRef}/${lang.choice} dp=${dp.toExponential(1)} dScore=${dsc.toFixed(3)}`, `greedy=${hyps[0] === r.hyps[0] ? "=" : "!"}`, `beams=${[1, 2, 3, 4].map((k) => (hyps[k] === r.hyps[k] ? "=" : "!")).join("")}`, `terms ${pairs.filter(([, q, p]) => Math.abs(p - q) < 1e-9).length}/${pairs.length}`, `${Math.round(ms)} ms`);
}
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
const g = (ok) => (ok ? "ok" : "FAIL");
const sum = {
  mode, windows: S.windows,
  featuresMaxAbsDiff: mode === "e2e" ? S.featMax : null,
  languageArgmaxEqual: `${S.langArgmax}/${S.windows}`, languageMaxAbsDiffProbability: S.langMaxDp, languageMaxAbsDiffTopProbability: S.langTopDp, languageMaxAbsDiffScore: S.langMaxDScore,
  termPairs: S.pairs, termFractionEqual: S.fracEq / S.pairs, termFractionWithin02: S.fracWithin / S.pairs, termDecisionEqual: S.decEq / S.pairs,
  greedyTextEqual: S.greedyEq / S.windows, beamTextEqual: S.beamTextEq / S.beamTexts,
  medianMs: { open: med(S.ms.open), language: med(S.ms.language), term: med(S.ms.term), total: med(S.ms.total) },
};
sum.gates = {
  "language argmax identical on every window": g(S.langArgmax === S.windows),
  "language top probability abs diff <= 0.02": g(S.langTopDp <= 0.02),
  "term fraction within 0.2 on >= 95% of pairs": g(S.fracWithin / S.pairs >= 0.95),
  "term fraction identical on >= 85% of pairs": g(S.fracEq / S.pairs >= 0.85),
  "term decision at 0.5 identical on >= 95% of pairs": g(S.decEq / S.pairs >= 0.95),
  ...(mode === "e2e" ? { "audio_features max abs diff <= 2e-3": g(S.featMax <= 2e-3) } : {}),
};
console.log(JSON.stringify(sum, null, 1));
if (outFile) fs.writeFileSync(outFile, JSON.stringify({ summary: sum, windows: out }, null, 1));
await model.dispose();
