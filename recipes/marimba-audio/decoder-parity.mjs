#!/usr/bin/env node
// Node CPU parity of the decoder questions (language, term) against the torch references of decoder_refs.py.
//   node recipes/marimba-audio/decoder-parity.mjs --ref <dir with dec_ref.jsonl and af/> --weights <dir holding marimba-audio/>
//        [--mode features|e2e] [--manifest file] [--limit N] [--only id,id] [--out result.json] [--device cpu]
// mode features: the decoder reads torch's `audio_features` (isolates the decoder build and the JS search).
// mode e2e:      the JS mel and the ONNX tower make the features from the left-aligned window (adds the tower's difference).
// Tolerances are those of spec/MULTIMODAL.md section 9.4; the summary says which hold. Heavy: tower 0.75 GB + decoder 0.4-0.7 GB.
import fs from "node:fs";
import path from "node:path";
import { compare, rowFromAnswers } from "./decoder-metrics.mjs";
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
const { loadMedia } = await import(path.join(ROOT, "dist/media.js"));
const args = process.argv.slice(2);
const opt = (k, d = null) => { const i = args.indexOf(k); return i < 0 ? d : args[i + 1]; };
const refDir = path.resolve(opt("--ref")), wdir = path.resolve(opt("--weights"));
const mode = opt("--mode", "features"), limit = Number(opt("--limit", 1e9)), only = opt("--only")?.split(","), outFile = opt("--out");
const rows = fs.readFileSync(path.join(refDir, "dec_ref.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((r) => !only || only.includes(r.id)).slice(0, limit);
const f32 = (f) => { const b = fs.readFileSync(f); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };

const model = await loadMedia({ manifest: path.resolve(opt("--manifest", path.join(ROOT, "manifests/marimba-audio.json"))), device: opt("--device", "cpu"), localDirs: { "eduardoworrel/Marimba-ONNX-S1S2": path.join(wdir, "marimba-audio") }, onStatus: (m) => console.log("[load]", m) });
console.log("loaded; windows", rows.length, "| mode", mode);
const results = [];
for (const r of rows) {
  const af = f32(path.join(refDir, "af", r.id + ".f32")), terms = Object.keys(r.terms);
  const qs = [{ task: "language" }, ...terms.map((t) => ({ task: "term", term: t }))];
  const timing = {}, t0 = performance.now();
  let ans, featDiff = null;
  if (mode === "features") ans = await model.decodeFeatures(af, qs, timing);
  else {
    const win = f32(path.join(refDir, "af", r.id + ".win.f32"));
    const enc = await model.encode({ samples: win.subarray(0, r.n).slice(), sampleRate: 16000 });
    ans = await model.read(enc, qs, timing);
    const left = enc.nValid >= enc.win ? enc.taps.audio_features : (await enc.left).audio_features;
    featDiff = 0; for (let i = 0; i < af.length; i++) featDiff = Math.max(featDiff, Math.abs(left[i] - af[i]));
  }
  const ms = performance.now() - t0;
  const row = rowFromAnswers(r.id, ans, terms, timing, ms, featDiff);
  results.push(row);
  const c = compare([r], [row], { mode }).perWindow[0];
  console.log(r.id, `lang ${c.lang[0]}/${c.lang[1]} dp=${c.dp.toExponential(1)}`, `greedy=${c.greedyEqual ? "=" : "!"}`, `beams=${c.beamsEqual.map((x) => (x ? "=" : "!")).join("")}`, `terms ${c.termEqual}/${c.terms}`, featDiff != null ? `feat ${featDiff.toExponential(1)}` : "", `${Math.round(ms)} ms`);
}
const cmp = compare(rows, results, { mode });
console.log(JSON.stringify(cmp.summary, null, 1));
if (outFile) fs.writeFileSync(outFile, JSON.stringify({ summary: cmp.summary, perWindow: cmp.perWindow, results }, null, 1));
await model.dispose();
