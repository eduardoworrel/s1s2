#!/usr/bin/env node
// Node CPU check of AudioModel.transcribe on the real package: the greedy transcription must be the term question's greedy hypothesis and the beam
// transcription its four beams (same engine), on the same windows; reports the time of each call and, as information only, how many texts equal the
// float32 torch references of decoder_refs.py (int4 can differ from them).
//   node recipes/marimba-audio/transcribe-parity.mjs --ref <dir with dec_ref.jsonl and af/> --package <dir with manifest.json and the files>
//        [--kind joined] [--limit 12] [--out result.json]
import fs from "node:fs";
import path from "node:path";
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
const { loadMedia } = await import(path.join(ROOT, "dist/media.js"));
const args = process.argv.slice(2);
const opt = (k, d = null) => { const i = args.indexOf(k); return i < 0 ? d : args[i + 1]; };
const refDir = path.resolve(opt("--ref")), pkgDir = path.resolve(opt("--package")), kind = opt("--kind", "joined"), limit = Number(opt("--limit", 12)), outFile = opt("--out");
const rows = fs.readFileSync(path.join(refDir, "dec_ref.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((r) => r.kind === kind).slice(0, limit);
const f32 = (f) => { const b = fs.readFileSync(f); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const manifest = JSON.parse(fs.readFileSync(path.join(pkgDir, "manifest.json"), "utf8"));
const model = await loadMedia({ manifest, localDirs: { [manifest.repo]: pkgDir }, device: "cpu", onStatus: (m) => console.log("[load]", m) });
console.log("loaded; windows", rows.length, kind);
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const out = [];
for (const r of rows) {
  const win = f32(path.join(refDir, "af", r.id + ".win.f32"));
  const enc = await model.encode({ samples: win.subarray(0, r.n).slice(), sampleRate: 16000 });
  const t0 = performance.now();
  const [term] = await model.read(enc, [{ task: "term", term: "marimba" }]);
  const msTerm = performance.now() - t0;
  const g = await model.transcribe(enc), b = await model.transcribe(enc, { beam: true });
  const row = {
    id: r.id, greedyEqualsTerm: g.text === term.hypotheses[0], beamEqualsTerm: eq(b.hypotheses, term.hypotheses.slice(1)), beamTextIsFirst: b.text === b.hypotheses[0],
    language: g.language, tokens: [g.tokens, b.tokens], finish: [g.finishReason, b.finishReason], ms: { term: msTerm, greedy: g.cost.ms, beam: b.cost.ms },
    torchGreedyEqual: g.text === r.hyps[0], torchBeamEqual: eq(b.hypotheses, r.hyps.slice(1)), text: g.text,
  };
  out.push(row);
  console.log(r.id, row.greedyEqualsTerm ? "greedy=term" : "greedy!=TERM", row.beamEqualsTerm ? "beam=term" : "beam!=TERM", row.language, `ms term ${Math.round(msTerm)} greedy ${Math.round(g.cost.ms)} beam ${Math.round(b.cost.ms)}`, `torch ${row.torchGreedyEqual ? "=" : "!"}${row.torchBeamEqual ? "=" : "!"}`);
}
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const summary = {
  windows: out.length, greedyEqualsTerm: out.filter((x) => x.greedyEqualsTerm).length, beamEqualsTerm: out.filter((x) => x.beamEqualsTerm).length,
  languages: [...new Set(out.map((x) => x.language))], torchGreedyEqual: out.filter((x) => x.torchGreedyEqual).length, torchBeamEqual: out.filter((x) => x.torchBeamEqual).length,
  medianMs: { term: med(out.map((x) => x.ms.term)), greedy: med(out.map((x) => x.ms.greedy)), beam: med(out.map((x) => x.ms.beam)) },
};
console.log(JSON.stringify(summary, null, 1));
if (outFile) fs.writeFileSync(outFile, JSON.stringify({ summary, windows: out }, null, 1));
await model.dispose();
