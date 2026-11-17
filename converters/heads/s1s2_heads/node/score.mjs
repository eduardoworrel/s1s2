// Scores fidelity cases with s1s2's own readers (dist/index.js of this checkout), for s1s2_heads.fidelity.
// argv: <job.json> <inputs.f32> <out.json>. job = {head: {path, spec}, norm?: {path, spec}, cases: [...]} where each
// case points into the f32 file by {offset, rows, dim} blocks (float32 LE):
//   pointer: {kind: "pointer", hq, hks}        -> z_k = headScore(hq, hk)          (after norm when given)
//   linear:  {kind: "linear", type, n, h}      -> linearScores(h, type, n)
//   joint:   {kind: "joint", hidden, lexical, questions}  -> jointScores
//   logits:  {kind: "logits", L, ids, T}       -> L[ids] / T (the runtime's logits readout, src/decide.ts)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.LIB_S12_DIST ?? path.resolve(here, "../../../../dist/index.js");
const lib = await import(dist);
const [jobPath, binPath, outPath] = process.argv.slice(2);
const job = JSON.parse(fs.readFileSync(jobPath, "utf8"));
const raw = fs.readFileSync(binPath);
const f32 = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const block = (b) => Array.from({ length: b.rows }, (_, r) => f32.slice(b.offset + r * b.dim, b.offset + (r + 1) * b.dim));
const ab = (p) => { const b = fs.readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const head = job.head ? lib.loadAnyHead(ab(job.head.path), job.head.spec ?? {}) : null;
const norm = job.norm ? lib.loadNorm(ab(job.norm.path), job.norm.spec ?? {}) : null;
const nh = (h) => (norm ? lib.applyNorm(norm, h) : h);
const out = [];
for (const c of job.cases) {
  let z;
  if (c.kind === "pointer") { const hq = nh(block(c.hq)[0]); z = block(c.hks).map((hk) => lib.headScore(head, hq, nh(hk))); }
  else if (c.kind === "linear") z = lib.linearScores(head, nh(block(c.h)[0]), c.type, c.n);
  else if (c.kind === "joint") { const lex = block(c.lexical); z = lib.jointScores(head, block(c.hidden), (i) => lex[i], c.questions); }
  else if (c.kind === "logits") { const L = block(c.L)[0]; z = c.ids.map((i) => L[i] / (c.T ?? 1)); }
  else throw new Error("unknown case kind " + c.kind);
  const flat = Array.isArray(z[0]) ? z : [z];
  out.push({ scores: flat, probabilities: flat.map((zz) => lib.choiceFromScores("q", zz.map((_, i) => String(i)), zz, "head").probabilities.map((p) => p.probability)) });
}
fs.writeFileSync(outPath, JSON.stringify({ lib: dist, results: out }));
