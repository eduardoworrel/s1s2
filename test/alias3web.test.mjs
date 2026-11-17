// head.bin compatibility with the published alias3-web reader (spec 6.1, 6.5): s1s2 reads every head.bin with
// alias3-web's semantics (scale = header.scale ?? 1/sqrt(headDim); RMSNorm from norm.bin, no eps) unless the header or
// the manifest states otherwise, and matches the vendored reference reader (test/reference/alias3web-2ab2e60.mjs).
// Real-file check: ALIAS3WEB_DIR=<dir holding heads/head.bin and heads/norm.bin of aliasrobotics/alias3-web@2ab2e60>.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import * as ref from "./reference/alias3web-2ab2e60.mjs";
import { loadHead, headScore, choiceFromScores, loadNorm, applyNorm } from "../dist/index.js";

function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 2 ** 32; }; }
const gauss = (r) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

/** fp32 -> fp16 bits (round to nearest even), enough for test weights in normal range. */
function f16bits(x) {
  const f = new Float32Array([x]), u = new Uint32Array(f.buffer)[0];
  const sign = (u >>> 16) & 0x8000, e = ((u >>> 23) & 0xff) - 127 + 15, m = u & 0x7fffff;
  if (e <= 0) return sign; // flush tiny values (not used by these tests)
  if (e >= 31) return sign | 0x7c00;
  let h = sign | (e << 10) | (m >>> 13);
  const rest = m & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && (h & 1))) h++;
  return h;
}
function headBin(meta, q, qb, k, kb) {
  const j = Buffer.from(JSON.stringify(meta));
  const arrs = [q, qb, k, kb], n = arrs.reduce((a, x) => a + x.length, 0);
  const b = Buffer.alloc(4 + j.length + 2 * n);
  b.writeUInt32LE(j.length, 0); j.copy(b, 4);
  let off = 4 + j.length;
  for (const a of arrs) for (const x of a) { b.writeUInt16LE(f16bits(x), off); off += 2; }
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
function normBin(w) {
  const j = Buffer.from(JSON.stringify({ kind: "final_rmsnorm", hiddenSize: w.length }));
  const b = Buffer.alloc(4 + j.length + 4 * w.length);
  b.writeUInt32LE(j.length, 0); j.copy(b, 4);
  w.forEach((x, i) => b.writeFloatLE(x, 4 + j.length + 4 * i));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
const randArr = (r, n, s = 1) => Float32Array.from({ length: n }, () => gauss(r) * s);
const ours = (hd, nw, hD, hOs) => choiceFromScores("q", hOs.map((_, i) => String(i)), hOs.map((h) => headScore(hd, applyNorm(nw, hD), applyNorm(nw, h))), "head").probabilities.map((p) => p.probability);
const theirs = (hd, w, hD, hOs) => ref.softmax(hOs.map((h) => ref.headLogit(hd, ref.aliasNorm(w, hD), ref.aliasNorm(w, h))));

test("alias3-web layout without a scale key: s1s2 = alias3-web reader (1/sqrt(headDim), out-of-graph RMSNorm)", async () => {
  const r = rng(7), D = 4, H = 8;
  const buf = headBin({ headDim: D, hiddenSize: H, temperature: 1.0, score_offset: 0.0 }, randArr(r, D * H), randArr(r, D), randArr(r, D * H), randArr(r, D));
  const nbuf = normBin(randArr(r, H, 0.5));
  const mine = loadHead(buf), nw = loadNorm(nbuf), rh = ref.loadHead(buf), rw = await ref.loadNorm(nbuf);
  assert.equal(mine.scale, 1 / Math.sqrt(D));
  for (let t = 0; t < 50; t++) {
    const hD = randArr(r, H, 3), hOs = Array.from({ length: 2 + (t % 4) }, () => randArr(r, H, 3));
    assert.deepEqual(ours(mine, nw, hD, hOs), theirs(rh, rw, hD, hOs));
  }
});

test("plain-text-marker heads (1/sqrt(d) folded, scale_folded_into_q) and ct0 heads (folded, no key + manifest scale 1) keep their meaning", () => {
  const r = rng(11), D = 4, H = 8, s = 1 / Math.sqrt(D);
  const q = randArr(r, D * H), qb = randArr(r, D), k = randArr(r, D * H), kb = randArr(r, D);
  const fold = (a) => a.map((x) => x * s); // s = 1/2 is exact in fp16
  const unfolded = loadHead(headBin({ headDim: D, hiddenSize: H, temperature: 1 }, q, qb, k, kb));
  const lab = loadHead(headBin({ headDim: D, hiddenSize: H, temperature: 1, scale_folded_into_q: s }, fold(q), fold(qb), k, kb));
  const ct0 = loadHead(headBin({ headDim: D, hiddenSize: H, temperature: 1, score_offset: 0 }, fold(q), fold(qb), k, kb), 1);
  const written = headBin({ headDim: D, hiddenSize: H, temperature: 1, scale: 1, scale_folded_into_q: s }, fold(q), fold(qb), k, kb);
  assert.equal(lab.scale, 1); assert.equal(ct0.scale, 1); assert.equal(loadHead(written).scale, 1);
  const rw = ref.loadHead(written); // a head written with an explicit scale: the alias3-web reader agrees too
  for (let t = 0; t < 20; t++) {
    const a = randArr(r, H), b = randArr(r, H), z = headScore(unfolded, a, b);
    assert.ok(Math.abs(headScore(lab, a, b) - z) < 1e-9 * Math.max(1, Math.abs(z)));
    assert.ok(Math.abs(headScore(ct0, a, b) - z) < 1e-9 * Math.max(1, Math.abs(z)));
    assert.equal(ref.headLogit(rw, a, b), headScore(loadHead(written), a, b));
  }
  // Without the manifest's scale the same ct0 bytes read 1/sqrt(d) too small: the case the manifest field exists for.
  const bare = loadHead(headBin({ headDim: D, hiddenSize: H, temperature: 1, score_offset: 0 }, fold(q), fold(qb), k, kb));
  assert.equal(bare.scale, s);
});

test("norm: eps and Gemma-style (1 + w) weights; size checks", () => {
  const w = new Float32Array([0, 1, 2, -1]);
  const nb = normBin(w);
  const h = new Float32Array([1, 2, 3, 4]), ms = (1 + 4 + 9 + 16) / 4;
  const a = applyNorm(loadNorm(nb), h);
  assert.ok(Math.abs(a[2] - (3 / Math.sqrt(ms)) * 2) < 1e-6);
  const g = applyNorm(loadNorm(nb, { eps: 1e-6, weight: "1+w" }), h);
  assert.ok(Math.abs(g[2] - (3 / Math.sqrt(ms + 1e-6)) * 3) < 1e-6);
  assert.throws(() => applyNorm(loadNorm(nb), new Float32Array(3)), /hidden size 3 != 4/);
  assert.throws(() => loadNorm(nb.slice(0, nb.byteLength - 4)), /bytes, expected/);
});

const dir = process.env.ALIAS3WEB_DIR;
const real = dir && existsSync(join(dir, "heads/head.bin")) && existsSync(join(dir, "heads/norm.bin"));
test("real alias3-web@2ab2e60 head.bin + norm.bin: same probabilities and decisions as its reader", { skip: real ? false : "set ALIAS3WEB_DIR to run" }, async () => {
  const ab = (p) => { const b = readFileSync(join(dir, p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
  const hb = ab("heads/head.bin"), nb = ab("heads/norm.bin");
  const mine = loadHead(hb), nw = loadNorm(nb), rh = ref.loadHead(hb), rw = await ref.loadNorm(nb);
  assert.equal(mine.headDim, 256); assert.equal(mine.hiddenSize, 2560); assert.equal(mine.scale, 1 / 16);
  const r = rng(2026);
  let maxDp = 0;
  for (let t = 0; t < 32; t++) {
    const hD = randArr(r, 2560, 2), hOs = Array.from({ length: 2 + (t % 4) }, () => randArr(r, 2560, 2));
    const p = ours(mine, nw, hD, hOs), q = theirs(rh, rw, hD, hOs);
    p.forEach((x, i) => { maxDp = Math.max(maxDp, Math.abs(x - q[i])); });
    assert.equal(p.indexOf(Math.max(...p)), q.indexOf(Math.max(...q)));
  }
  assert.equal(maxDp, 0);
});
