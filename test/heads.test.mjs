// Head types of spec 0.2 (section 6) with mocks: safetensors, pointer (safetensors fp32), linear (rows, softcap,
// temperature, read position) through decide(), joint (Clef) against a fixture computed by the reference code
// (converters/heads/tests/make_joint_fixture.py: sglang@da614ba JointSchemaHead), the joint prompt encoder.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  parseSafetensors, loadPointerSafetensors, loadLinearHead, linearScores, linearRaw, loadJointHead, jointScores,
  encodeJointSchema, jointOptions, pyJson, loadAnyHead, headScore, S12, validateManifest,
} from "../dist/index.js";
import { toy, toyManifest, clone } from "./helpers.mjs";

/** safetensors bytes from {name: {dtype, shape, data: number[]}} (F32 / F16 / BF16). */
function st(tensors, metadata) {
  const header = {}; const parts = []; let off = 0;
  if (metadata) header.__metadata__ = metadata;
  for (const [n, t] of Object.entries(tensors)) {
    const w = t.dtype === "F32" ? 4 : 2, b = Buffer.alloc(t.data.length * w);
    t.data.forEach((x, i) => {
      if (t.dtype === "F32") b.writeFloatLE(x, i * w);
      else if (t.dtype === "BF16") { const f = Buffer.alloc(4); f.writeFloatLE(x); b.writeUInt16LE(f.readUInt16LE(2), i * w); }
      else { const f = new Float32Array([x]); const u = new Uint32Array(f.buffer)[0]; b.writeUInt16LE(((u >>> 16) & 0x8000) | ((((u >>> 23) & 0xff) - 112) << 10) | ((u >>> 13) & 0x3ff), i * w); }
    });
    header[n] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + b.length] }; off += b.length; parts.push(b);
  }
  const h = Buffer.from(JSON.stringify(header)), len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(h.length));
  const all = Buffer.concat([len, h, ...parts]);
  return all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength);
}

test("safetensors: F32, F16, BF16 widened exactly for representable values; shape/size errors", () => {
  const s = parseSafetensors(st({ a: { dtype: "F32", shape: [2], data: [1.5, -2] }, b: { dtype: "BF16", shape: [2], data: [0.5, 3] }, c: { dtype: "F16", shape: [1], data: [0.25] } }, { k: "v" }));
  assert.deepEqual([...s.tensors.a.data], [1.5, -2]); assert.deepEqual([...s.tensors.b.data], [0.5, 3]); assert.deepEqual([...s.tensors.c.data], [0.25]);
  assert.equal(s.metadata.k, "v");
  assert.throws(() => parseSafetensors(st({ a: { dtype: "I8", shape: [1], data: [1] } })), /I8 not supported|bytes for/);
});

test("pointer head from safetensors: fp32 weights, scale default 1/sqrt(D), metadata temperature", () => {
  // D = 4, H = 2: q = rows of [1, 0] -> q(h) = [h0, h0, h0, h0]; k likewise from h1.
  const q = [1, 0, 1, 0, 1, 0, 1, 0], k = [0, 1, 0, 1, 0, 1, 0, 1];
  const buf = st({ "q.weight": { dtype: "F32", shape: [4, 2], data: q }, "q.bias": { dtype: "F32", shape: [4], data: [0, 0, 0, 0] }, "k.weight": { dtype: "F32", shape: [4, 2], data: k }, "k.bias": { dtype: "F32", shape: [4], data: [0, 0, 0, 0] } }, { temperature: "2" });
  const hd = loadPointerSafetensors(buf);
  assert.equal(hd.scale, 0.5); assert.equal(hd.temperature, 2);
  // z = scale * 4 * h0 * h1' / T = 0.5 * 4 * 3 * 5 / 2 = 15
  assert.equal(headScore(hd, new Float32Array([3, 0]), new Float32Array([0, 5])), 15);
  assert.equal(loadAnyHead(buf, { kind: "pointer", format: "safetensors", scale: 1 }).scale, 1);
});

const LIN = () => st({ "proj.weight": { dtype: "F32", shape: [5, 2], data: [1, 0, 0, 1, 2, 0, 0, 2, 1, 1] }, "proj.bias": { dtype: "F32", shape: [5], data: [0, 0, 0, 0, 0.5] } });

test("linear head: rows per type, bias, softcap, temperature (JEV / pplx semantics)", () => {
  const hd = loadLinearHead(LIN(), { tensors: { weight: "proj.weight", bias: "proj.bias" }, rows: { predicate: { start: 0, end: 2 }, choice: { start: 2 } }, softcap: 2, temperature: { choice: 0.5 } });
  const h = new Float32Array([1, 3]);
  // predicate rows 0..1: [1, 3] softcapped: 2 tanh(z/2)
  assert.deepEqual(linearRaw(hd, h, "predicate", 2), [2 * Math.tanh(0.5), 2 * Math.tanh(1.5)]);
  // choice rows 2..4: [2, 6, 4.5] -> softcap -> / 0.5
  const z = linearScores(hd, h, "choice", 3);
  [2, 6, 4.5].forEach((v, i) => assert.ok(Math.abs(z[i] - 2 * Math.tanh(v / 2) / 0.5) < 1e-12));
  assert.throws(() => linearScores(hd, h, "choice", 4), /at most 3/);
  assert.throws(() => linearScores(hd, h, "score", 2), /no rows for question type score/);
  // no rows given: every type reads from row 0 (pplx / d3 codes)
  const all = loadLinearHead(LIN(), { tensors: { weight: "proj.weight", bias: null } });
  assert.deepEqual(linearScores(all, h, "score", 2), [1, 3]);
});

test("linear head through decide(): one read position per question, probabilities = softmax(rows / T)", async () => {
  const m = toyManifest();
  m.hiddenSize = 2; m.defaults.method = { predicate: "head", choice: "head", score: "head" };
  validateManifest(m);
  const hd = loadAnyHead(LIN(), { kind: "linear", tensors: { weight: "proj.weight", bias: "proj.bias" }, rows: { predicate: { start: 0, end: 2 }, choice: { start: 2 }, score: { start: 2 } }, read: "decide" });
  const seen = [];
  const be = { kind: "fake", hiddenName: "hidden", async prefill(ids, want) { seen.push(want.positions); return { hiddens: want.positions.map(() => new Float32Array([1, 3])), lastLogits: new Float32Array(200) }; } };
  const s = S12.fromBackend(m, "genai-int4-cpu", be, toy, { predicate: hd, choice: hd, score: hd });
  const r = await s.decide({ input: "We share data.", questions: [
    { name: "p", type: "predicate", instructions: "Shared?" },
    { name: "c", type: "choice", instructions: "Which?", choices: ["a", "b", "c"] },
    { name: "s", type: "score", instructions: "How much?", levels: ["low", "high"] }] });
  for (const p of seen) assert.equal(p.length, 1, "linear heads read one position");
  const sm = (xs) => { const e = xs.map((x) => Math.exp(x - Math.max(...xs))), z = e.reduce((a, b) => a + b); return e.map((v) => v / z); };
  const [p, c, sc] = r.answers;
  assert.ok(Math.abs(p.probability - sm([1, 3])[1]) < 1e-12);
  assert.equal(c.choice, "b"); assert.ok(Math.abs(c.probabilities[1].probability - sm([2, 6, 4.5])[1]) < 1e-12);
  assert.ok(Math.abs(sc.score - sm([2, 6])[1]) < 1e-12); assert.equal(sc.method, "head");
  await r.dispose?.();
});

test("joint head (Clef) = reference JointSchemaHead on a tiny random head (fixture from sglang@da614ba)", () => {
  const dir = new URL("./fixtures/joint-tiny/", import.meta.url);
  const c = JSON.parse(readFileSync(new URL("case.json", dir), "utf8"));
  const b = readFileSync(new URL("joint_head.safetensors", dir));
  const hd = loadJointHead(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), c.config);
  const lex = c.lexical.map((r) => Float32Array.from(r));
  const z = jointScores(hd, c.hidden.map((r) => Float32Array.from(r)), (i) => lex[i], c.questions).flat();
  assert.equal(z.length, c.expected.length);
  z.forEach((v, i) => assert.ok(Math.abs(v - c.expected[i]) < 1e-5, `logit ${i}: ${v} vs ${c.expected[i]}`));
  assert.throws(() => jointScores(hd, c.hidden.map((r) => Float32Array.from(r)), (i) => lex[i], [{ type: 1, span: [3, 99], options: [[1, 2]] }]), /outside the prompt/);
});

test("joint prompt encoder: pieces encoded one by one, spans in token positions, Clef option order", () => {
  const enc = (s) => Array.from(s, (ch) => ch.codePointAt(0)); // one token per character: spans are character spans
  const qs = [{ id: "tone", type: "choice", instructions: "What tone?", criteria: { pushy: "Pressures", calm: null } }, { id: "spam", type: "noul" }];
  const { ids, questions } = encodeJointSchema(enc, { msg: "hi" }, qs);
  const text = String.fromCodePoint(...ids);
  const sp = ([a, b]) => text.slice(a, b);
  assert.equal(sp(questions[0].span), "What tone?");
  assert.deepEqual(questions[0].options.map(sp), ['{"option_id":"calm"}', '{"description":"Pressures","option_id":"pushy"}']);
  assert.equal(sp(questions[1].span), "spam", "empty instructions fall back to the question id");
  assert.deepEqual(questions[1].options.map(sp), ['{"description":"The proposition is true or the answer is yes.","option_id":"true"}', '{"description":"The proposition is false or the answer is no.","option_id":"false"}']);
  assert.ok(text.includes('STATE:\n{"msg":"hi"}\n\nSCHEMA FIELDS:\n\nFIELD 1\nID: tone\nTYPE: choice\nINSTRUCTION: What tone?\nALLOWED OPTIONS:\nOPTION 1: '));
  assert.ok(text.endsWith("JOINT SCHEMA DECISIONS:"));
  assert.deepEqual(jointOptions({ type: "noul", criteria: { true: null } }).map((o) => o[1]), [null, "The proposition is false or the answer is no."]);
  assert.equal(pyJson({ b: [1, "x"], a: null }), '{"a":null,"b":[1,"x"]}');
});

test("manifest: head kinds, formats, layer, read and softcap are validated", () => {
  const m = toyManifest();
  const b = m.builds["genai-int4-cpu"];
  const file = { path: "heads/h.safetensors", sha256: "0".repeat(64) };
  b.heads = { h: { file, types: ["choice"], kind: "linear", format: "safetensors", read: "last", softcap: 30, layer: "final" } };
  validateManifest(clone(m));
  for (const [k, v, re] of [["kind", "mlp", /kind/], ["format", "npz", /format/], ["read", "first", /read/], ["softcap", 0, /softcap/], ["layer", -1, /layer/]]) {
    const x = clone(m); x.builds["genai-int4-cpu"].heads.h[k] = v;
    assert.throws(() => validateManifest(x), re);
  }
  const j = clone(m); j.builds["genai-int4-cpu"].heads.h = { file, types: ["choice"], kind: "joint" };
  assert.throws(() => validateManifest(j), /config/);
  const hb = clone(m); hb.builds["genai-int4-cpu"].heads.h = { file, types: ["choice"], kind: "linear", format: "head.bin" };
  assert.throws(() => validateManifest(hb), /head.bin holds pointer heads only/);
  assert.throws(() => loadAnyHead(LIN(), { kind: "linear", layer: 19 }), /layer 19/);
});
