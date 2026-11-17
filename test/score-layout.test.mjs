// Score questions read by label logits (study 012 of lab-s12): SmolLM2/3 tokenizers write " 1" as [" ", "1"], so after
// a prompt ending in "Answer:" the model's next token is the space and the digits carry ~1e-5 of the probability. The
// default layout reads the level tokens after that space; the hidden-state read position of a head does not move.
import test from "node:test";
import assert from "node:assert/strict";
import { S1S2, planQuestion, normalizeQuestions, validateManifest, scoreLayout, appendLead, encodeTagged, loadAnyHead } from "../dist/index.js";
import { toy, toyManifest, TOY_ID as ID, TOY_PIECES } from "./helpers.mjs";

const LEVELS = ["1", "2", "3", "4", "5"];
const SPACE = ID.get("▁"), COLON = ID.get(":"), DIGITS = LEVELS.map((d) => ID.get(d));

// A model like the ones of study 012: after the cue it writes a space first; after the space, the digit "4".
function modelLogits(ids) {
  const L = new Float32Array(TOY_PIECES.length);
  if (ids.at(-1) === COLON) L[SPACE] = 12;
  else if (ids.at(-1) === SPACE) L[DIGITS[3]] = 14;
  return L;
}
const labelMass = (L, ids) => { const mx = Math.max(...L), z = L.reduce((a, x) => a + Math.exp(x - mx), 0); return ids.reduce((a, i) => a + Math.exp(L[i] - mx), 0) / z; };
const fake = (seen) => ({ kind: "fake", hiddenName: "hidden", async prefill(ids, want) {
  seen.push({ ids, positions: want.positions });
  return { hiddens: want.positions.map(() => new Float32Array([1, 3])), lastLogits: modelLogits(ids) };
} });
// The toy manifest copies the shipped SmolLM2 manifest, which pins lead " "; `auto()` drops it to test the default rule.
const auto = () => { const m = toyManifest(); delete m.logits.score.lead; return validateManifest(m); };
const sevQ = { name: "sev", type: "score", instructions: "Rate it from 1 to 5.", levels: LEVELS };

test("toy tokenizer writes ' 1' as [space, '1'], like SmolLM2 and SmolLM3", () => {
  assert.deepEqual(toy.encode("Answer: 1").slice(-3), [COLON, SPACE, DIGITS[0]]);
});

test("the shipped layout (no lead) reads the digit where the model writes a space: label mass ~1e-5", async () => {
  const m = toyManifest(); m.logits.score = { map: "first-token", lead: "" };
  const seen = [];
  const s = S1S2.fromBackend(validateManifest(m), "genai-int4-cpu", fake(seen), toy);
  const r = await s.decide({ input: "Great product.", questions: [sevQ] }, { method: "logits" });
  assert.equal(seen[0].ids.at(-1), COLON);
  assert.ok(labelMass(modelLogits(seen[0].ids), DIGITS) < 1e-4);
  assert.notEqual(r.answers[0].level, "4");
});

test("default score layout reads the level after the space: the digit holds the mass and the level is right", async () => {
  assert.equal(toyManifest().logits.score.lead, " ", "the shipped SmolLM2 manifests pin the space");
  const m = auto();
  const seen = [];
  const s = S1S2.fromBackend(m, "genai-int4-cpu", fake(seen), toy);
  const r = await s.decide({ input: "Great product.", questions: [sevQ] }, { method: "logits" });
  assert.deepEqual(seen[0].ids.slice(-2), [COLON, SPACE], "the prompt ends with the space token");
  assert.ok(labelMass(modelLogits(seen[0].ids), DIGITS) > 0.99);
  assert.equal(r.answers[0].level, "4");
  assert.ok(r.answers[0].confidence > 0.99);
});

test("the lead moves only the logits: head read positions stay on the original prompt", () => {
  const m = auto();
  const q = normalizeQuestions([sevQ], m)[0];
  const e = { manifest: m, tok: toy, backend: { kind: "stub", hiddenName: "hidden" }, heads: {}, maxSafeTokens: 4096 };
  const plain = encodeTagged(toy, m, q, "Great product.");
  const p = planQuestion(e, q, "Great product.", "logits");
  assert.equal(p.enc.ids.length, plain.ids.length + 1);
  assert.equal(p.enc.last, plain.last);
  assert.equal(p.enc.ids[p.enc.last], COLON);
  assert.deepEqual(scoreLayout(toy, m, q, plain.text), { lead: " ", labels: LEVELS });
  // a prompt that already ends in whitespace gets no lead; an explicit lead "" disables it
  assert.equal(scoreLayout(toy, m, q, plain.text + "\n").lead, "");
  const off = toyManifest(); off.logits.score = { map: "first-token", lead: "" };
  assert.equal(scoreLayout(toy, validateManifest(off), q, plain.text).lead, "");
  assert.deepEqual(appendLead(toy, plain, "").ids, plain.ids);
});

test("method both with a linear score head: the head reads the cue token, the logits read after the space", async () => {
  const m = toyManifest(); m.hiddenSize = 2;
  const W = { "w.weight": { dtype: "F32", shape: [5, 2], data: [1, 0, 0, 1, 1, 1, 2, 0, 0, 2] } };
  const st = (obj) => { const hdr = {}; let off = 0; const parts = [];
    for (const [k, v] of Object.entries(obj)) { const b = Buffer.from(new Float32Array(v.data).buffer); hdr[k] = { dtype: v.dtype, shape: v.shape, data_offsets: [off, off + b.length] }; off += b.length; parts.push(b); }
    const h = Buffer.from(JSON.stringify(hdr)); const n = Buffer.alloc(8); n.writeBigUInt64LE(BigInt(h.length));
    const all = Buffer.concat([n, h, ...parts]); return all.buffer.slice(all.byteOffset, all.byteOffset + all.length); };
  const hd = loadAnyHead(st(W), { kind: "linear", tensors: { weight: "w.weight" }, rows: { score: { start: 0 } }, read: "last" });
  const seen = [];
  const s = S1S2.fromBackend(validateManifest(m), "genai-int4-cpu", fake(seen), toy, { score: hd });
  const r = await s.decide({ input: "Great product.", questions: [sevQ] }, { method: "both" });
  const { ids, positions } = seen[0];
  assert.equal(ids.at(-1), SPACE);
  assert.deepEqual(positions, [ids.length - 2], "head reads the hidden state at the cue, as it was fitted");
  assert.equal(r.answers[0].method, "head");
  assert.equal(r.answers[0].logits.level, "4");
});
