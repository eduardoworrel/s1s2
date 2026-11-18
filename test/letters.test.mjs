// Choice by label logits with logits.choice.map "letters": the option keys must be read as the model writes them after
// the cue. After "Answer:" a Qwen-style tokenizer writes " A" as one token (Qwen3: " A" = 362), so reading the bare "A"
// puts the readout on a token the model does not write there (seen on the LiteRT / WebLLM SIB-200 builds, which had to
// pin " A", " B", ... per task). A tokenizer that writes the space as its own token (SmolLM2/3) gets the lead " ".
import test from "node:test";
import assert from "node:assert/strict";
import { S1S2, planQuestion, normalizeQuestions, validateManifest, letterLayout, logitIds, encodeTagged } from "../dist/index.js";
import { toy, toyManifest, makeToy, TOY_PIECES, TOY_ID, ct0Tokenizer, CT0, clone } from "./helpers.mjs";

const KEYS = ["A", "B", "C"];
const QPIECES = [...TOY_PIECES, ...KEYS.map((k) => "▁" + k)];
const qtok = makeToy(QPIECES, { keepLeadingSpace: true }); // " A" is one token, like Qwen2/Qwen3
const QID = new Map(QPIECES.map((p, i) => [p, i]));
const COLON = TOY_ID.get(":"), SPACE = TOY_ID.get("▁");
const VALUES = ["cat", "dog", "bird"];
const freeQ = { name: "pet", type: "choice", instructions: "Which pet?", options: VALUES };

const lettersManifest = (task) => {
  const m = toyManifest();
  m.logits.choice = { map: "letters" };
  if (task) m.tasks = { pet: { type: "choice", instructions: "Which pet?", choices: task } };
  return validateManifest(m);
};
const engine = (m, tok) => ({ manifest: m, tok, backend: { kind: "stub", hiddenName: "hidden" }, heads: {}, maxSafeTokens: 4096 });
// The model writes " B" right after the cue (one token), and "B" after a lone space; the bare "B" after the cue gets nothing.
const fake = (seen, tok, idOf) => ({ kind: "fake", hiddenName: "hidden", async prefill(ids, want) {
  seen.push(ids);
  const L = new Float32Array(QPIECES.length);
  if (ids.at(-1) === COLON && idOf.get("▁B") !== undefined) L[idOf.get("▁B")] = 12;
  else if (ids.at(-1) === SPACE) L[idOf.get("B")] = 12;
  return { hiddens: want.positions.map(() => new Float32Array([1, 3])), lastLogits: L };
} });

test("letters, Qwen-style tokenizer: free-form choice reads ' A', ' B', ... after the cue (no lead)", async () => {
  const m = lettersManifest();
  const q = normalizeQuestions([freeQ], m)[0];
  const plain = encodeTagged(qtok, m, q, "I like barking.");
  assert.equal(plain.ids.at(-1), COLON);
  assert.deepEqual(letterLayout(qtok, q, plain.text), { lead: "", labels: [" A", " B", " C"] });
  const p = planQuestion(engine(m, qtok), q, "I like barking.", "logits");
  assert.deepEqual(p.enc.ids, plain.ids, "the prompt is not extended");
  assert.deepEqual(logitIds(engine(m, qtok), q, plain.text), KEYS.map((k) => QID.get("▁" + k)));
  const seen = [];
  const s = S1S2.fromBackend(m, "genai-int4-cpu", fake(seen, qtok, QID), qtok);
  const r = await s.decide({ input: "I like barking.", questions: [freeQ] }, { method: "logits" });
  assert.equal(r.answers[0].choice, "dog");
  assert.ok(r.answers[0].confidence > 0.99);
});

test("letters: a task without pinned tokens reads the same ids a task pinning ' A', ' B', ' C' does", async () => {
  const pinned = lettersManifest(VALUES.map((v, i) => ({ value: v, token: { id: QID.get("▁" + KEYS[i]), text: " " + KEYS[i] } })));
  const plain = lettersManifest(VALUES);
  const ids = (m) => { const q = normalizeQuestions(["pet"], m)[0]; return planQuestion(engine(m, qtok), q, "I like barking.", "logits"); };
  assert.deepEqual(ids(plain).enc.ids, ids(pinned).enc.ids);
  const q = normalizeQuestions(["pet"], plain)[0];
  assert.deepEqual(logitIds(engine(plain, qtok), q, ids(plain).enc.text), KEYS.map((k) => QID.get("▁" + k)));
  for (const m of [plain, pinned]) {
    const s = S1S2.fromBackend(m, "genai-int4-cpu", fake([], qtok, QID), qtok);
    const r = await s.decide({ input: "I like barking.", questions: ["pet"] }, { method: "logits" });
    assert.equal(r.answers[0].choice, "dog");
  }
});

test("letters, SmolLM-style tokenizer (' A' = [space, 'A']): the lead ' ' is appended and 'A' is read after it", async () => {
  const m = lettersManifest();
  const q = normalizeQuestions([freeQ], m)[0];
  const plain = encodeTagged(toy, m, q, "I like barking.");
  assert.deepEqual(letterLayout(toy, q, plain.text), { lead: " ", labels: KEYS });
  const p = planQuestion(engine(m, toy), q, "I like barking.", "logits");
  assert.deepEqual(p.enc.ids, [...plain.ids, SPACE]);
  assert.equal(p.enc.decide, plain.decide, "head read positions do not move");
  const seen = [];
  const s = S1S2.fromBackend(m, "genai-int4-cpu", fake(seen, toy, TOY_ID), toy);
  const r = await s.decide({ input: "I like barking.", questions: [freeQ] }, { method: "logits" });
  assert.equal(seen[0].at(-1), SPACE);
  assert.equal(r.answers[0].choice, "dog");
});

test("letters: a prompt ending in whitespace reads the bare keys; pinned task tokens keep precedence", () => {
  const m = lettersManifest();
  const q = normalizeQuestions([freeQ], m)[0];
  const plain = encodeTagged(qtok, m, q, "x");
  assert.deepEqual(letterLayout(qtok, q, plain.text + "\n"), { lead: "", labels: KEYS });
  const pinned = lettersManifest(VALUES.map((v, i) => ({ value: v, token: { id: TOY_ID.get(KEYS[i]), text: KEYS[i] } })));
  const pq = normalizeQuestions(["pet"], pinned)[0];
  const p = planQuestion(engine(pinned, qtok), pq, "x", "logits");
  assert.deepEqual(logitIds(engine(pinned, qtok), pq, p.enc.text), KEYS.map((k) => TOY_ID.get(k)));
});

test("letters with the real CT0 (Qwen3) tokenizer: ' A' after 'Answer:'", async (t) => {
  const tok = await ct0Tokenizer();
  if (!tok) return t.skip("tokenizer unavailable (offline)");
  const m0 = clone(CT0); m0.logits.choice = { map: "letters" };
  const m = validateManifest(m0);
  const q = normalizeQuestions([freeQ], m)[0];
  const ctx = encodeTagged(tok, m, q, "I like barking.").text + "Answer:"; // the SIB-200 specialists' cue
  assert.deepEqual(letterLayout(tok, q, ctx), { lead: "", labels: [" A", " B", " C"] });
  assert.deepEqual(logitIds(engine(m, tok), q, ctx).map((i) => tok.decode([i])), [" A", " B", " C"]);
});
