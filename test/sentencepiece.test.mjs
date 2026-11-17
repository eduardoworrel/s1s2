// SentencePiece-style tokenizers (EuroLLM, Salamandra): a marker token such as '>' is a piece without '▁' that only
// appears after other text; encoding it alone gives '▁>'. Decoding inserts '▁' after special tokens, so offsets from
// incremental decoding do not reconstruct the prompt. A toy tokenizer with exactly these two properties checks that
// the lib verifies marker and label tokens in the context of the rendered prompt, and still refuses real mismatches.
import test from "node:test";
import assert from "node:assert/strict";
import { assertManifestTokens, encodeTagged, firstTokens, logitIds, normalizeQuestions, planQuestion, validateManifest, S12 } from "../dist/index.js";
import { toy, toyManifest, TOY_ID as ID, TOY_PIECES } from "./helpers.mjs";
const TOY_PIECES_LEN = TOY_PIECES.length;

const Q = (m, choices = [" Yes", " No"]) => normalizeQuestions([{ type: "choice", name: "q", instructions: "Is it so?", choices }], m)[0];

test("toy tokenizer behaves like SentencePiece: '>' alone is '▁>', decode is not invertible", () => {
  assert.deepEqual(toy.encode(">"), [ID.get("▁>")]);
  assert.equal(toy.encode("a </option>").at(-1), ID.get(">"));
  assert.notEqual(toy.decode(toy.encode("<|im_start|>system")), "<|im_start|>system");
  assert.equal(toy.encode(" Yes")[0], toy.encode(" No")[0], "labels alone start with the same bare '▁'");
});

test("load-time checks accept a marker piece without '▁' (checked in the rendered prompt)", () => {
  const m = validateManifest(toyManifest());
  assertManifestTokens(toy, m);
  assert.doesNotThrow(() => S12.fromBackend(m, "genai-int4-cpu", { kind: "stub", hiddenName: "hidden_states" }, toy));
});

test("read positions by prefix encoding land on the '>' of each marker", () => {
  const m = validateManifest(toyManifest());
  const e = encodeTagged(toy, m, Q(m), "We share data.");
  const gt = ID.get(">");
  assert.equal(e.ids[e.decide], gt);
  assert.equal(e.options.length, 2);
  for (const p of e.options) assert.equal(e.ids[p], gt);
  // the decide position is the last token of encode(prompt up to the end of " <decide>")
  const cut = e.text.indexOf(" <decide>") + " <decide>".length;
  assert.equal(e.decide, toy.encode(e.text.slice(0, cut)).length - 1);
  assert.ok(e.prefixLen > 0 && e.prefixLen < e.decide);
});

test("label tokens for the logits read come from the prompt context, not from the label alone", () => {
  const m = validateManifest(toyManifest());
  const e = encodeTagged(toy, m, Q(m), "We share data.");
  assert.throws(() => firstTokens(toy, [" Yes", " No"], "choice"), /same token/);
  assert.deepEqual(firstTokens(toy, [" Yes", " No"], "choice", e.text), [ID.get("▁Yes"), ID.get("▁No")]);
  assert.deepEqual(firstTokens(toy, [" Yes", " No"], "choice", e.text), [m.logits.predicate.yes.id, m.logits.predicate.no.id]);
});

test("real mismatches still fail: wrong marker id, wrong decoded text, token that never sits at the read position", () => {
  const a = toyManifest(); a.markers.decide.tokens = [{ id: ID.get("e"), text: ">" }];
  assert.throws(() => assertManifestTokens(toy, validateManifest(a)), /markers.decide token .* decodes to "e"/);
  // '▁>' decodes alone to ">" too, but it is not the token read in the prompt: the probe catches it
  const b = toyManifest(); b.markers.option.tokens = [{ id: ID.get("▁>"), text: ">" }];
  assert.throws(() => assertManifestTokens(toy, validateManifest(b)), /probe prompt.*option 0 read position/);
  const c = toyManifest(); c.logits.predicate.yes = { id: ID.get(">"), text: "Yes" };
  assert.throws(() => assertManifestTokens(toy, validateManifest(c)), /mismatch.*logits.predicate.yes/);
});

// Lab kev markers (S5-nano, Jurema): the option text is "Yes" (no space after the marker), but the lab reads the label at
// the cue as " Yes" (contract token). The first token of "Yes" after the prompt is another token ('Y' here, 9175 vs 7179 on
// Qwen); the manifest pins the contract ids per option label (logits.choice map "tokens") and decide reads those.
test("pinned choice tokens: decide reads the contract label ids, not the first token of the option text", () => {
  const m = toyManifest();
  m.logits.choice = { map: "tokens", tokens: [{ label: "Yes", id: ID.get("▁Yes"), text: "Yes" }, { label: "No", id: ID.get("▁No"), text: "No" }] };
  const v = validateManifest(m);
  assertManifestTokens(toy, v);
  const q = Q(v, ["No", "Yes"]);
  const e = encodeTagged(toy, v, q, "We share data.");
  // the old readout: first token of the option text after the prompt is not the contract token
  assert.deepEqual(firstTokens(toy, q.values, "choice", e.text), [ID.get("N"), ID.get("Y")]);
  assert.deepEqual(logitIds({ manifest: v, tok: toy }, q, e.text), [ID.get("▁No"), ID.get("▁Yes")]);
  // planQuestion reads the logits at the pinned ids
  const p = planQuestion({ manifest: v, tok: toy, backend: { kind: "stub" }, heads: {}, maxSafeTokens: 4096 }, q, "We share data.", "logits");
  const L = new Float32Array(TOY_PIECES_LEN).fill(-9); L[ID.get("▁Yes")] = 3; L[ID.get("N")] = 5; L[ID.get("Y")] = 5;
  const a = p.read({ lastLogits: L, hiddens: [] });
  assert.equal(a.choice, "Yes");
  assert.deepEqual(a.raw, [-9, 3]);
  // a choice outside the pinned labels is refused (no silent fallback to the first-token readout)
  assert.throws(() => logitIds({ manifest: v, tok: toy }, Q(v, [" Yes", " No"]), e.text), /no pinned logits token/);
});

test("pinned choice tokens: validation", () => {
  const bad = (tokens, re) => { const m = toyManifest(); m.logits.choice = { map: "tokens", tokens }; assert.throws(() => validateManifest(m), re); };
  bad([{ label: "Yes", id: 1, text: "Yes" }], /at least 2/);
  bad([{ label: "Yes", id: 1, text: "Yes" }, { label: "No", id: 1, text: "No" }], /ids must be distinct/);
  bad([{ label: "Yes", id: 1, text: "Yes" }, { label: "Yes", id: 2, text: "No" }], /labels must be distinct/);
  bad([{ id: 1, text: "Yes" }, { label: "No", id: 2, text: "No" }], /label: non-empty/);
  const m = toyManifest(); m.logits.choice = { map: "first-token", tokens: [] };
  assert.throws(() => validateManifest(m), /only with map "tokens"/);
  const w = toyManifest(); w.logits.choice = { map: "tokens", tokens: [{ label: "Yes", id: ID.get(">"), text: "Yes" }, { label: "No", id: ID.get("▁No"), text: "No" }] };
  assert.throws(() => assertManifestTokens(toy, validateManifest(w)), /mismatch.*logits.choice.tokens/);
});
