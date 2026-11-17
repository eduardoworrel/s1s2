import test from "node:test";
import assert from "node:assert/strict";
import { S1S2, normalizeQuestions, inputToText, validateManifest, decisionOf } from "../dist/index.js";
import { CT0, clone, fixture, ct0Tokenizer } from "./helpers.mjs";

const tok = await ct0Tokenizer();
const skip = tok ? false : "CT0 tokenizer unavailable (offline)";
const fx = fixture("ask_expected.json");
const C = Object.fromEntries(fx.cases.map((c) => [c.name, c]));
const mkHead = (j) => ({ q: Float32Array.from(j.q), qb: Float32Array.from(j.qb), k: Float32Array.from(j.k), kb: Float32Array.from(j.kb),
  headDim: j.headDim, hiddenSize: j.hiddenSize, temperature: j.temperature, scoreOffset: j.scoreOffset });
const heads = { predicate: mkHead(fx.headNoulChoice), choice: mkHead(fx.headNoulChoice), score: mkHead(fx.headScore) };
const near = (a, b, t = 1e-4) => Math.abs(a - b) <= t * Math.max(1, Math.abs(b));
const sig = (z) => 1 / (1 + Math.exp(-z));
const softmax = (xs) => { const m = Math.max(...xs), e = xs.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };

// Manifest with the fixture's tiny hidden size (heads were made for hiddenSize 16).
const M16 = (() => { const m = clone(CT0); m.hiddenSize = 16; return validateManifest(m); })();

/** Fake backend replaying a fixture case; optional state keeping to exercise the session path. */
function fake(c, { keep = false } = {}) {
  const calls = [];
  const be = {
    calls, kind: "fake", hiddenName: "hidden",
    async prefill(ids, want, o = {}) {
      calls.push({ op: "prefill", n: ids.length, positions: want.positions, keep: !!o.keep });
      assert.deepEqual(ids, c.ids, "prompt ids == fixture ids");
      const L = new Float32Array(248320);
      for (const [k, v] of Object.entries(c.logits)) L[+k] = v;
      const r = { hiddens: want.positions.map((i) => Float32Array.from(c.hseq[i])), lastLogits: L };
      if (keep && o.keep) r.state = { ids: ids.slice(), dispose() { calls.push({ op: "dispose" }); } };
      return r;
    },
    async dispose() {},
    async *generate(ids) { calls.push({ op: "generate", n: ids.length }); yield* [9175, 13]; },
  };
  if (keep) {
    be.snapshot = async (ids) => ({ ids, dispose() {} });
    be.prefillFrom = async () => { throw new Error("unused"); };
    be.generateFrom = async function* (h, ids) { calls.push({ op: "generateFrom", from: h.ids.length, n: ids.length, head: ids.slice(0, h.ids.length) }); yield* [9175, 13]; };
  }
  return be;
}
const lib = (c, o) => S1S2.fromBackend(M16, "decode-only-q4", fake(c, o), tok, heads);

test("predicate (alias noul): head and logits reads match ct0-edge reference numbers", { skip }, async () => {
  const c = C.noul, e = c.expected;
  for (const type of ["noul", "predicate"]) {
    const s = await lib(c).decide({ input: c.probe.state, questions: [{ name: "ioc", type, instructions: c.probe.q }] }, { method: "head" });
    const a = s.answers[0];
    assert.equal(a.type, "predicate"); assert.equal(a.name, "ioc"); assert.equal(a.method, "head");
    assert.ok(near(a.raw, e.raw) && near(a.probability, e.p_head));
    assert.ok(near(a.confidence, Math.max(e.p_head, 1 - e.p_head)));
  }
  const l = (await lib(c).decide({ input: c.probe.state, questions: [{ name: "ioc", type: "predicate", instructions: c.probe.q }] })).answers[0];
  assert.equal(l.method, "logits", "manifest default for predicate = logits");
  assert.ok(near(l.probability, e.p_logits));
  const both = (await lib(c).decide({ input: c.probe.state, questions: [{ name: "ioc", type: "predicate", instructions: c.probe.q, method: "both" }] })).answers[0];
  assert.equal(both.method, "head"); assert.ok(near(both.logits.probability, e.p_logits));
});

test("choice: Decisions shape, head and logits match reference scores", { skip }, async () => {
  const c = C.choice, e = c.expected;
  const req = { input: c.probe.state, questions: [{ name: "cls", type: "choice", choices: c.probe.options.map((value) => ({ value })) }] };
  const h = (await lib(c).decide(req)).answers[0];
  assert.equal(h.method, "head");
  const ph = softmax(e.scores_head);
  assert.deepEqual(h.probabilities.map((p) => p.value), c.probe.options);
  h.probabilities.forEach((p, i) => assert.ok(near(p.probability, ph[i])));
  assert.equal(h.choice, e.pred_head);
  assert.ok(near(h.confidence, Math.max(...ph)));
  const l = (await lib(c).decide(req, { method: "logits" })).answers[0];
  const pl = softmax(e.scores_logits);
  l.probabilities.forEach((p, i) => assert.ok(near(p.probability, pl[i])));
  assert.equal(l.choice, e.pred_logits);
});

test("choice: labels sharing a first token are rejected for the logits read, fine for the head", { skip }, async () => {
  const c = C.choice_tie;
  const req = { input: c.probe.state, questions: [{ name: "t", type: "choice", choices: c.probe.options }] };
  await assert.rejects(lib(c).decide(req, { method: "logits" }), /"dom-xss" and "dom-based" start with the same token 5425/);
  await assert.rejects(lib(c).decide(req, { method: "both" }), /same token/);
  const h = (await lib(c).decide(req, { method: "head" })).answers[0];
  assert.equal(h.choice, c.expected.pred_head);
});

test("score: head scalar (no distribution) and logits distribution over the level tokens", { skip }, async () => {
  const c = C.score, e = c.expected;
  const h = (await lib(c).decide({ input: c.probe.state, questions: [{ name: "sev", type: "score" }] })).answers[0];
  assert.equal(h.type, "score"); assert.equal(h.method, "head");
  assert.ok(near(h.raw, e.raw)); assert.ok(near(h.score, sig(e.raw) * 4));
  assert.equal(h.level, String(e.level_head));
  assert.equal(h.probabilities, null); assert.equal(h.confidence, null);
  const l = (await lib(c).decide({ input: c.probe.state, questions: [{ name: "sev", type: "score" }] }, { method: "logits" })).answers[0];
  const p = softmax([16, 17, 18, 19, 20].map((k) => c.logits[k]));
  assert.deepEqual(l.probabilities.map((x) => [x.value, x.label]), [[0, "1"], [1, "2"], [2, "3"], [3, "4"], [4, "5"]]);
  l.probabilities.forEach((x, i) => assert.ok(near(x.probability, p[i])));
  assert.ok(near(l.score, p.reduce((a, q, i) => a + q * i, 0)));
  assert.ok(near(l.confidence, Math.max(...p)));
  assert.equal(String(p.indexOf(Math.max(...p)) + 1), String(e.level_logits));
  await assert.rejects(lib(c).decide({ input: c.probe.state, questions: [{ name: "sev", type: "score", levels: ["low", "high"] }] }, { method: "logits" }), /reads 5 level tokens/);
});

test("open-jev question objects: keyed object, options, descriptions", () => {
  const qs = normalizeQuestions({
    area: { type: "choice", instructions: "Which area?", options: ["fees", "refund"], descriptions: { fees: "charges" } },
    urgent: { type: "noul", instructions: "Is it urgent?" },
    mood: { type: "score", instructions: "Mood?", options: ["bad", "ok", "good"] },
  }, CT0);
  assert.deepEqual(qs.map((q) => [q.name, q.type]), [["area", "choice"], ["urgent", "predicate"], ["mood", "score"]]);
  assert.deepEqual(qs[0].values, ["fees", "refund"]);
  assert.deepEqual(qs[0].texts, ["fees: charges", "refund"]);
  assert.deepEqual(qs[2].values, ["bad", "ok", "good"]);
});

test("Decisions API shapes: choices/levels objects, names, validation errors", () => {
  const qs = normalizeQuestions([
    { name: "d", type: "choice", instructions: "Dept?", choices: [{ value: "billing", description: "Payments" }, "other"] },
    { name: "s", type: "score", instructions: "Sev?", levels: [{ label: "Cosmetic" }, { label: "Blocked", description: "no workaround" }] },
  ], CT0);
  assert.deepEqual(qs[0].texts, ["billing: Payments", "other"]);
  assert.deepEqual(qs[1].values, ["Cosmetic", "Blocked"]);
  assert.equal(normalizeQuestions([{ type: "predicate", instructions: "x" }], CT0)[0].name, "q0");
  assert.throws(() => normalizeQuestions([{ type: "maybe", instructions: "x" }], CT0), /unknown type/);
  assert.throws(() => normalizeQuestions([{ name: "a", type: "predicate", instructions: "x" }, { name: "a", type: "predicate", instructions: "y" }], CT0), /duplicate question name/);
  assert.throws(() => normalizeQuestions([{ type: "choice", instructions: "x", choices: ["a"] }], CT0), /at least 2/);
  assert.throws(() => normalizeQuestions([{ type: "choice", instructions: "x", choices: ["a", "a"] }], CT0), /duplicate/);
  assert.throws(() => normalizeQuestions([{ type: "choice", instructions: "x", choices: ["a</opt>", "b"] }], CT0), /reserved/);
  assert.throws(() => normalizeQuestions([{ type: "predicate" }], CT0), /instructions required/);
  assert.throws(() => normalizeQuestions([{ type: "predicate", instructions: "x", choices: ["a", "b"] }], CT0), /predicate takes no/);
  assert.throws(() => normalizeQuestions([], CT0), /non-empty/);
  // manifest defaults fill choice/score instructions and score levels
  const d = normalizeQuestions([{ type: "score" }], CT0)[0];
  assert.equal(d.instructions, CT0.defaults.instructions.score); assert.deepEqual(d.values, ["1", "2", "3", "4", "5"]);
});

test("input: string, or messages with input_text / input_json parts", () => {
  assert.equal(inputToText("abc"), "abc");
  assert.equal(inputToText([{ role: "user", content: "hi" }]), "hi");
  assert.equal(inputToText([{ role: "user", content: [{ type: "input_text", text: "t" }, { type: "input_json", data: { a: 1 } }] }, { role: "assistant", content: "ok" }]), 'user: t\n{"a":1}\nassistant: ok');
  assert.throws(() => inputToText([{ role: "user", content: [{ type: "input_image", image_url: "x" }] }]), /unsupported part/);
});

test("session: System 2 continues from the kept decision prefill (sharedPrefill)", { skip }, async () => {
  const c = C.noul;
  const be = fake(c, { keep: true });
  const m = S1S2.fromBackend(M16, "decode-only-q4", be, tok, heads);
  assert.equal(m.sharedPrefill, true);
  const s = await m.decide({ input: c.probe.state, questions: [{ name: "ioc", type: "noul", instructions: c.probe.q }] });
  assert.equal(s.sharedPrefill, true);
  assert.equal(be.calls[0].keep, true);
  const g = await s.generate("Why?", { maxNewTokens: 8 });
  assert.equal(g.sharedPrefill, true); assert.equal(g.reusedTokens, c.ids.length);
  const gf = be.calls.find((x) => x.op === "generateFrom");
  assert.deepEqual(gf.head, c.ids, "continuation extends the decision prompt ids (not re-tokenized)");
  assert.equal(g.text, tok.decode([9175, 13], { skipSpecial: true }));
  await s.dispose();
  assert.ok(be.calls.some((x) => x.op === "dispose"), "kept state disposed with the session");
  await assert.rejects(s.generate("again"), /disposed/);
});

test("session: backends without kept state fall back to a fresh prefill, flagged", { skip }, async () => {
  const c = C.noul;
  const be = fake(c);
  const m = S1S2.fromBackend(M16, "decode-only-q4", be, tok, heads);
  const s = await m.decide({ input: c.probe.state, questions: [{ name: "ioc", type: "noul", instructions: c.probe.q }] });
  assert.equal(s.sharedPrefill, false);
  const g = await s.generate("Why?");
  assert.equal(g.sharedPrefill, false); assert.equal(g.reusedTokens, 0);
  const gen = be.calls.find((x) => x.op === "generate");
  const expect = c.ids.length + tok.encode(`${decisionOf(s.answers[0]) === "yes" ? "Yes" : "No"}<|im_end|>\n<|im_start|>user\nWhy?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`).length;
  assert.equal(gen.n, expect, "same ids as the shared path: decision prompt + answer + new turn");
});

test("fromBackend rejects heads whose hidden size differs from the manifest", { skip }, () => {
  assert.throws(() => S1S2.fromBackend(CT0, "decode-only-q4", fake(C.noul), tok, heads), /hiddenSize 16 != manifest 2560/);
});
