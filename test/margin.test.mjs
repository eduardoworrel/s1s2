// Margin field + margin policy (escalation) + manifest tasks, against mock backends and the toy tokenizer.
// Nothing here measures anything: thresholds are test values, not recommendations.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  S12, validateManifest, ManifestError, predicateFromP, scoreFromHead, topGap, mergePolicy, shouldEscalate,
  thresholdFromQuantile, MARGIN_POLICY_DEFAULTS, normalizeQuestions, logitIds, encodeTagged, assertManifestTokens, decisionOf,
} from "../dist/index.js";
import { clone, toy, toyManifest, TOY_ID as ID, TOY_PIECES } from "./helpers.mjs";

const YES = ID.get("▁Yes"), NO = ID.get("▁No");
const near = (a, b, t = 1e-6) => Math.abs(a - b) <= t;

/** Mock backend: `logits(ids)` -> { tokenId: value } for the last position; everything else is -20. */
function fake(logits, { keep = false, out = [YES] } = {}) {
  const calls = [];
  const read = (ids) => { const L = new Float32Array(TOY_PIECES.length).fill(-20); for (const [k, v] of Object.entries(logits(ids))) L[+k] = v; return L; };
  const be = {
    calls, kind: "fake", hiddenName: null,
    async prefill(ids, want, o = {}) {
      calls.push({ op: "prefill", n: ids.length, keep: !!o.keep });
      const r = { hiddens: [], lastLogits: read(ids) };
      if (keep && o.keep) r.state = { ids: ids.slice(), dispose() { calls.push({ op: "dispose" }); } };
      return r;
    },
    async *generate(ids, o) { calls.push({ op: "generate", ids: ids.slice(), temperature: o.temperature, max: o.maxNewTokens }); yield* out; },
    async dispose() { calls.push({ op: "backend-dispose" }); },
  };
  if (keep) {
    be.snapshot = async (ids) => ({ ids: ids.slice(), dispose() {} });
    be.prefillFrom = async (h, ids, want, o) => be.prefill(ids, want, o);
    be.generateFrom = async function* (h, ids, o) { calls.push({ op: "generateFrom", from: h.ids.length, ids: ids.slice(), temperature: o.temperature }); yield* out; };
  }
  return be;
}

/** Toy manifest with an int8 build, the "lab" choice tokens, and tasks. */
function M({ escalation, buildEscalation, tasks } = {}) {
  const m = toyManifest();
  m.builds["genai-int8-webgpu"] = clone(m.builds["genai-int4-webgpu"]);
  if (escalation) m.escalation = escalation;
  if (buildEscalation) m.builds["genai-int4-webgpu"].escalation = buildEscalation;
  if (tasks) m.tasks = tasks;
  return validateManifest(m);
}
const lib = (m, be, build = "genai-int4-webgpu") => S12.fromBackend(m, build, be, toy);
// Logits by question: the instruction text decides the yes/no gap.
const byText = (table) => (ids) => { const t = toy.decode(ids); for (const [k, v] of Object.entries(table)) if (t.includes(k)) return v; return { [YES]: 1, [NO]: 0 }; };

test("margin (probability gap) and rawMargin (score gap) on every answer, per readout", () => {
  const pr = predicateFromP("p", 0.9, "logits", -2.5);
  assert.ok(near(pr.margin, 0.8), "predicate margin: |2p - 1|"); assert.equal(pr.rawMargin, 2.5, "predicate rawMargin: |raw|");
  assert.ok(near(predicateFromP("p", 0.8807970779778823, "head").rawMargin, 2), "predicate without raw: |logit(p)|");
  assert.equal(topGap([1, 4, 3]), 1);
  assert.equal(topGap([5]), null);
  const h = scoreFromHead("s", ["1", "2", "3"], 0.3);
  assert.equal(h.margin, null, "score head: a scalar, no runner-up"); assert.equal(h.rawMargin, null);
  assert.equal(h.path, "s1");
});

test("margin: decide() reports margin and path s1; escalation is off by default", async () => {
  const m = M({ escalation: { below: { predicate: 100, choice: 100 }, to: "generate" } });
  const be = fake(() => ({ [YES]: 2, [NO]: 0.5 }));
  const s = await lib(m, be).decide({ input: "We share data.", questions: [
    { name: "p", type: "predicate", instructions: "Is it so?" },
    { name: "c", type: "choice", instructions: "Is it so?", choices: [" Yes", " No"] },
  ] });
  const pgap = 2 / (1 + Math.exp(-1.5)) - 1;
  assert.ok(near(s.answers[0].margin, pgap) && near(s.answers[1].margin, pgap), "p(1st) - p(2nd), predicate and 2-way choice agree");
  assert.ok(near(s.answers[0].rawMargin, 1.5) && near(s.answers[1].rawMargin, 1.5));
  assert.deepEqual(s.answers.map((a) => a.path), ["s1", "s1"]);
  assert.equal(s.answers[0].escalation, undefined);
  assert.ok(!be.calls.some((c) => c.op.startsWith("generate")), "no System 2 without escalate");
});

test("escalate to generate: continues from the kept prefill, attaches text, leaves the typed decision unchanged", async () => {
  const m = M({ escalation: { below: { predicate: 0.5 }, to: "generate", source: "test value" } });
  const be = fake(byText({ "Unsure?": { [YES]: 0.3, [NO]: 0 }, "Sure?": { [YES]: 5, [NO]: 0 } }), { keep: true });
  const s = await lib(m, be).decide({ input: "We share data.", questions: [
    { name: "unsure", type: "predicate", instructions: "Unsure?" },
    { name: "sure", type: "predicate", instructions: "Sure?" },
  ] }, { escalate: true });
  const [u, k] = s.answers;
  assert.equal(k.path, "s1"); assert.equal(k.escalation, undefined);
  assert.equal(u.path, "escalated");
  assert.equal(decisionOf(u), "yes", "S1 decision kept");
  assert.ok(near(u.probability, 1 / (1 + Math.exp(-0.3)), 1e-6));
  const e = u.escalation;
  assert.equal(e.to, "generate"); assert.equal(e.reason, "margin");
  assert.equal(e.metric, "margin"); assert.ok(near(e.margin, 2 / (1 + Math.exp(-0.3)) - 1, 1e-6)); assert.equal(e.threshold, 0.5);
  assert.equal(e.provisional, true, "a manifest threshold without provisional:false stays provisional");
  assert.equal(e.source, "test value");
  assert.equal(e.text, "Yes"); assert.equal(e.sharedPrefill, true); assert.ok(e.reusedTokens > 0);
  const g = be.calls.filter((c) => c.op === "generateFrom");
  assert.equal(g.length, 1, "only the low-margin question goes to System 2");
  assert.equal(g[0].temperature, 0, "greedy");
  // The continuation does not inject the uncertain S1 answer: ids = decision prompt + escalate continuation.
  const cont = toy.decode(g[0].ids.slice(g[0].from));
  assert.ok(cont.startsWith("<|im_end|>"), "continue template with an empty {answer}");
  assert.ok(cont.includes("Reconsider the question above step by step"), "default prompt (provisional wording)");
  await s.dispose();
});

test("escalate to generate without kept state: same ids are prefilled again (sharedPrefill false)", async () => {
  const m = M({ escalation: { below: { choice: 0.5 }, to: "generate", prompt: "Pick one of {options}." } });
  m.generation.escalate = "\n{prompt}<|im_end|>\n";
  const be = fake(() => ({ [YES]: 0.2, [NO]: 0 }), { keep: true });
  const s = await lib(validateManifest(m), be).decide({ input: "x", questions: [{ name: "c", type: "choice", instructions: "Is it so?", choices: [" Yes", " No"] }] }, { escalate: true, keepState: false });
  const e = s.answers[0].escalation;
  assert.equal(e.sharedPrefill, false); assert.equal(e.reusedTokens, 0);
  const g = be.calls.find((c) => c.op === "generate");
  assert.ok(toy.decode(g.ids).includes("Pick one of  Yes,  No."), "manifest generation.escalate + {options}");
  assert.ok(!be.calls.some((c) => c.op === "generateFrom"));
});

test("escalate to another build (int4 -> int8): low-margin questions re-read there, original kept in escalation.from", async () => {
  const m = M({ buildEscalation: { metric: "rawMargin", below: { choice: 1 }, to: { build: "genai-int8-webgpu" }, provisional: true } });
  const int4 = fake(byText({ "Unsure?": { [YES]: 0.1, [NO]: 0 }, "Sure?": { [YES]: 4, [NO]: 0 } }));
  const int8 = fake(() => ({ [YES]: 0, [NO]: 3 }));
  let loads = 0;
  const l = lib(m, int4).withBuildLoader(async (b) => { loads++; assert.equal(b, "genai-int8-webgpu"); return lib(m, int8, b); });
  const q = [{ name: "u", type: "choice", instructions: "Unsure?", choices: [" Yes", " No"] }, { name: "s", type: "choice", instructions: "Sure?", choices: [" Yes", " No"] }];
  const s = await l.decide({ input: "x", questions: q }, { escalate: true });
  const [u, k] = s.answers;
  assert.equal(k.path, "s1"); assert.equal(k.choice, " Yes");
  assert.equal(u.path, "escalated"); assert.equal(u.choice, " No", "int8 answer replaces");
  assert.ok(near(u.rawMargin, 3)); assert.equal(u.escalation.metric, "rawMargin"); assert.equal(u.escalation.to, "build"); assert.equal(u.escalation.build, "genai-int8-webgpu");
  assert.equal(u.escalation.from.choice, " Yes"); assert.ok(near(u.escalation.from.rawMargin, 0.1, 1e-6)); assert.equal(u.escalation.from.path, "s1");
  assert.equal(int8.calls.filter((c) => c.op === "prefill").length, 1, "int8 reads only the escalated question");
  await l.decide({ input: "y", questions: q }, { escalate: true });
  assert.equal(loads, 1, "the int8 build is loaded once");
  await l.dispose();
  assert.ok(int8.calls.some((c) => c.op === "backend-dispose"), "disposed with the parent");
  // explicit target, no loader
  const t = await lib(m, fake(() => ({ [YES]: 0.1, [NO]: 0 }))).decide({ input: "x", questions: q.slice(0, 1) }, { escalate: { target: lib(m, int8, "genai-int8-webgpu") } });
  assert.equal(t.answers[0].choice, " No");
});

test("policy levels: call overrides manifest/build/task; thresholds without a target refuse; no threshold, no escalation", async () => {
  const m = M({ escalation: { below: { predicate: 0.1 }, to: "generate", provisional: false, source: "manifest" } }); // p-gap of raw 0.5 = 0.245
  const be = () => fake(() => ({ [YES]: 0.5, [NO]: 0 }), { keep: true });
  const Q = [{ name: "p", type: "predicate", instructions: "Is it so?" }];
  assert.equal((await lib(m, be()).decide({ input: "x", questions: Q }, { escalate: true })).answers[0].path, "s1", "0.5 >= 0.1");
  const c = (await lib(m, be()).decide({ input: "x", questions: Q }, { escalate: { below: { predicate: 0.5 } } })).answers[0];
  assert.equal(c.path, "escalated"); assert.equal(c.escalation.threshold, 0.5);
  assert.equal(c.escalation.provisional, true, "a call threshold without provisional:false is provisional");
  assert.equal(c.escalation.source, "manifest < unspecified");
  const bare = M();
  await assert.rejects(lib(bare, be()).decide({ input: "x", questions: Q }, { escalate: { below: { predicate: 1 } } }), /thresholds but no target/);
  await assert.rejects(lib(m, be()).decide({ input: "x", questions: Q }, { escalate: { belw: { predicate: 1 } } }), /escalate.belw: unknown key/, "call-level typos are not ignored");
  await assert.rejects(lib(m, be()).decide({ input: "x", questions: Q }, { escalate: { to: "int8" } }), /escalate.to/);
  assert.equal((await lib(bare, be()).decide({ input: "x", questions: Q }, { escalate: true })).answers[0].path, "s1", "no thresholds anywhere: nothing escalates");
});

test("mergePolicy / shouldEscalate / thresholdFromQuantile", () => {
  const p = mergePolicy({ below: { choice: 2 }, to: "generate", provisional: false, source: "a" }, { below: { predicate: 1 }, source: "b" });
  assert.deepEqual(p.below, { choice: 2, predicate: 1 });
  assert.equal(p.provisional, true); assert.equal(p.source, "a < b");
  assert.equal(mergePolicy().source, MARGIN_POLICY_DEFAULTS.source);
  assert.match(MARGIN_POLICY_DEFAULTS.source, /provisional - set by study 002/);
  assert.equal(p.metric, "margin", "default metric: the probability gap");
  assert.deepEqual(shouldEscalate({ type: "choice", margin: 1.5 }, p), { threshold: 2, value: 1.5 });
  assert.equal(shouldEscalate({ type: "choice", margin: 3, rawMargin: 0.1 }, p), null, "reads margin, not rawMargin");
  assert.deepEqual(shouldEscalate({ type: "choice", margin: 3, rawMargin: 0.1 }, mergePolicy({ metric: "rawMargin", below: { choice: 1 } })), { threshold: 1, value: 0.1 });
  assert.equal(shouldEscalate({ type: "choice", margin: 2 }, p), null, "strictly below");
  assert.equal(shouldEscalate({ type: "score", margin: 0 }, p), null, "type without threshold");
  assert.equal(shouldEscalate({ type: "choice", margin: null }, p), null, "no margin (score head)");
  const ms = [5, 1, 3, 2, 4, 0.5, 6, 7, 8, 9];
  const t = thresholdFromQuantile(ms, 0.2);
  assert.equal(ms.filter((x) => x < t).length, 2, "escalates 20% of 10");
  assert.throws(() => thresholdFromQuantile(ms, 1), /in \(0, 1\)/);
});

test("manifest: escalation blocks validate; shipped manifests validate unchanged", () => {
  for (const f of ["smollm2-135m-instruct.en.json", "smollm2-135m-instruct.pt.json", "ct0-alias3-micro-1026.json"]) {
    const raw = JSON.parse(readFileSync(new URL(`../manifests/${f}`, import.meta.url), "utf8"));
    assert.equal(raw.escalation, undefined); assert.equal(raw.tasks, undefined);
    validateManifest(raw);
  }
  const bad = (mut, re) => { const m = toyManifest(); mut(m); assert.throws(() => validateManifest(m), (e) => e instanceof ManifestError && re.test(e.message), re.toString()); };
  bad((m) => { m.escalation = { below: { choice: -1 } }; }, /escalation.below.choice: finite number >= 0/);
  bad((m) => { m.escalation = { below: { label: 1 } }; }, /escalation.below.label: not a question type/);
  bad((m) => { m.escalation = { to: { build: "nope" } }; }, /unknown build "nope"/);
  bad((m) => { m.escalation = { to: "int8" }; }, /escalation.to: "generate" or \{ build/);
  bad((m) => { m.escalation = { threshold: 1 }; }, /escalation.threshold: unknown key/);
  bad((m) => { m.builds["genai-int4-cpu"].escalation = { provisional: "yes" }; }, /builds.genai-int4-cpu.escalation.provisional: boolean/);
  bad((m) => { m.generation.escalate = "{answer}{prompt}"; }, /generation.escalate/);
  bad((m) => { m.escalation = { metric: "entropy" }; }, /escalation.metric/);
});

const TASKS = {
  sentiment: { type: "choice", instructions: "What is the sentiment?", choices: [{ value: "positive", description: "likes it" }, "negative"] },
  pinned: { type: "choice", instructions: "Is it so?", choices: [{ value: "pos", token: { id: YES, text: "Yes" } }, { value: "neg", token: { id: NO, text: "No" } }],
    escalation: { below: { choice: 0.5 }, to: "generate" } },
  spam: { type: "predicate", instructions: "Is this spam?", tokens: { yes: { id: ID.get("Y"), text: "Y" }, no: { id: ID.get("N"), text: "N" } } },
};

test("tasks: any label set from the manifest; strings name tasks; pinned or in-context label tokens", async () => {
  const m = M({ tasks: TASKS });
  assertManifestTokens(toy, m);
  const [s, p, sp] = normalizeQuestions(["sentiment", { name: "x", task: "pinned" }, "spam"], m);
  assert.equal(s.name, "sentiment"); assert.equal(s.type, "choice"); assert.deepEqual(s.values, ["positive", "negative"]);
  assert.deepEqual(s.texts, ["positive: likes it", "negative"]); assert.equal(s.labelIds, undefined);
  assert.deepEqual(p.labelIds, [YES, NO]); assert.equal(p.name, "x"); assert.ok(p.escalation);
  assert.deepEqual(sp.labelIds, [ID.get("N"), ID.get("Y")], "predicate task: [no, yes]");
  // The manifest's global choice map is "tokens" (lab labels " Yes"/" No"); a task without pinned tokens reads the
  // first token of each label after the prompt instead of refusing.
  const ctx = encodeTagged(toy, m, s, "x").text;
  assert.deepEqual(logitIds({ manifest: m, tok: toy }, s, ctx), [ID.get("p"), ID.get("n")]);
  const be = fake(byText({ "Is it so?": { [YES]: 0.4, [NO]: 0 }, "sentiment": { [ID.get("n")]: 3, [ID.get("p")]: 1 }, "spam": { [ID.get("Y")]: 2, [ID.get("N")]: 0 } }), { keep: true });
  const r = await lib(m, be).decide({ input: "Great product.", questions: ["sentiment", "pinned", "spam"] }, { escalate: true });
  const [a, b, c] = r.answers;
  assert.equal(a.choice, "negative"); assert.ok(near(a.rawMargin, 2)); assert.equal(a.path, "s1");
  assert.equal(b.choice, "pos"); assert.equal(b.path, "escalated", "task-level policy"); assert.equal(b.escalation.to, "generate");
  assert.ok(near(c.probability, 1 / (1 + Math.exp(-2)))); assert.equal(c.path, "s1");
});

test("tasks: validation and request errors", () => {
  const bad = (tasks, re) => { const m = toyManifest(); m.tasks = tasks; assert.throws(() => validateManifest(m), re); };
  bad({ t: { type: "label", instructions: "x" } }, /tasks.t.type/);
  bad({ t: { type: "choice", instructions: "x", choices: ["a"] } }, /at least 2 labels/);
  bad({ t: { type: "choice", instructions: "x", choices: ["a", "a"] } }, /duplicate labels/);
  bad({ t: { type: "choice", instructions: "x", choices: [{ value: "a", token: { id: 1, text: "a" } }, "b"] } }, /every label or for none/);
  bad({ t: { type: "choice", instructions: "x", choices: [{ value: "a", token: { id: 1, text: "a" } }, { value: "b", token: { id: 1, text: "b" } }] } }, /distinct/);
  bad({ t: { type: "predicate", instructions: "x", choices: ["a", "b"] } }, /choices: not used by a predicate task/);
  bad({ t: { type: "choice", instructions: "", choices: ["a", "b"] } }, /instructions/);
  bad({ t: { type: "choice", instructions: "x", choices: ["a", "b"], escalation: { below: { choice: "1" } } } }, /tasks.t.escalation.below.choice/);
  const m = M({ tasks: TASKS });
  assert.throws(() => normalizeQuestions(["nope"], m), /unknown task "nope"/);
  assert.throws(() => normalizeQuestions([{ task: "sentiment", choices: ["a", "b"] }], m), /takes its labels from the manifest task/);
  assert.throws(() => normalizeQuestions([{ task: "sentiment", type: "predicate" }], m), /differs from task/);
  assert.throws(() => normalizeQuestions(["x"], M()), /declares no tasks/);
  const w = toyManifest(); w.tasks = { t: { type: "predicate", instructions: "x", tokens: { yes: { id: YES, text: "Nope" }, no: { id: NO, text: "No" } } } };
  assert.throws(() => assertManifestTokens(toy, validateManifest(w)), /mismatch: tasks.t.tokens.yes/);
});
