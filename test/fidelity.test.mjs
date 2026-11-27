import test from "node:test";
import assert from "node:assert/strict";
import { agreement, fidelityReport, wilson, invertQuestion, orderInversion, runOrderInversion, runCases, parseJsonl, toJsonl, fidelitySlot, recordOf } from "../dist/fidelity.js";

const R = (id, decision, type = "choice") => ({ id, decision, type });

test("agreement: changed %, missing ids, per type", () => {
  const ref = [R("a", "x"), R("b", "y"), R("c", "yes", "predicate"), R("d", "no", "predicate"), R("e", "3", "score")];
  const got = [R("a", "x"), R("b", "z"), R("c", "no", "predicate"), R("d", "no", "predicate")];
  const a = agreement(ref, got);
  assert.equal(a.n, 4); assert.equal(a.changed, 2); assert.equal(a.changedPct, 50);
  assert.deepEqual(a.changedIds, ["b", "c"]); assert.deepEqual(a.missing, ["e"]);
  assert.deepEqual(a.byType.choice, { n: 2, changed: 1, changedPct: 50 });
  assert.deepEqual(a.byType.predicate, { n: 2, changed: 1, changedPct: 50 });
  assert.equal(a.byType.score, undefined);
  assert.throws(() => agreement([R("a", "x"), R("a", "y")], []), /duplicate id a/);
  assert.throws(() => agreement([{ id: "a" }], []), /needs string id and decision/);
});

test("wilson interval", () => {
  assert.deepEqual(wilson(0, 0), [0, 0]);
  const [lo, hi] = wilson(0, 172);
  assert.equal(lo, 0); assert.ok(hi > 2 && hi < 2.3);
  const [l2, h2] = wilson(20, 172);
  assert.ok(l2 < 100 * 20 / 172 && h2 > 100 * 20 / 172);
  assert.ok(Math.abs(l2 - 7.66) < 0.1 && Math.abs(h2 - 17.36) < 0.1);
});

test("fidelityReport + manifest slot", () => {
  const ref = [R("a", "x"), R("b", "y")];
  const rep = fidelityReport(ref, { "genai-webgpu-int4": [R("a", "x"), R("b", "x")], "decode-only-q4": ref });
  assert.equal(rep["genai-webgpu-int4"].changedPct, 50); assert.equal(rep["decode-only-q4"].changedPct, 0);
  const slot = fidelitySlot("pytorch-bf16", { logits: rep["genai-webgpu-int4"] }, { logits: orderInversion(ref, ref) });
  assert.deepEqual(slot.decisionChanged, { logits: 0.5 }); assert.deepEqual(slot.orderInversion, { logits: 0 }); assert.equal(slot.n, 2);
});

test("invertQuestion: reverses choice order; predicate/score have no option order", () => {
  assert.deepEqual(invertQuestion({ type: "choice", instructions: "q", choices: ["a", { value: "b" }, "c"] }).choices, ["c", { value: "b" }, "a"]);
  assert.deepEqual(invertQuestion({ type: "choice", instructions: "q", options: ["yes", "no"] }).options, ["no", "yes"]);
  assert.equal(invertQuestion({ type: "predicate", instructions: "q" }), null);
  assert.equal(invertQuestion({ type: "score", instructions: "q", levels: ["l", "h"] }), null);
});

test("orderInversion: flips, rate and pass threshold", () => {
  const o = [R("a", "yes"), R("b", "no"), R("c", "x"), R("d", "y")];
  const s = [R("a", "yes"), R("b", "yes"), R("c", "x"), R("d", "y")];
  const r = orderInversion(o, s, { maxFlipPct: 30 });
  assert.equal(r.n, 4); assert.equal(r.flips, 1); assert.equal(r.flipPct, 25); assert.deepEqual(r.flippedIds, ["b"]); assert.equal(r.pass, true);
  assert.equal(orderInversion(o, s, { maxFlipPct: 20 }).pass, false);
  assert.equal(orderInversion([], []).pass, false, "nothing measured is not a pass");
});

test("runOrderInversion with an injected decide (position-biased fake flips every time)", async () => {
  const cases = [
    { id: "1", input: "s", question: { type: "choice", instructions: "q", choices: ["a", "b"] } },
    { id: "2", input: "s", question: { type: "choice", instructions: "q", choices: ["c", "d", "e"] } },
    { id: "3", input: "s", question: { type: "predicate", instructions: "q" } },
  ];
  const firstOption = async ({ questions: [q] }) => ({ answers: [{ name: q.name, type: "choice", choice: (q.choices ?? q.options)[0], probabilities: [], confidence: 1, method: "logits" }] });
  const r = await runOrderInversion(cases, firstOption, { maxFlipPct: 5 });
  assert.equal(r.n, 2); assert.equal(r.flips, 2); assert.equal(r.skipped, 1); assert.equal(r.pass, false);
  const labelBased = async ({ questions: [q] }) => ({ answers: [{ name: q.name, type: "choice", choice: [...(q.choices ?? q.options)].sort()[0], probabilities: [], confidence: 1, method: "head" }] });
  assert.equal((await runOrderInversion(cases, labelBased)).pass, true);
});

test("runCases + jsonl round trip", async () => {
  const cases = parseJsonl('{"id":"p","input":"s","question":{"type":"noul","instructions":"q"}}\n// comment\n\n{"id":"c","input":"s","question":{"type":"choice","instructions":"q","choices":["a","b"]}}\n');
  let disposed = 0;
  const out = await runCases(cases, async ({ questions: [q] }) => ({
    answers: [q.type === "noul" ? { name: q.name, type: "predicate", probability: 0.7, confidence: 0.7, method: "logits" } : { name: q.name, type: "choice", choice: "b", probabilities: [], confidence: 0.6, method: "head" }],
    dispose: () => { disposed++; },
  }));
  assert.deepEqual(out, [{ id: "p", type: "predicate", decision: "yes", method: "logits", probability: 0.7 }, { id: "c", type: "choice", decision: "b", method: "head" }]);
  assert.equal(disposed, 2);
  assert.deepEqual(parseJsonl(toJsonl(out)), out);
  assert.throws(() => parseJsonl("{bad"), /jsonl line 1/);
  assert.equal(recordOf("s", { type: "score", level: "4", method: "head" }).decision, "4");
});

import { forecastParity } from "../dist/fidelity.js";
const F = (id, ...ps) => ({ id, behaviors: Object.fromEntries(ps.map((p, i) => [`b${i}`, { p }])) });

test("forecastParity: tolerance max |dp| 0.05 and 5% flips, written before any measurement", () => {
  const ref = [F("a", 0.9, 0.1), F("b", 0.4, 0.6), F("c", 0.2, 0.8)];
  const same = forecastParity(ref, ref);
  assert.equal(same.n, 3); assert.equal(same.maxAbsDp, 0); assert.equal(same.flips, 0); assert.equal(same.pass, true);
  assert.deepEqual(same.tolerance, { maxAbsDp: 0.05, maxFlipPct: 5, threshold: 0.5 });
  const close = forecastParity(ref, [F("a", 0.92, 0.09), F("b", 0.41, 0.58), F("c", 0.2, 0.8)]);
  assert.ok(Math.abs(close.maxAbsDp - 0.02) < 1e-12); assert.equal(close.pass, true);
  const far = forecastParity(ref, [F("a", 0.7, 0.1), F("b", 0.4, 0.6), F("c", 0.2, 0.8)]);
  assert.ok(Math.abs(far.maxAbsDp - 0.2) < 1e-12); assert.equal(far.pass, false, "|dp| 0.2 > 0.05");
  const flip = forecastParity(ref, [F("a", 0.9, 0.1), F("b", 0.51, 0.6), F("c", 0.2, 0.8)]);
  assert.equal(flip.flips, 1); assert.ok(Math.abs(flip.flipPct - 100 / 6) < 1e-9); assert.equal(flip.pass, false, "one flip of 6 pairs = 16.7% > 5%");
  assert.equal(forecastParity(ref, [F("a", 0.9, 0.1)]).missing.length, 2);
  assert.equal(forecastParity(ref, [F("a", 0.9, 0.1)]).pass, false, "partial coverage is not a pass");
  const empty = [{ id: "a", behaviors: {} }];
  assert.equal(forecastParity(empty, empty).pass, false, "no behavior pairs is not a pass");
  assert.equal(forecastParity(ref, []).pass, false, "no overlap is not a pass");
  assert.throws(() => forecastParity(ref, [{ id: "a", behaviors: { b0: { p: 0.9 } } }]), /no behavior "b1"/);
});
