// Shared prefill must be bit-identical to a full prefill on a runtime where a block shorter than 32 tokens takes another
// numeric path (onnxruntime-web 1.31 WebGPU int4, measured on Qwen3-0.6B in Chrome). The mock session below perturbs every token fed in a
// block of S < 32 (S > 1: one-token decode steps are the same in both paths), so reuse equals a full prefill only when the
// backend keeps every block >= 32.
import test from "node:test";
import assert from "node:assert/strict";
import { genaiFromSession, blockPlan, backOff, S1S2, validateManifest } from "../dist/index.js";
import { toy, toyManifest, TOY_PIECES } from "./helpers.mjs";

const V = TOY_PIECES.length, MIN = 32;
const EOS = new Set([2]);
class Tensor {
  constructor(type, data, dims, location = "cpu") { this.type = type; this.dims = dims; this._d = data; this.location = location; this.disposed = false; }
  async getData() { return this._d; }
  get data() { return this._d; }
  dispose() { this.disposed = true; }
}
const ort = { Tensor };

function mkSession() {
  const meta = [{ name: "input_ids", type: "int64", shape: [1, "s"] }, { name: "attention_mask", type: "int64", shape: [1, "t"] },
    { name: "position_ids", type: "int64", shape: [3, 1, "s"] }, { name: "past_key_values.0.key", type: "float32", shape: [1, 1, "p", 1] },
    { name: "past_key_values.0.value", type: "float32", shape: [1, 1, "p", 1] }];
  const s = { blocks: [], short: 0, inputNames: meta.map((m) => m.name), inputMetadata: meta,
    outputNames: ["logits", "present.0.key", "present.0.value", "hidden_states"],
    async run(feeds, fetches) {
      const S = feeds.input_ids.dims[1], ids = Array.from(feeds.input_ids._d, Number), past = feeds["past_key_values.0.key"].dims[2];
      if (feeds["past_key_values.0.key"].disposed) throw new Error("fed a disposed tensor");
      s.blocks.push(S);
      if (S > 1 && S < MIN) s.short++;
      const kv = new Float32Array(past + S); kv.set(feeds["past_key_values.0.key"]._d);
      for (let i = 0; i < S; i++) kv[past + i] = Math.fround(Math.sin(ids[i] * 7.31 + (past + i) * 0.17) + (S > 1 && S < MIN ? 1 / 1024 : 0));
      const acc = new Float64Array(S); let a = 0;
      for (let j = 0; j < past + S; j++) { a += kv[j]; if (j >= past) acc[j - past] = a; }
      const out = { "present.0.key": new Tensor("float32", kv, [1, 1, past + S, 1], "gpu-buffer"), "present.0.value": new Tensor("float32", kv.slice(), [1, 1, past + S, 1], "gpu-buffer") };
      if (fetches.includes("hidden_states")) out.hidden_states = new Tensor("float32", Float32Array.from(Array.from(acc).flatMap((x) => [x, -x])), [1, S, 2], "gpu-buffer");
      if (fetches.includes("logits")) { const l = new Float32Array(S * V); for (let i = 0; i < S; i++) for (let c = 0; c < V; c++) l[i * V + c] = Math.sin(acc[i] * (c + 1)); out.logits = new Tensor("float32", l, [1, S, V], "gpu-buffer"); }
      return out;
    }, async release() {} };
  return s;
}
const mk = (o = {}) => { const s = mkSession(); return { s, be: genaiFromSession(s, ort, { eos: EOS, hiddenOutput: "hidden_states", chunk: 128, ...o }) }; };
const ids = (n, seed = 1) => Array.from({ length: n }, (_, i) => 3 + ((i * 37 + seed * 11) % (V - 4)));
const same = (a, b) => a.length === b.length && a.every((x, i) => Object.is(x, b[i]));
const gen = async (it) => { const g = []; for await (const t of it) g.push(t); return g; };
const G = { temperature: 0, repetitionPenalty: 1, maxNewTokens: 6 };

test("blockPlan: chunk-sized blocks, a tail shorter than minBlock merged into the previous block", () => {
  assert.deepEqual(blockPlan(300, 128, 32), [128, 128, 44]);
  assert.deepEqual(blockPlan(268, 128, 32), [128, 140]);
  assert.deepEqual(blockPlan(20, 128, 32), [20]);
  assert.deepEqual(blockPlan(268, 128, 0), [128, 128, 12]);
});

test("the mock enforces the rule: the 0.1.2 chunking (minBlock 0) makes reuse differ from a full prefill", async () => {
  const { be } = mk({ minBlock: 0 }), x = ids(300);
  const full = await be.prefill(x, { positions: [299] });
  const pre = await be.prefill(x.slice(0, 280), { positions: [] }, { keep: true });
  const r = await be.prefillFrom(pre.state, x, { positions: [299] });
  assert.ok(!same(r.lastLogits, full.lastLogits), "a 20-token continuation takes the short path");
});

test("every prefill block is >= 32 tokens; prefill(keep) + prefillFrom == full prefill, bit for bit, for any cut", async () => {
  const { s, be } = mk();
  for (const n of [70, 130, 160, 161, 268, 300, 400]) {
    const x = ids(n, n), pos = [n - 1, n - 2, Math.floor(n / 2)];
    const full = await be.prefill(x, { positions: pos });
    for (const cut of [64, 96, 128, n - 40, n - 31, n - 20, n - 5, n - 1].filter((c) => c >= 64 && c < n)) {
      const pre = await be.prefill(x.slice(0, cut), { positions: [] }, { keep: true });
      assert.ok(pre.state, `a state is kept for cut ${cut}`);
      assert.equal(pre.state.ids.length, cut - MIN, "the kept state stops minBlock tokens before the end");
      const r = await be.prefillFrom(pre.state, x, { positions: pos.filter((p) => p >= pre.state.ids.length) });
      assert.ok(same(r.lastLogits, full.lastLogits), `logits n=${n} cut=${cut}`);
      const want = pos.filter((p) => p >= pre.state.ids.length);
      r.hiddens.forEach((h, i) => assert.ok(same(h, full.hiddens[pos.indexOf(want[i])]), `hidden n=${n} cut=${cut}`));
      await pre.state.dispose();
    }
  }
  assert.equal(s.short, 0, `blocks fed: ${[...new Set(s.blocks)].sort((a, b) => a - b).join(",")}`);
});

test("short prefills keep no state (continuation = full prefill); snapshot/continuation shorter than minBlock throws", async () => {
  const { be } = mk(), x = ids(100);
  assert.equal((await be.prefill(x.slice(0, 63), { positions: [] }, { keep: true })).state, undefined);
  await assert.rejects(be.snapshot(x.slice(0, 31)), /minBlock/);
  const h = await be.snapshot(x.slice(0, 50));
  assert.equal(h.ids.length, 50);
  await assert.rejects(be.prefillFrom(h, x.slice(0, 70), { positions: [] }), /minBlock/);
  const r = await be.prefillFrom(h, x.slice(0, 82), { positions: [] });
  assert.ok(same(r.lastLogits, (await be.prefill(x.slice(0, 82), { positions: [] })).lastLogits));
  await h.dispose();
});

test("generateFrom a kept state == generate from scratch (tokens), with a continuation of a few tokens", async () => {
  const { s, be } = mk(), x = ids(200);
  const pre = await be.prefill(x.slice(0, 190), { positions: [] }, { keep: true });
  for (const n of [191, 195, 200]) assert.deepEqual(await gen(be.generateFrom(pre.state, x.slice(0, n), G)), await gen(be.generate(x.slice(0, n), G)));
  await pre.state.dispose();
  assert.equal(s.short, 0);
});

test("decide(): several questions on one input (shared prefix) == each question alone; System 2 from the kept state == fresh", async () => {
  const m = validateManifest(toyManifest());
  const input = "Clause 1: the receiving party may share data with affiliates. ".repeat(6);
  const qs = [{ name: "a", type: "predicate", instructions: "May it share data?" }, { name: "b", type: "predicate", instructions: "Is it so?" },
    { name: "c", type: "choice", instructions: "Yes or no?", choices: [" Yes", " No"] }];
  const { s, be } = mk();
  const lib = S1S2.fromBackend(m, "genai-int4-webgpu", be, toy);
  const snaps = [], snap = be.snapshot.bind(be);
  be.snapshot = async (x) => { snaps.push(x.length); return snap(x); };
  const all = await lib.decide({ input, questions: qs });
  assert.equal(snaps.length, 1, "the questions shared one prefix snapshot");
  for (const [i, q] of qs.entries()) {
    const one = await lib.decide({ input, questions: [q] }, { keepState: false });
    assert.ok(Object.is(all.answers[i].probability ?? all.answers[i].confidence, one.answers[0].probability ?? one.answers[0].confidence), `question ${q.name}`);
    assert.deepEqual(all.answers[i].logits ?? null, one.answers[0].logits ?? null);
    await one.dispose();
  }
  assert.equal(all.sharedPrefill, true);
  const g = await all.generate("Why?", { ...G, question: "b" });
  assert.equal(g.sharedPrefill, true); assert.ok(g.reusedTokens > 0);
  const { be: be2 } = mk();
  const fresh = await S1S2.fromBackend(m, "genai-int4-webgpu", be2, toy).decide({ input, questions: [qs[1]] }, { keepState: false });
  const g2 = await fresh.generate("Why?", G);
  assert.equal(g2.sharedPrefill, false); assert.deepEqual(g.tokenIds, g2.tokenIds);
  await all.dispose();
  assert.equal(s.short, 0);
});

test("backOff: shared prefix placed so the snapshot and every continuation are >= minBlock", () => {
  assert.equal(backOff(100, [200, 150], 0), 100);
  assert.equal(backOff(100, [200, 150 - 64], 32), 86);
  assert.equal(backOff(100, [200, 40], 32), 40);
  assert.equal(backOff(100, [200, 31], 32), 0);
});
