// genai backend against a mock ONNX Runtime session (mock ported from ct0-edge test_genai_mock.mjs).
import test from "node:test";
import assert from "node:assert/strict";
import { genaiFromSession, createGenai, parsePlacement, assertWebGpuPlacement } from "../dist/index.js";

const V = 16, H = 3;
const EOS = new Set([15]);
const O = (o = {}) => ({ eos: EOS, hiddenOutput: "hidden_states", ...o });
class Tensor {
  constructor(type, data, dims, location = "cpu") { this.type = type; this.dims = dims; this._d = data; this.location = location; this.disposed = false; }
  async getData() { return this._d; }
  get data() { return this._d; }
  dispose() { this.disposed = true; }
}
const ort = { Tensor };
function mkSession(opts = {}) {
  const meta = [
    { name: "input_ids", type: "int64", shape: [1, "s"] }, { name: "attention_mask", type: "int64", shape: [1, "t"] },
    { name: "position_ids", type: "int64", shape: [3, 1, "s"] },
    { name: "past_key_values.1.key", type: "float16", shape: [1, 4, "p", 8] }, { name: "past_key_values.1.value", type: "float16", shape: [1, 4, "p", 8] },
    { name: "past.0.conv", type: "float16", shape: [1, 6, 3] }, { name: "past.0.recurrent", type: "float16", shape: [1, 2, 4, 4] },
  ];
  const s = { calls: [], released: false, running: 0, overlap: false, made: [],
    inputNames: meta.map((m) => m.name), inputMetadata: meta,
    outputNames: ["logits", "present.1.key", "present.1.value", "present.0.conv", "present.0.recurrent", "hidden_states"],
    async run(feeds, fetches) {
      if (s.running++) s.overlap = true;
      await new Promise((r) => setTimeout(r, opts.delay ?? 0));
      const S = feeds.input_ids.dims[1], ids = Array.from(feeds.input_ids._d, Number), pos = Array.from(feeds.position_ids._d.slice(0, S), Number);
      const past = feeds["past_key_values.1.key"].dims[2];
      for (const k of ["past.0.conv", "past.0.recurrent", "past_key_values.1.key"]) if (feeds[k].disposed) throw new Error(`fed a disposed tensor ${k}`);
      s.calls.push({ S, ids, pos, past, fetches: [...fetches], feeds, mask: feeds.attention_mask.dims[1] });
      const out = {};
      if (fetches.includes("logits")) { const l = new Float32Array(S * V); for (let i = 0; i < S; i++) l[i * V + ((ids[i] + 1) % V)] = 5; out.logits = new Tensor("float32", l, [1, S, V], "gpu-buffer"); }
      if (fetches.includes("hidden_states")) { const h = new Float32Array(S * H); for (let i = 0; i < S; i++) for (let j = 0; j < H; j++) h[i * H + j] = pos[i] * 10 + j; out.hidden_states = new Tensor("float32", h, [1, S, H], "gpu-buffer"); }
      const mk = (dims) => { const t = new Tensor("float16", new Uint16Array(dims.reduce((a, b) => a * b, 1)), dims, "gpu-buffer"); s.made.push(t); return t; };
      out["present.1.key"] = mk([1, 4, past + S, 8]); out["present.1.value"] = mk([1, 4, past + S, 8]);
      out["present.0.conv"] = mk([1, 6, 3]); out["present.0.recurrent"] = mk([1, 2, 4, 4]);
      s.running--; return out;
    },
    async release() { s.released = true; } };
  return s;
}
const gen = async (it) => { const g = []; for await (const t of it) g.push(t); return g; };
const G = { temperature: 0, repetitionPenalty: 1 };

test("chunked prefill, absolute positions, hidden at requested positions, logits of the last token", async () => {
  const s = mkSession(), be = genaiFromSession(s, ort, O({ chunk: 4, minBlock: 0 }));
  assert.equal(be.kind, "genai"); assert.equal(be.hiddenName, "hidden_states");
  const r = await be.prefill([3, 4, 5, 6, 7, 8, 9, 10, 11, 12], { positions: [9, 3, 3, 0] });
  assert.deepEqual(s.calls.map((c) => c.S), [4, 4, 2]);
  assert.deepEqual(s.calls.map((c) => c.past), [0, 4, 8]);
  assert.ok(s.calls.every((c) => c.mask === c.past + c.S));
  assert.deepEqual(Array.from(r.hiddens[0]), [90, 91, 92]); assert.equal(r.hiddens[1][1], 31); assert.equal(r.hiddens[3][2], 2);
  assert.equal(r.lastLogits[(12 + 1) % V], 5);
  assert.ok(!s.calls[0].fetches.includes("logits") && s.calls[2].fetches.includes("logits"));
  assert.ok(!s.calls[1].fetches.includes("hidden_states"));
  assert.equal(r.state, undefined, "no state without keep");
  await assert.rejects(be.prefill([1, 2], { positions: [2] }), /out of/);
});

test("manifest hiddenOutput must exist in the graph; null = logits only", async () => {
  assert.throws(() => genaiFromSession(mkSession(), ort, O({ hiddenOutput: "/model/norm/Mul_1_output_0" })), /not an output of the graph/);
  const be = genaiFromSession(mkSession(), ort, O({ hiddenOutput: null }));
  assert.equal(be.hiddenName, null);
  await assert.rejects(be.prefill([1, 2], { positions: [0] }), /no hidden output/);
  assert.throws(() => genaiFromSession(mkSession(), ort, { hiddenOutput: null }), /eos/);
});

test("greedy generate stops at the manifest EOS", async () => {
  const s = mkSession(), be = genaiFromSession(s, ort, O({ chunk: 8 }));
  assert.deepEqual(await gen(be.generate([1, 2, 3], { maxNewTokens: 5, ...G })), [4, 5, 6, 7, 8]);
  assert.deepEqual(await gen(be.generate([12], { maxNewTokens: 9, ...G })), [13, 14], "15 is EOS");
});

test("kept state: prefill(keep) keeps its tensors alive; generateFrom continues from them and frees only derived states", async () => {
  const s = mkSession(), be = genaiFromSession(s, ort, O({ chunk: 4, minBlock: 0 }));
  const ids = [1, 2, 3, 4, 5, 6];
  const r = await be.prefill(ids, { positions: [5] }, { keep: true });
  assert.ok(r.state && r.state.ids.length === 6);
  const kept = s.calls.at(-1); // last prefill chunk produced the kept state
  const keptTensors = s.made.slice(-4);
  assert.ok(keptTensors.every((t) => !t.disposed), "kept state not disposed");
  const n0 = s.calls.length;
  const g = await gen(be.generateFrom(r.state, [...ids, 7, 8], { maxNewTokens: 3, ...G }));
  assert.deepEqual(g, [9, 10, 11]);
  const first = s.calls[n0];
  assert.equal(first.past, 6, "continuation starts after the kept prefix"); assert.deepEqual(first.ids, [7, 8]);
  assert.equal(first.feeds["past.0.conv"], keptTensors[2], "fed the kept tensors");
  assert.ok(keptTensors.every((t) => !t.disposed), "kept state still alive after generateFrom");
  assert.ok(s.made.slice(-4).every((t) => t.disposed), "derived states freed");
  // full prefill + generate gives the same tokens (shared vs fresh)
  assert.deepEqual(await gen(be.generate([...ids, 7, 8], { maxNewTokens: 3, ...G })), g);
  // second continuation from the same kept state
  assert.deepEqual(await gen(be.generateFrom(r.state, [...ids, 1], { maxNewTokens: 2, ...G })), [2, 3]);
  await assert.rejects(gen(be.generateFrom(r.state, [9, 9, 9, 9, 9, 9, 9], {})), /extend the kept prefix/);
  void kept;
});

test("snapshot -> prefillFrom(keep) twice: independent kept states; snapshot never freed by prefillFrom", async () => {
  const s = mkSession(), be = genaiFromSession(s, ort, O({ chunk: 4, minBlock: 0 }));
  const h = await be.snapshot([1, 2, 3]);
  const snapT = s.made.slice(-4);
  const a = await be.prefillFrom(h, [1, 2, 3, 4, 5], { positions: [4] }, { keep: true });
  const aT = s.made.slice(-4);
  const b = await be.prefillFrom(h, [1, 2, 3, 6], { positions: [] }, { keep: true });
  assert.ok(snapT.every((t) => !t.disposed) && aT.every((t) => !t.disposed));
  assert.deepEqual(Array.from(a.hiddens[0]), [40, 41, 42]);
  assert.equal(a.lastLogits[6], 5); assert.equal(b.lastLogits[7], 5);
  await h.dispose();
  assert.ok(snapT.every((t) => t.disposed), "snapshot disposed explicitly");
  assert.ok(aT.every((t) => !t.disposed), "kept state outlives the snapshot");
  await a.state.dispose(); assert.ok(aT.every((t) => t.disposed));
  await assert.rejects(be.prefillFrom(h, [1, 2, 3, 4], { positions: [] }), /unknown or released/);
  await assert.rejects(gen(be.generateFrom(a.state, [1, 2, 3, 4, 5, 6], {})), /unknown or released/);
  await assert.rejects(be.prefillFrom(b.state, [1, 2, 9, 6, 1], { positions: [] }), /extend the kept prefix/);
  await assert.rejects(be.prefillFrom(b.state, [1, 2, 3, 6], { positions: [] }), /at least one token/);
});

test("calls are serialized; abort mid-prefill frees the state; dispose releases", async () => {
  const s = mkSession({ delay: 2 }), be = genaiFromSession(s, ort, O({ chunk: 2 }));
  const ac = new AbortController(), g = [];
  for await (const t of be.generate([1], { maxNewTokens: 100, ...G, signal: ac.signal })) { g.push(t); if (g.length === 3) ac.abort(); }
  assert.equal(g.length, 3);
  await Promise.all([gen(be.generate([1], { maxNewTokens: 4, ...G })), be.prefill([1, 2, 3], { positions: [1] })]);
  assert.equal(s.overlap, false);
  await be.dispose(); assert.ok(s.released);
  await assert.rejects(be.prefill([1], { positions: [] }), /disposed/);
});

test("EP placement checks and createGenai with graph bytes", async () => {
  const p = parsePlacement(["Node(s) placed on [WebGpuExecutionProvider]. Number of nodes: 900", "Node(s) placed on [CPUExecutionProvider]. Number of nodes: 6"]);
  assert.deepEqual(p, { WebGpuExecutionProvider: 900, CPUExecutionProvider: 6 });
  assertWebGpuPlacement(p);
  // onnxruntime-web 1.31.1 (session_state.cc) when one EP takes the whole graph
  assert.deepEqual(parsePlacement(["2026-11-12 [V:onnxruntime:, session_state.cc:1581 VerifyEachNodeIsAssignedToAnEp]  All nodes placed on [WebGpuExecutionProvider]. Number of nodes: 16"]), { WebGpuExecutionProvider: 16 });
  assert.throws(() => assertWebGpuPlacement({ WebGpuExecutionProvider: 300, CPUExecutionProvider: 591 }), /CPU/);
  assert.throws(() => assertWebGpuPlacement({}), /could not verify/);
  await assert.rejects(createGenai("m", "d", "model.onnx.data", O({ providers: ["cpu"], ort: {} })), /webgpu.*wasm/);
  const created = [];
  const mod = { Tensor, env: { versions: { web: "1.30.0" } }, InferenceSession: { async create(model, o) { created.push([model, o.externalData]); return mkSession(); } } };
  const graph = new Uint8Array([1, 2, 3]);
  const be = await createGenai(graph, "https://h/d.data", "model.onnx.data", O({ ort: mod, warmupToken: 1 }));
  assert.equal(created[0][0], graph, "session created from verified bytes");
  assert.deepEqual(created[0][1], [{ path: "model.onnx.data", data: "https://h/d.data" }]);
  assert.equal(be.kind, "genai");
  await assert.rejects(createGenai(graph, "d", "x", O({ ort: { ...mod, env: { versions: { web: "1.29.0" } } } })), /too old/);
});
