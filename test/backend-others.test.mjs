// decode-only, transformers.js, LiteRT.js and WebLLM backends against mocks (ported from ct0-edge tests).
import test from "node:test";
import assert from "node:assert/strict";
import { createDecodeOnly, transformersJsFromModel, createLiteRt, buildMask, prefillChunks, embedFromF16Reader, webllmBackend, logitCapture, recoverHidden, canShareState } from "../dist/index.js";

const gen = async (it) => { const g = []; for await (const t of it) g.push(t); return g; };
const G = { temperature: 0, repetitionPenalty: 1 };

// ---- decode-only ---------------------------------------------------------------
class OTensor { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }
const V = 6, H = 4, HID = "/model/norm/Mul_1_output_0";
function dSession() {
  const M = (name, type, shape) => ({ name, type, shape });
  // State = running sum of fed token ids; logits peak at (sum % V); hidden = [pos, sum, 0, 0].
  const s = { inputNames: ["input_ids", "cache_position", "past_conv.0", "past_key_values.3.key"], outputNames: ["logits", "present_conv.0", "present.3.key", HID], runs: [], released: false,
    inputMetadata: [M("input_ids", "int64", [1, 1]), M("cache_position", "int64", [1]), M("past_conv.0", "float32", [1, 1]), M("past_key_values.3.key", "float32", [1, 1, "p", 1])],
    async release() { s.released = true; },
    async run(feed, names) {
      const id = Number(feed.input_ids.data[0]), pos = Number(feed.cache_position.data[0]), sum = feed["past_conv.0"].data[0] + id;
      s.runs.push({ id, pos, names });
      const L = new Float32Array(V); L[sum % V] = 5;
      const r = { logits: new OTensor("float32", L, [1, 1, V]), "present_conv.0": new OTensor("float32", Float32Array.of(sum), [1, 1]), "present.3.key": new OTensor("float32", Float32Array.of(pos), [1, 1, 1, 1]) };
      if (names.includes(HID)) r[HID] = new OTensor("float32", Float32Array.of(pos, sum, 0, 0), [1, 1, H]);
      return r;
    } };
  return s;
}
const DO = (o = {}) => ({ ort: { Tensor: OTensor }, eos: new Set([0]), hiddenOutput: HID, ...o });

test("decode-only: prefill one token per run, hidden by manifest name, EOS from the manifest", async () => {
  const s = dSession(), be = await createDecodeOnly(s, DO());
  assert.equal(be.hiddenName, HID); assert.ok(canShareState(be));
  const r = await be.prefill([1, 2, 3], { positions: [2, 0] });
  assert.deepEqual(s.runs.map((x) => [x.id, x.pos]), [[1, 0], [2, 1], [3, 2]]);
  assert.deepEqual(Array.from(r.hiddens[0]), [2, 6, 0, 0]); assert.deepEqual(Array.from(r.hiddens[1]), [0, 1, 0, 0]);
  assert.equal(r.lastLogits[0], 5);
  await assert.rejects(createDecodeOnly(dSession(), DO({ hiddenOutput: "hidden_states" })), /not an output of the graph/);
  await assert.rejects(createDecodeOnly(dSession(), { ort: { Tensor: OTensor }, hiddenOutput: null }), /eos/);
  // sum 1 -> emits 1, sum 2 -> 2, sum 4 -> 4 = EOS of this backend (not emitted)
  const be4 = await createDecodeOnly(dSession(), DO({ eos: new Set([4]) }));
  assert.deepEqual(await gen(be4.generate([1], { maxNewTokens: 10, ...G })), [1, 2]);
  assert.deepEqual(await gen(be.generate([1], { maxNewTokens: 5, ...G })), [1, 2, 4, 2, 4]);
});

test("decode-only: kept states are independent and continuation equals a fresh prefill", async () => {
  const s = dSession(), be = await createDecodeOnly(s, DO({ eos: new Set([99]) }));
  const h = await be.snapshot([1, 1]);
  const a = await be.prefillFrom(h, [1, 1, 2], { positions: [2] }, { keep: true });
  const b = await be.prefillFrom(h, [1, 1, 3], { positions: [] }, { keep: true });
  assert.deepEqual(Array.from(a.hiddens[0]), [2, 4, 0, 0]);
  assert.equal(b.lastLogits[5], 5);
  const n = s.runs.length;
  const g1 = await gen(be.generateFrom(a.state, [1, 1, 2, 1], { maxNewTokens: 3, ...G }));
  assert.equal(s.runs[n].pos, 3, "continues after the kept 3 tokens"); assert.equal(s.runs[n].id, 1);
  assert.deepEqual(g1, await gen(be.generate([1, 1, 2, 1], { maxNewTokens: 3, ...G })), "shared == fresh");
  assert.deepEqual(await gen(be.generateFrom(b.state, [1, 1, 3, 1], { maxNewTokens: 2, ...G })), await gen(be.generate([1, 1, 3, 1], { maxNewTokens: 2, ...G })));
  const k = await be.prefill([4, 4], { positions: [] }, { keep: true });
  assert.deepEqual(k.state.ids, [4, 4]);
  k.state.dispose();
  await assert.rejects(gen(be.generateFrom(k.state, [4, 4, 1], {})), /unknown or released/);
  await assert.rejects(be.prefillFrom(a.state, [9, 9, 9, 9], { positions: [] }), /extend the kept prefix/);
  await be.dispose(); assert.ok(s.released);
});

// ---- transformers.js -------------------------------------------------------------
function tModel(hid = "mul_13450") {
  const m = { forwards: [], disposed: false, sessions: { model: { outputNames: ["logits", hid] } },
    async forward(f) { m.forwards.push(f); const S = f.input_ids.dims[1];
      const h = new Float32Array(S * 3); for (let i = 0; i < S; i++) for (let j = 0; j < 3; j++) h[i * 3 + j] = i * 10 + j;
      const l = new Float32Array(S * 5); for (let i = 0; i < S; i++) l[i * 5 + (i % 5)] = 1 + i;
      return { logits: new OTensor("float32", l, [1, S, 5]), [hid]: new OTensor("float32", h, [1, S, 3]) }; },
    async generate({ streamer, max_new_tokens }) { streamer?.put([[1n, 2n]]); for (const t of [7n, 8n, 99n, 9n].slice(0, max_new_tokens)) streamer?.put([[t]]); streamer?.end?.(); },
    async dispose() { m.disposed = true; } };
  return m;
}
test("transformers.js: one forward per prefill, no kept state (falls back to fresh prefill)", async () => {
  const m = tModel(), be = transformersJsFromModel(m, OTensor, { eos: new Set([99]), hiddenOutput: "mul_13450" });
  assert.equal(be.kind, "transformers.js"); assert.equal(canShareState(be), false);
  const r = await be.prefill([1, 2, 3, 4], { positions: [2, 0] }, { keep: true });
  assert.equal(r.state, undefined);
  assert.equal(r.hiddens[0][1], 21); assert.equal(r.lastLogits[3], 4);
  assert.deepEqual(await gen(be.generate([1, 2], { maxNewTokens: 4, ...G })), [7, 8]);
  assert.throws(() => transformersJsFromModel(tModel(), OTensor, { eos: new Set([99]), hiddenOutput: "type_as_80" }), /not an output/);
  await be.dispose(); assert.ok(m.disposed);
});

// ---- LiteRT.js ---------------------------------------------------------------------
class LTensor {
  constructor(data, shape, dtype) { this.data = data; this.shape = shape; this.dtype = dtype; this.deleted = false; }
  toTypedArray() { return this.data; } async moveTo() { return this; } async copyTo() { return new LTensor(this.data.slice(), this.shape); }
  delete() { this.deleted = true; }
}
const LH = 2, LV = 6, CACHE = 64, P = 4;
function liteRt() {
  const log = [];
  const io = (rows) => ({ inputs: [{ name: "embeddings", shape: [1, rows, LH], dtype: "float32" }, { name: "input_pos", shape: [rows], dtype: "int32" },
    { name: "mask", shape: [1, 1, rows, CACHE], dtype: "float32" }, { name: "kv_cache_r_0", shape: [1, 1], dtype: "float32" }],
    outputs: [{ name: "kv_cache_r_0", shape: [1, 1], dtype: "float32" }, { name: "hidden", shape: [1, rows, LH], dtype: "float32" }] });
  const manifest = { format: "ct0-litert-bundle/1", vocab: LV, hidden: LH, signatures: { [`prefill_${P}`]: io(P), decode: io(1) } };
  manifest.signatures.decode.outputs.push({ name: "logits", shape: [1, 1, LV], dtype: "float32" });
  const model = { async run(sig, feed) {
    const pos = feed.input_pos.data, e = feed.embeddings.data;
    log.push({ sig, pos: Array.from(pos) });
    let r = feed.kv_cache_r_0.data[0];
    const hid = new Float32Array(pos.length * LH);
    for (let i = 0; i < pos.length; i++) { if (i === 0 || pos[i] > pos[i - 1]) r = 0.5 * r + e[i * LH]; hid[i * LH] = r; hid[i * LH + 1] = pos[i]; }
    const out = { kv_cache_r_0: new LTensor(Float32Array.of(r), [1, 1]), hidden: new LTensor(hid, [1, pos.length, LH]) };
    if (sig === "decode") out.logits = new LTensor(Float32Array.from({ length: LV }, (_, i) => -Math.abs(i - (Math.round(r) % LV))), [1, 1, LV]);
    return out;
  }, delete() {} };
  const lrt = { Tensor: LTensor, async loadAndCompile() { return model; } };
  const embed = { async rows(ids) { const o = new Float32Array(ids.length * LH); ids.forEach((id, k) => o.fill(id, k * LH, k * LH + LH)); return o; } };
  return { log, mk: (o = {}) => createLiteRt(lrt, new Uint8Array(4), manifest, embed, { quiet: true, accelerator: "wasm", eos: new Set([99]), hiddenOutput: "hidden", ...o }) };
}
test("litert: helpers, hidden output named by the manifest, kept states and generateFrom", async () => {
  assert.deepEqual(prefillChunks(9, 4), [{ start: 0, len: 4 }, { start: 4, len: 4 }, { start: 8, len: 1 }]);
  const m = buildMask(Int32Array.of(2, 0), 4, "float32"); assert.ok(m[0] === 0 && m[2] === 0 && m[3] < -1e29);
  const f16 = new Uint16Array([0x3c00, 0xc000, 0x0000, 0x3800]), b = new Uint8Array(f16.buffer);
  assert.deepEqual(Array.from(await embedFromF16Reader(async (o, n) => b.subarray(o, o + n), 2).rows([1, 0])), [0, 0.5, 1, -2]);
  const t = liteRt();
  await assert.rejects(t.mk({ hiddenOutput: "hidden_states" }), /not a decode output/);
  const be = await t.mk();
  assert.equal(be.hiddenName, "hidden"); assert.ok(canShareState(be));
  const ids = [3, 1, 4, 1, 5, 9];
  const r = await be.prefill(ids, { positions: [1, 5] }, { keep: true });
  assert.equal(r.hiddens[0][1], 1); assert.equal(r.hiddens[1][1], 5);
  assert.deepEqual(r.state.ids, ids);
  const g1 = await gen(be.generateFrom(r.state, [...ids, 2], { maxNewTokens: 3, ...G }));
  const g2 = await gen(be.generate([...ids, 2], { maxNewTokens: 3, ...G }));
  assert.deepEqual(g1, g2, "shared == fresh");
  const h = await be.snapshot([3, 1, 4]);
  const a = await be.prefillFrom(h, [3, 1, 4, 1, 5], { positions: [4] });
  const full = await be.prefill([3, 1, 4, 1, 5], { positions: [4] });
  assert.deepEqual(Array.from(a.hiddens[0]), Array.from(full.hiddens[0]));
  await assert.rejects(t.mk({ eos: undefined }), /eos/);
});

// ---- WebLLM ---------------------------------------------------------------------------
test("webllm: logits capture, hidden recovery, EOS from the manifest, no kept state", async () => {
  const Vw = 40, D = 2, rows = Int32Array.from([3, 7, 11]);
  const Pm = new Float32Array(D * rows.length); Pm[0] = 1; Pm[1 * rows.length + 1] = 1; // h = [L[3], L[7]]
  const cap = logitCapture();
  const calls = [];
  const engine = {
    async resetChat() {}, async interruptGenerate() {},
    completions: { async create(req) {
      calls.push(req);
      const n = (req.prompt.match(/</g) || []).length;
      if (req.stream) {
        let i = 0; const toks = [21, 22, 30, 23];
        return { [Symbol.asyncIterator]() { return this; }, async next() { if (i >= toks.length) return { done: true }; const t = toks[i++]; cap.processor.processSampledToken(t); return { done: false, value: {} }; }, async return() { return { done: true }; } };
      }
      const L = new Float32Array(Vw); L[3] = n; L[7] = 2 * n; cap.processor.processLogits(L);
      return { usage: { prompt_tokens: n } };
    } },
  };
  const tok = { decode: (ids) => ids.map((i) => `<${i}>`).join("") };
  assert.throws(() => webllmBackend(engine, cap, { tok }), /eos/);
  const be = webllmBackend(engine, cap, { tok, eos: new Set([30]), recovery: { rows, P: Pm } });
  assert.equal(be.hiddenName, "recovered-from-logits"); assert.equal(canShareState(be), false);
  const r = await be.prefill([5, 6, 7], { positions: [0, 2] }, { keep: true });
  assert.equal(r.state, undefined);
  assert.deepEqual(Array.from(r.hiddens[0]), [1, 2]); assert.deepEqual(Array.from(r.hiddens[1]), [3, 6]);
  assert.deepEqual(Array.from(recoverHidden({ rows, P: Pm }, r.lastLogits)), [3, 6]);
  assert.deepEqual(await gen(be.generate([5], { maxNewTokens: 10 })), [21, 22]);
});
