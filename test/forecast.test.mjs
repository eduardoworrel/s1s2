// Forecast reader (spec/FORECAST.md): head arithmetic on hand-computed numbers, pooling, manifest rules, one prefill
// for decide() + forecast() + generate(), and the "forecast unsupported" errors. All on mocks.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import {
  S1S2, loadForecast, validateManifest, ManifestError, genaiFromSession, ForecastReader, loadForecastHead, forecastScores, forecastPositions, poolRows,
} from "../dist/index.js";
import { schemaProblems } from "../dist/package.js";
import { toy, toyManifest, clone, TOY_PIECES, TOY_ID } from "./helpers.mjs";

/** safetensors fp32 bytes from {name: {shape, data}}. */
function st(tensors) {
  const header = {}; const parts = []; let off = 0;
  for (const [n, t] of Object.entries(tensors)) {
    const b = Buffer.alloc(t.data.length * 4); t.data.forEach((x, i) => b.writeFloatLE(x, i * 4));
    header[n] = { dtype: "F32", shape: t.shape, data_offsets: [off, off + b.length] }; off += b.length; parts.push(b);
  }
  const h = Buffer.from(JSON.stringify(header)), len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(h.length));
  const all = Buffer.concat([len, h, ...parts]);
  return all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength);
}
const sig = (x) => 1 / (1 + Math.exp(-x));
const near = (a, b, t = 1e-6) => Math.abs(a - b) <= t;
const H = 2;
const sha = "a".repeat(64), rev = "b".repeat(40);

const spec = (over = {}) => ({
  id: "gate-v1", layer: 3, pooling: "last",
  head: { type: "linear", file: { path: "forecast.safetensors", sha256: sha }, dims: { in: H, out: 2 } },
  behaviors: [{ name: "refuse" }, { name: "list", description: "numbered list" }],
  calibration: { temperature: 1 }, trainedOn: { baseId: "x", weightsSha256: sha }, ...over,
});
const linearBytes = () => st({ mean: { shape: [2], data: [1, 2] }, std: { shape: [2], data: [2, 4] }, weight: { shape: [2, 2], data: [1, 0, 0.5, -1] }, bias: { shape: [2], data: [0.25, -0.5] } });

test("linear head: x = (h - mean) / std, z = W x + b, z / T, a z + b, sigmoid (hand-computed)", () => {
  // h = [5, 10]: x = [(5-1)/2, (10-2)/4] = [2, 2]; z0 = 2 + 0.25 = 2.25; z1 = 0.5*2 - 2 - 0.5 = -1.5
  const hd = loadForecastHead(linearBytes(), "linear", { in: 2, out: 2 });
  const z = forecastScores(hd, Float32Array.from([5, 10]));
  assert.ok(near(z[0], 2.25) && near(z[1], -1.5));
  const plain = new ForecastReader(spec(), hd).read([Float32Array.from([5, 10])], "q", 7);
  assert.ok(near(plain.behaviors.refuse.logit, 2.25) && near(plain.behaviors.refuse.p, sig(2.25)));
  assert.ok(near(plain.behaviors.list.p, sig(-1.5)));
  assert.equal(plain.layer, 3); assert.equal(plain.cost.extraPrefillTokens, 0); assert.equal(plain.cost.prefillTokens, 7);
  // temperature 2 then Platt a = 3, b = -1 for the first behavior (per-behavior arrays)
  const cal = new ForecastReader(spec({ calibration: { temperature: [2, 1], platt: { a: [3, 1], b: [-1, 0] } } }), hd).read([Float32Array.from([5, 10])], "q", 7);
  assert.ok(near(cal.behaviors.refuse.logit, 3 * (2.25 / 2) - 1), String(cal.behaviors.refuse.logit));
  assert.ok(near(cal.behaviors.list.logit, -1.5));
});

test("mlp head: z = w2 relu(w1 x + b1) + b2 (hand-computed)", () => {
  // x = [2, 2] again; w1 = [[1,1],[-1,-1]], b1 = [0, 0.5] -> [4, -3.5] -> relu [4, 0]; w2 = [[0.5, 1],[1, 1]], b2 = [0, 1] -> [2, 5]
  const b = st({ mean: { shape: [2], data: [1, 2] }, std: { shape: [2], data: [2, 4] }, w1: { shape: [2, 2], data: [1, 1, -1, -1] }, b1: { shape: [2], data: [0, 0.5] },
    w2: { shape: [2, 2], data: [0.5, 1, 1, 1] }, b2: { shape: [2], data: [0, 1] } });
  const hd = loadForecastHead(b, "mlp", { in: 2, out: 2, hidden: 2 });
  const z = forecastScores(hd, Float32Array.from([5, 10]));
  assert.ok(near(z[0], 2) && near(z[1], 5));
});

test("head loading errors name the mismatch", () => {
  assert.throws(() => loadForecastHead(linearBytes(), "linear", { in: 3, out: 2 }), /mean has shape \[2\], expected \[3\]/);
  assert.throws(() => loadForecastHead(linearBytes(), "mlp", { in: 2, out: 2, hidden: 2 }), /missing tensor w1/);
  const zero = st({ mean: { shape: [2], data: [0, 0] }, std: { shape: [2], data: [1, 0] }, weight: { shape: [1, 2], data: [1, 1] }, bias: { shape: [1], data: [0] } });
  assert.throws(() => loadForecastHead(zero, "linear", { in: 2, out: 1 }), /std\[1\] = 0/);
  assert.throws(() => ForecastReader.load(spec({ head: { ...spec().head, dims: { in: 2, out: 2 } } }), new Uint8Array(linearBytes()), 4), /input width 2 != 4/);
  assert.throws(() => ForecastReader.load(spec({ behaviors: [{ name: "a" }] }), new Uint8Array(linearBytes()), 2), /2 outputs for 1 behaviors/);
});

test("pooling: last, mean, bins32 (hand-computed) and the positions each one reads", () => {
  const rows = Array.from({ length: 64 }, (_, i) => Float32Array.from([i, 2 * i]));
  assert.deepEqual(Array.from(poolRows("last", [rows[63]])), [63, 126]);
  assert.deepEqual(Array.from(poolRows("mean", rows)), [31.5, 63]);
  const b = poolRows("bins32", rows); // 64 tokens -> bins of 2: bin j = mean(2j, 2j+1) = 2j + 0.5
  assert.equal(b.length, 64);
  assert.deepEqual([b[0], b[1], b[2], b[3], b[62], b[63]], [0.5, 1, 2.5, 5, 62.5, 125]);
  const odd = poolRows("bins32", Array.from({ length: 33 }, (_, i) => Float32Array.from([i]))); // bins [floor(33j/32), floor(33(j+1)/32))
  assert.equal(odd.length, 32); assert.equal(odd[0], 0); assert.equal(odd[31], 31.5); // bin 31 = [31, 33) -> mean(31, 32)
  assert.deepEqual(forecastPositions("last", 10), [9]);
  assert.deepEqual(forecastPositions("mean", 3), [0, 1, 2]);
  assert.throws(() => forecastPositions("bins32", 31), /at least 32 tokens/);
  assert.throws(() => poolRows("bins32", rows.slice(0, 5)), /at least 32/);
});

// ---- manifest ----
const withForecast = (m, f = spec()) => { const c = clone(m); c.hiddenSize = H; c.forecast = { ...f, trainedOn: { ...f.trainedOn, baseId: c.id } }; return c; };
const bad = (mut, re) => { const m = withForecast(toyManifest()); mut(m); assert.throws(() => validateManifest(m), (e) => e instanceof ManifestError && re.test(e.message), re.toString()); };
const withBuild = (m) => { m.builds["genai-int4-webgpu"].forecastOutput = "aux_hidden_states"; m.builds["genai-int4-webgpu"].forecastLayer = 3; return m; };

test("manifest: forecast block and build forecastOutput validate; layer.read accepts an integer", () => {
  const m = validateManifest(withBuild(withForecast(toyManifest())));
  assert.equal(m.forecast.layer, 3);
  const c = clone(toyManifest()); c.layer.read = 7; assert.equal(validateManifest(c).layer.read, 7);
  c.layer.read = "middle"; assert.throws(() => validateManifest(c), /layer:/);
});

test("manifest: forecast errors", () => {
  bad((m) => { m.forecast.pooling = "max"; }, /forecast.pooling/);
  bad((m) => { m.forecast.head.dims.in = 5; }, /dims.in: 5 != 1 x hiddenSize \(2\)/);
  bad((m) => { m.forecast.pooling = "bins32"; }, /dims.in: 2 != 32 x hiddenSize/);
  bad((m) => { m.forecast.head.dims.out = 3; }, /dims.out: 3 != behaviors.length \(2\)/);
  bad((m) => { m.forecast.head.type = "mlp"; }, /dims.hidden: required for an mlp head/);
  bad((m) => { m.forecast.head.file.sha256 = "xyz"; }, /head.file/);
  bad((m) => { m.forecast.calibration.temperature = 0; }, /temperature/);
  bad((m) => { m.forecast.calibration.platt = { a: 1 }; }, /platt/);
  bad((m) => { m.forecast.behaviors.push({ name: "refuse" }); m.forecast.head.dims.out = 3; }, /names must be unique/);
  bad((m) => { m.forecast.trainedOn.weightsSha256 = "1"; }, /trainedOn/);
  bad((m) => { m.forecast.extra = 1; }, /forecast.extra: unknown key/);
  bad((m) => { m.builds["genai-int4-webgpu"].forecastOutput = "aux"; }, /forecastOutput and forecastLayer go together/);
  bad((m) => { withBuild(m); m.builds["genai-int4-webgpu"].forecastLayer = 4; }, /forecastLayer: 4 != forecast.layer \(3\)/);
  bad((m) => { withBuild(m); m.builds["genai-int4-webgpu"].runtime = "webllm"; }, /no extra graph output \(forecast unsupported\)/);
});

test("backward compatibility: every shipped manifest and example still validates, and none gains a forecast", () => {
  const dirs = ["../manifests/", "../spec/examples/s1s2-toy-smollm2-135m/"];
  let n = 0;
  for (const d of dirs) for (const f of readdirSync(new URL(d, import.meta.url)).filter((x) => /^manifest.*\.json$|\.json$/.test(x))) {
    let j; try { j = JSON.parse(readFileSync(new URL(d + f, import.meta.url), "utf8")); } catch { continue; }
    if (j.schema !== "s1s2.manifest/1") continue;
    assert.equal(j.forecast, undefined);
    if (j.modalities && !j.modalities.includes("text")) continue; // media packages are validated by validateMediaManifest
    assert.doesNotThrow(() => validateManifest(j), f); n++;
  }
  assert.ok(n >= 3, `checked ${n} manifests`);
});

test("S1 refuses an integer layer.read at load (the forecast layer is separate)", () => {
  const m = toyManifest(); m.layer.read = 5;
  assert.throws(() => S1S2.fromBackend(m, "genai-int4-webgpu", { kind: "fake", hiddenName: null, prefill: async () => ({}), dispose: async () => {}, generate: async function* () {} }, toy), /layer.read is 5; this runtime reads the S1 heads on the final hidden state only/);
});

// ---- one prefill ----
const V = TOY_PIECES.length;
class Tensor { constructor(type, data, dims) { this.type = type; this.dims = dims; this._d = data; } async getData() { return this._d; } get data() { return this._d; } dispose() {} }
function mkSession({ aux = true } = {}) {
  const meta = [{ name: "input_ids", type: "int64", shape: [1, "s"] }, { name: "attention_mask", type: "int64", shape: [1, "t"] },
    { name: "position_ids", type: "int64", shape: [3, 1, "s"] }, { name: "past_key_values.0.key", type: "float32", shape: [1, 1, "p", 1] },
    { name: "past_key_values.0.value", type: "float32", shape: [1, 1, "p", 1] }];
  const s = { runs: [], fetches: [], fed: [], inputNames: meta.map((m) => m.name), inputMetadata: meta,
    outputNames: ["logits", "present.0.key", "present.0.value", "hidden_states", ...(aux ? ["aux_hidden_states"] : [])],
    async run(feeds, fetches) {
      const S = feeds.input_ids.dims[1], ids = Array.from(feeds.input_ids._d, Number), past = feeds["past_key_values.0.key"].dims[2];
      s.runs.push(S); s.fetches.push([...fetches]); if (past === s.fed.length) s.fed.push(...ids);
      const kv = new Float32Array(past + S); kv.set(feeds["past_key_values.0.key"]._d);
      for (let i = 0; i < S; i++) kv[past + i] = ids[i];
      const out = { "present.0.key": new Tensor("float32", kv, [1, 1, past + S, 1]), "present.0.value": new Tensor("float32", kv.slice(), [1, 1, past + S, 1]) };
      // layer-3 row of token at position p: [token id, position]
      if (fetches.includes("aux_hidden_states")) out.aux_hidden_states = new Tensor("float32", Float32Array.from(ids.flatMap((t, i) => [t, past + i])), [1, S, 2]);
      if (fetches.includes("hidden_states")) out.hidden_states = new Tensor("float32", new Float32Array(S * 2), [1, S, 2]);
      if (fetches.includes("logits")) { const l = new Float32Array(S * V); for (let i = 0; i < S; i++) l[i * V + (ids[i] % V)] = 1; out.logits = new Tensor("float32", l, [1, S, V]); }
      return out;
    }, async release() {} };
  return s;
}
const EOS = new Set([TOY_ID.get("<|im_end|>")]);
const req = { input: "some text", questions: [{ name: "q1", type: "predicate", instructions: "Is it?" }] };
const boot = (o = {}) => {
  const s = mkSession(o.session);
  const be = genaiFromSession(s, { Tensor }, { eos: EOS, hiddenOutput: "hidden_states", forecastOutput: o.noExtra ? null : "aux_hidden_states", chunk: 128, minBlock: 0 });
  const m = withBuild(withForecast(toyManifest(), spec(o.spec)));
  const reader = o.noExtra ? undefined : ForecastReader.load(m.forecast, new Uint8Array(linearBytes()), H);
  return { s, be, model: S1S2.fromBackend(m, "genai-int4-webgpu", be, toy, {}, "cpu", undefined, undefined, reader) };
};

test("decide + forecast + generate: the prompt is prefilled once; forecast() makes no backend call", async () => {
  const { s, model } = boot();
  const session = await model.decide(req, { forecast: true });
  const n = s.runs.reduce((a, b) => a + b, 0);
  assert.ok(session.answers.length === 1 && n > 0);
  const before = s.runs.length;
  const f = session.forecast();
  assert.equal(s.runs.length, before, "forecast() ran nothing");
  assert.deepEqual(Object.keys(f.behaviors), ["refuse", "list"]);
  assert.equal(f.cost.prefillTokens, n); assert.equal(f.cost.extraPrefillTokens, 0); assert.equal(f.question, "q1");
  const g = await session.generate("hi", { maxNewTokens: 2, temperature: 0, repetitionPenalty: 1 });
  assert.equal(g.sharedPrefill, true); assert.equal(g.reusedTokens, n, "System 2 continued from the decide prefill");
  // the prompt tokens were fed exactly once; generate() fed only the continuation and the new tokens
  const promptFeeds = s.runs.slice(0, before).reduce((a, b) => a + b, 0);
  assert.equal(promptFeeds, n);
  await session.dispose();
});

test("forecast values: the head reads the layer row of the last prompt token", async () => {
  const { model, s } = boot();
  const { forecast, session } = await model.forecast(req);
  const n = forecast.cost.prefillTokens, last = s.fed[n - 1];
  // the mock row is [token id, position]; x = ((id - 1) / 2, (pos - 2) / 4); z0 = x0 + 0.25; z1 = 0.5 x0 - x1 - 0.5
  const x0 = (last - 1) / 2, x1 = (n - 1 - 2) / 4;
  assert.ok(near(forecast.behaviors.refuse.logit, x0 + 0.25, 1e-5), `${forecast.behaviors.refuse.logit} vs ${x0 + 0.25}`);
  assert.ok(near(forecast.behaviors.list.logit, 0.5 * x0 - x1 - 0.5, 1e-5));
  assert.equal(forecast.pooling, "last");
  await session.dispose();
});

test("without forecast:true the extra output is not fetched and S1 is unchanged", async () => {
  const { s, model } = boot();
  const session = await model.decide(req);
  assert.ok(s.fetches.every((f) => !f.includes("aux_hidden_states")));
  assert.throws(() => session.forecast(), /did not read the forecast layer/);
  await session.dispose();
});

test("forecast unsupported: build without the layer, runtimes without extra outputs, missing graph output", async () => {
  const { model } = boot({ noExtra: true });
  await assert.rejects(model.decide(req, { forecast: true }), /forecast unsupported: build .* exposes no forecast layer/);
  assert.match(model.forecastUnavailable, /forecast unsupported/);
  const noManifest = S1S2.fromBackend(toyManifest(), "genai-int4-webgpu", mkBackendOnly(), toy);
  await assert.rejects(noManifest.decide(req, { forecast: true }), /the manifest has no forecast section/);
  // a genai graph without the declared output fails at construction, naming the output
  assert.throws(() => genaiFromSession(mkSession({ aux: false }), { Tensor }, { eos: EOS, hiddenOutput: "hidden_states", forecastOutput: "aux_hidden_states" }), /hiddenOutput "aux_hidden_states" is not an output of the graph/);
  // the genai backend without the layer rejects a request for it
  const be = genaiFromSession(mkSession({ aux: false }), { Tensor }, { eos: EOS, hiddenOutput: "hidden_states" });
  await assert.rejects(be.prefill([3, 4], { positions: [], extra: { positions: [1] } }), /forecast unsupported/);
});
function mkBackendOnly() { return genaiFromSession(mkSession(), { Tensor }, { eos: EOS, hiddenOutput: "hidden_states", minBlock: 0 }); }

test("fp16 overflow: a non-finite forecast row is refused with a clear error", async () => {
  const s = mkSession(); const run = s.run;
  s.run = async (f, fe) => { const o = await run(f, fe); if (o.aux_hidden_states) o.aux_hidden_states = new Tensor("float32", Float32Array.from([Infinity, 0]), [1, 1, 2]); return o; };
  const be = genaiFromSession(s, { Tensor }, { eos: EOS, hiddenOutput: "hidden_states", forecastOutput: "aux_hidden_states", minBlock: 0 });
  await assert.rejects(be.prefill([3], { positions: [], extra: { positions: [0] } }), /not finite/);
});

test("mean pooling reads every prompt position (and so disables prefix sharing)", async () => {
  const { s, model } = boot({ spec: { pooling: "mean" } });
  const session = await model.decide(req, { forecast: true });
  const f = session.forecast();
  assert.equal(f.pooling, "mean");
  assert.ok(s.fetches[0].includes("aux_hidden_states"));
  await session.dispose();
});

test("load checks: the build must expose the manifest's layer; the head must be trained on this manifest id", async () => {
  const m = validateManifest(withBuild(withForecast(toyManifest())));
  const files = { bytes: async () => new Uint8Array(linearBytes()) };
  const b = m.builds["genai-int4-webgpu"];
  assert.ok(await loadForecast(m, b, files, () => {}));
  await assert.rejects(loadForecast(m, { ...b, forecastLayer: 5 }, files, () => {}), /build exposes layer 5, the manifest forecast reads layer 3/);
  const other = clone(m); other.forecast.trainedOn.baseId = "other-model";
  await assert.rejects(loadForecast(other, b, files, () => {}), /trained on "other-model", this manifest is/);
  assert.equal(await loadForecast(m, { ...b, forecastOutput: undefined, forecastLayer: undefined }, files, () => {}), undefined);
});

test("the JSON Schema accepts the forecast block and rejects a bad one", async () => {
  const good = withBuild(withForecast(toyManifest()));
  assert.deepEqual((await schemaProblems(good)).filter((x) => /forecast|layer/.test(x)), []);
  const badm = clone(good); badm.forecast.pooling = "max";
  assert.ok((await schemaProblems(badm)).some((x) => /forecast/.test(x)));
  const c = clone(good); c.layer.read = 4; assert.deepEqual((await schemaProblems(c)).filter((x) => /layer/.test(x)), []);
});
