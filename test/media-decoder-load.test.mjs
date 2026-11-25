// loadMedia with the decoder, on a fake ONNX Runtime: the three `decoder` modes ("auto", true, false), the real ortDecoder wrapper
// (cache inputs, float16 / float32 / array logits, blocked prefill), and dispose() while a question is running.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import path from "node:path";
import { loadMedia, ortDecoder, f32ToF16Bits } from "../dist/media.js";
import { safetensors } from "./media-helpers.mjs";
import { clone } from "./helpers.mjs";

const MANIFEST = JSON.parse(readFileSync(new URL("../manifests/marimba-audio.json", import.meta.url), "utf8"));
const H = 1024, V = 12, FRAMES = 104, LAYERS = 2, KVH = 2, HD = 8;

// ---- a package on disk: tiny files with the right hashes ----------------------------------------------------------------------------------

function pkg({ cpuBuild = true } = {}) {
  const dir = mkdtempSync(path.join(tmpdir(), "s1s2-dec-"));
  const put = (rel, bytes) => {
    const f = path.join(dir, rel); mkdirSync(path.dirname(f), { recursive: true }); writeFileSync(f, bytes);
    return { path: rel, sha256: createHash("sha256").update(bytes).digest("hex"), size: bytes.length };
  };
  const m = clone(MANIFEST);
  const a = m.audio;
  a.tower.file = put("audio_tower.onnx", "tower");
  a.tower.outputs = { audio_features: { layer: "final", frames: FRAMES, dim: H } };
  const w = safetensors({ coef: { dtype: "F32", shape: [1, H], data: new Array(H).fill(0.01) }, intercept: { dtype: "F32", shape: [1], data: [0] } });
  a.readers = {
    turn_end: { kind: "pooled-linear", type: "predicate", tap: "audio_features", pooling: { kind: "tail-mean", seconds: 2 }, weights: put("turn_end.safetensors", Buffer.from(w instanceof ArrayBuffer ? new Uint8Array(w) : w)), tensors: { coef: "coef", intercept: "intercept" }, calibration: { kind: "none" } },
    language: { kind: "decoder-language", type: "choice", tap: "audio_features" },
    term: { kind: "decoder-term", type: "predicate", tap: "audio_features" },
  };
  const d = a.decoder;
  d.graph = put("decoder/model.onnx", "decoder"); d.data = put("decoder/model.onnx.data", "data");
  if (cpuBuild) d.cpuBuild = { graph: put("decoder/cpu/model.onnx", "decoder"), data: put("decoder/cpu/model.onnx.data", "data") };
  else { delete d.cpuBuild; d.devices = ["webgpu"]; }
  d.embeddings = put("decoder/embed.bin", Buffer.alloc(V * H * 2));
  d.tokens = put("decoder/tokens.json", JSON.stringify({ tokens: Array.from({ length: V }, (_, i) => (i < 10 ? "abcdefghij"[i] : "")), added: [], special: [10, 11] }));
  d.model = { layers: LAYERS, kvHeads: KVH, headDim: HD, hidden: H, vocab: V };
  d.prompt = { before: [1, 2], after: [3, 4], audioFrames: FRAMES, languageToken: 5 };
  d.eos = [10, 11]; d.languages = { aa: { name: "Aa", ids: [6] }, bb: { name: "Bb", ids: [7] }, cc: { name: "Cc", ids: [8, 9] } };
  d.term = { beams: 4, maxNewTokens: 6, lengthPenalty: 1, repetitionThreshold: 20 };
  return { dir, manifest: m };
}

// ---- fake ONNX Runtime ------------------------------------------------------------------------------------------------------------------

class Tensor { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }

/** `logits`: "f32" | "u16" | "array"; `delay`: ms each decoder run takes; `failDecoder`: creating the decoder session throws. */
function fakeOrt({ logits = "f32", delay = 0, failDecoder = false } = {}) {
  const log = { created: [], released: [], active: 0, maxActive: 0, runs: 0, releasedWhileActive: false };
  const row = new Float32Array(V); row[6] = 3; row[7] = 1; row[10] = 2; row[8] = 0.5; row[9] = 0.5;
  const session = (kind) => ({
    kind,
    inputNames: kind === "decoder" ? ["inputs_embeds", "attention_mask", ...Array.from({ length: LAYERS }, (_, l) => ["key", "value"].map((x) => `past_key_values.${l}.${x}`)).flat()] : ["mel"],
    outputNames: kind === "decoder" ? ["logits", ...Array.from({ length: LAYERS }, (_, l) => ["key", "value"].map((x) => `present.${l}.${x}`)).flat()] : ["audio_features"],
    inputMetadata: kind === "decoder"
      ? [{ name: "inputs_embeds", type: "float16", shape: [1, "S", H] }, { name: "attention_mask", type: "int64", shape: [1, "T"] },
        ...Array.from({ length: LAYERS }, (_, l) => ["key", "value"].map((x) => ({ name: `past_key_values.${l}.${x}`, type: "float16", shape: [1, KVH, "past", HD] }))).flat()]
      : [{ name: "mel", type: "float32", shape: [128, 800] }],
    async run(feed, fetches) {
      if (kind === "tower") return { audio_features: new Tensor("float32", new Float32Array(FRAMES * H).fill(0.1), [FRAMES, H]) };
      log.runs++; log.active++; log.maxActive = Math.max(log.maxActive, log.active);
      try {
        if (delay) await new Promise((r) => setTimeout(r, delay));
        const S = feed.inputs_embeds.dims[1], past = feed.past_key_values?.length, out = {};
        const full = new Float32Array(S * V); full.set(row, (S - 1) * V);
        if (logits === "f32") out.logits = new Tensor("float32", full, [1, S, V]);
        else if (logits === "u16") out.logits = new Tensor("float16", Uint16Array.from(full, f32ToF16Bits), [1, S, V]);
        else out.logits = new Tensor("float16", Array.from(full), [1, S, V]);      // what a Float16Array looks like to the reader: numbers
        const T = feed.attention_mask.dims[1];
        for (let l = 0; l < LAYERS; l++) for (const x of ["key", "value"]) out[`present.${l}.${x}`] = new Tensor("float16", new Uint16Array(KVH * T * HD), [1, KVH, T, HD]);
        void past;
        return Object.fromEntries(fetches.map((n) => [n, out[n]]));
      } finally { log.active--; }
    },
    async release() { if (kind === "decoder" && log.active > 0) log.releasedWhileActive = true; log.released.push(kind); },
  });
  return {
    log, Tensor,
    InferenceSession: {
      async create(src) {
        const kind = readFileSync(src, "utf8");
        log.created.push(kind);
        if (kind === "decoder" && failDecoder) throw new Error("boom: out of memory");
        return session(kind === "decoder" ? "decoder" : "tower");
      },
    },
  };
}

const sine = (n) => Float32Array.from({ length: n }, (_, i) => 0.3 * Math.sin((2 * Math.PI * 440 * i) / 16000));
const A = () => ({ samples: sine(128000), sampleRate: 16000 });
const load = (p, ort, extra = {}) => { const status = []; return loadMedia({ manifest: p.manifest, localDirs: { [p.manifest.repo]: p.dir }, modules: { ort }, device: "cpu", onStatus: (s) => status.push(s), ...extra }).then((model) => ({ model, status })); };

// ---- the three modes --------------------------------------------------------------------------------------------------------------------

test("decoder default ('auto'): loads the decoder and answers language and term through the real ortDecoder wrapper", async () => {
  const ort = fakeOrt(), p = pkg();
  const { model } = await load(p, ort);
  assert.deepEqual(model.decoder, { loaded: true });
  assert.deepEqual(model.tasks.sort(), ["language", "term", "turn_end"]);
  const d = await model.decide({ audio: A(), questions: ["turn_end", "language", { task: "term", term: "a" }] });
  assert.equal(d.answers[1].choice, "aa");
  assert.equal(d.answers[2].hypotheses.length, 5);
  assert.ok(ort.log.runs > 10);
  await model.dispose();
  assert.ok(ort.log.released.includes("decoder") && ort.log.released.includes("tower"));
});

test("decoder 'auto': a decoder that fails to load leaves the other questions working; language and term say why", async () => {
  const ort = fakeOrt({ failDecoder: true }), p = pkg();
  const { model, status } = await load(p, ort);
  assert.equal(model.decoder.loaded, false); assert.match(model.decoder.reason, /boom: out of memory/);
  assert.ok(status.some((s) => /decoder not loaded.*boom/.test(s)), status.join(" | "));
  assert.deepEqual(model.tasks, ["turn_end"]);
  const ok = await model.decide({ audio: A(), questions: ["turn_end"] });
  assert.equal(ok.answers[0].available, true);
  await assert.rejects(() => model.decide({ audio: A(), questions: ["language"] }), /needs the decoder, which failed to load: boom: out of memory/);
  await assert.rejects(() => model.decide({ audio: A(), questions: [{ task: "term", term: "a" }] }), /failed to load/);
  await model.dispose();
});

test("decoder 'auto': a manifest whose decoder has no build for this device also goes on without it", async () => {
  const p = pkg({ cpuBuild: false });
  const { model, status } = await load(p, fakeOrt());
  assert.match(model.decoder.reason, /lists devices webgpu, not cpu/);
  assert.ok(status.some((s) => /decoder not loaded/.test(s)));
  assert.equal((await model.decide({ audio: A(), questions: ["turn_end"] })).answers[0].available, true);
  await model.dispose();
});

test("decoder true: a decoder that fails to load fails the load and releases what was opened", async () => {
  const ort = fakeOrt({ failDecoder: true }), p = pkg();
  await assert.rejects(() => load(p, ort, { decoder: true }), /boom: out of memory/);
  assert.ok(ort.log.released.includes("tower"));
  const { model } = await load(pkg(), fakeOrt(), { decoder: true });
  assert.equal(model.decoder.loaded, true);
  await model.dispose();
});

test("decoder false: nothing of the decoder is read or created", async () => {
  const ort = fakeOrt(), p = pkg();
  const { model } = await load(p, ort, { decoder: false });
  assert.deepEqual(model.decoder, { loaded: false, reason: "off" });
  assert.deepEqual(ort.log.created, ["tower"]);
  await assert.rejects(() => model.decide({ audio: A(), questions: ["language"] }), /needs the decoder, which was not loaded \(loadMedia option decoder: false\)/);
  await model.dispose();
});

// ---- logits of every shape the runtimes hand out ---------------------------------------------------------------------------------------------

test("logits as float32, float16 bits (Uint16Array) and a Float16Array-like array give the same language scores", async () => {
  const got = [];
  for (const logits of ["f32", "u16", "array"]) {
    const { model } = await load(pkg(), fakeOrt({ logits }));
    const d = await model.decide({ audio: A(), questions: [{ task: "language" }] });
    got.push(d.answers[0].probabilities.map((x) => x.probability));
    await model.dispose();
  }
  for (const g of got.slice(1)) for (let i = 0; i < g.length; i++) assert.ok(Math.abs(g[i] - got[0][i]) < 1e-6);
});

test("ortDecoder: cache inputs are checked against layers and kvHeads of the manifest; wrong graphs are refused", () => {
  const ort = fakeOrt(), s = ort.InferenceSession;
  const mk = async () => { const f = path.join(pkg().dir, "decoder/model.onnx"); return s.create(f); };
  return mk().then((session) => {
    const o = { hidden: H, vocab: V, kvHeadDim: HD };
    assert.doesNotThrow(() => ortDecoder(session, ort, { ...o, layers: LAYERS, kvHeads: KVH }));
    assert.throws(() => ortDecoder(session, ort, { ...o, layers: LAYERS + 1 }), /cache inputs.*layers/);
    assert.throws(() => ortDecoder(session, ort, { ...o, kvHeads: KVH + 1 }), /heads/);
    assert.throws(() => ortDecoder({ ...session, inputNames: ["x"] }, ort, o), /inputs_embeds/);
    assert.throws(() => ortDecoder({ ...session, outputNames: ["x"] }, ort, o), /logits/);
  });
});

// ---- dispose() while a question runs ------------------------------------------------------------------------------------------------------

test("dispose() during a running term question waits for it: the decoder session is released after its last run", async () => {
  const ort = fakeOrt({ delay: 3 }), p = pkg();
  const { model } = await load(p, ort);
  const q = model.decide({ audio: A(), questions: [{ task: "term", term: "a" }] });
  for (let i = 0; i < 2000 && ort.log.runs < 3; i++) await new Promise((r) => setTimeout(r, 2));
  assert.ok(ort.log.runs >= 3, "the question is running");
  const settled = q.then(() => "answered", (e) => `failed: ${e.message}`);
  await model.dispose();
  assert.equal(ort.log.releasedWhileActive, false, "released while a run was active");
  assert.equal(await settled, "answered");
  await assert.rejects(() => model.decide({ audio: A(), questions: [{ task: "term", term: "a" }] }), /disposed/);
});

test("dispose() twice, and a decoder question after dispose, are refused cleanly", async () => {
  const ort = fakeOrt(), { model } = await load(pkg(), ort);
  const a = model.dispose(), b = model.dispose();
  await Promise.all([a, b]);
  assert.equal(ort.log.released.filter((k) => k === "decoder").length, 1);
  await assert.rejects(() => model.decide({ audio: A(), questions: ["language"] }), /disposed/);
});

// ---- request validation ----------------------------------------------------------------------------------------------------------------

test("language options: codes that are not the package's (including names inherited from Object) are refused before the tower runs", async () => {
  const { model } = await load(pkg(), fakeOrt());
  for (const options of [["constructor", "toString"], ["aa", "zz"], ["aa", "__proto__"]])
    await assert.rejects(() => model.decide({ audio: A(), questions: [{ task: "language", options }] }), /not codes of this package/);
  await assert.rejects(() => model.decide({ audio: A(), questions: [{ task: "language", options: ["aa", 3] }] }), /not codes of this package/);
  const ok = await model.decide({ audio: A(), questions: [{ task: "language", options: ["aa", "cc"] }] });
  assert.deepEqual(ok.answers[0].probabilities.map((x) => x.value), ["aa", "cc"]);
  await model.dispose();
});
