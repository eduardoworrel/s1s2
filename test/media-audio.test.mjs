// Audio: log-mel, pooling, readers, AudioModel. Torch references: the lab's refs.py -> ref.json (torch Marimba 0.1.2, float32 CPU),
// 56 FLEURS pt-BR clips; a window of two of them is stored in test/fixtures/media/audio (see README there).
//
// TOLERANCES, fixed before these tests ran (spec/MULTIMODAL.md section 6):
//   log-mel vs torch feature extractor ........ max abs <= 1e-4        (lab: 3.2e-5)
//   every probability vs torch Marimba ........ abs <= 1e-4            (lab, Chrome WebGPU: <= 3.8e-5)
//   decisions at their thresholds ............. identical
//   float16 rounding fallback vs every half and midpoint (and Math.f16round where it exists) exactly equal
// Opt-in tests (need the weights; skipped without them): S1S2_MM_WEIGHTS=<dir holding marimba-audio/> (default <repo>/.weights).
// S1S2_AUDIO_REF=<reference run directory> + S1S2_AUDIO_FULL=1 runs all 56 windows through the real tower on Node CPU (several GB of memory).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { createLogMel, melFilters, f16round, f16roundSoft, frameCenter, pooledFeature, createPooledReader, rightAlign, frameMask, AudioModel, validateMediaManifest } from "../dist/media.js";
import { parseSafetensors, Files, f16ToF32 } from "../dist/index.js";
import { safetensors, fixtureF32, audioMeta, audioWeightsDir, ROOT } from "./media-helpers.mjs";

const MANIFEST = JSON.parse(readFileSync(path.join(ROOT, "manifests/marimba-audio.json"), "utf8"));
const SPEC = validateMediaManifest(MANIFEST).audio;
const TOL_MEL = 1e-4, TOL_P = 1e-4;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

test("float16 rounding: the JavaScript fallback is exact on every half, every midpoint and just beside it (no Math.f16round needed)", () => {
  const half = f16ToF32(Uint16Array.from({ length: 0x7c00 }, (_, i) => i));          // every finite non-negative half, decoded by the safetensors/embedding reader
  for (let i = 0; i < half.length; i++) {
    const a = half[i];
    assert.equal(f16roundSoft(a), a); assert.equal(f16roundSoft(-a), -a);
    const b = i + 1 < half.length ? half[i + 1] : 65536, mid = (a + b) / 2;         // a and b are adjacent halves; mid is exact in float64
    const tie = i % 2 === 0 ? a : b;                                                   // ties go to the even mantissa (the even bit pattern)
    assert.equal(f16roundSoft(mid), i + 1 < half.length ? tie : Infinity, `tie ${mid}`);
    const eps = (b - a) * 2 ** -20;
    if (i + 1 < half.length) { assert.equal(f16roundSoft(mid - eps), a); assert.equal(f16roundSoft(mid + eps), b); }
  }
  assert.equal(f16roundSoft(65519.9), 65504); assert.equal(f16roundSoft(65520), Infinity); assert.equal(f16roundSoft(-65520), -Infinity);
  assert.equal(f16roundSoft(1e-9), 0); assert.ok(Object.is(f16roundSoft(-1e-9), -0));
  assert.ok(Number.isNaN(f16roundSoft(NaN)));
  assert.equal(f16round(2049), 2048); assert.equal(f16round(2051), 2052);            // ties to even, with whichever implementation this Node uses
});

test("float16 rounding: the fallback equals Math.f16round on edge values and 20000 random values", { skip: typeof Math.f16round !== "function" && "Math.f16round needs Node 24+; the exhaustive test above checks the fallback without it" }, () => {
  const edge = [0, -0, 1, -1, 0.1, 65504, 65519.9, 65520, 1e-8, 6e-8, 5.96e-8, 2.98e-8, 2.99e-8, 6.1e-5, 6.104e-5, 0.0009765625, 1.0009765625, 1.00048828125, 2049, 2050, 2051, NaN, Infinity];
  const rnd = (() => { let a = 12345; return () => { a = (a * 1664525 + 1013904223) >>> 0; return a / 4294967296; }; })();
  const xs = [...edge];
  for (let i = 0; i < 20000; i++) xs.push((rnd() - 0.5) * 2 * 10 ** (Math.floor(rnd() * 14) - 8));
  for (const x of xs) assert.ok(Object.is(f16roundSoft(x), Math.f16round(x)) || (Number.isNaN(x) && Number.isNaN(f16roundSoft(x))), `${x}: ${f16roundSoft(x)} vs ${Math.f16round(x)}`);
  assert.equal(f16round(2049), 2048); assert.equal(f16round(2051), 2052);   // ties to even
});

test("mel filters: Slaney bank has the shape, area norm and edges of the Whisper one", () => {
  const fb = melFilters(16000, 400, 128), bins = 201;
  assert.equal(fb.length, bins * 128);
  // Slaney norm: each filter integrates to about 2/(hi-lo) * triangle area = 1 over frequency (check one mid filter numerically)
  let s = 0; for (let k = 0; k < bins; k++) s += fb[k * 128 + 60] * (8000 / 200);
  assert.ok(Math.abs(s - 1) < 0.02, `filter 60 integrates to ${s}`);
  assert.ok(fb.every((v) => v >= 0 && Number.isFinite(v)));
});

test("log-mel: a sine and silence behave (shape, clamp, range)", () => {
  const mel = createLogMel(SPEC.preprocess);
  const x = new Float32Array(128000).map((_, i) => 0.5 * Math.sin((2 * Math.PI * 440 * i) / 16000));
  const m = mel(x);
  assert.equal(m.length, 128 * 800);
  const hi = Math.max(...m), lo = Math.min(...m);
  assert.ok(hi - lo <= 8 / 4 + 1e-6, `dynamic range ${(hi - lo) * 4} log10 units must be <= 8`);
  assert.ok(m.every(Number.isFinite));
  const z = mel(new Float32Array(128000)); // silence: all at the floor, then clamped to max - 8 -> constant
  assert.ok(z.every((v) => v === z[0]));
  assert.throws(() => mel(new Float32Array(100)), /shorter than the reflect padding/);
});

test("window alignment and the frame mask", () => {
  const a = rightAlign(Float32Array.of(1, 2, 3), 8);
  assert.deepEqual([...a.window], [0, 0, 0, 0, 0, 1, 2, 3]); assert.equal(a.nValid, 3);
  const b = rightAlign(Float32Array.from({ length: 10 }, (_, i) => i), 8);
  assert.deepEqual([...b.window], [2, 3, 4, 5, 6, 7, 8, 9]); assert.equal(b.nValid, 8);
  // 8 blocks of 1000 samples; 2500 real samples: blocks 6, 7 are fully real, block 5 (samples 5000..5999) ends at 5999 >= 8000-2500=5500: real
  assert.deepEqual([...frameMask(2500, 8000, 8)], [0, 0, 0, 0, 0, 1, 1, 1]);
  assert.deepEqual([...frameMask(8000, 8000, 8)], [1, 1, 1, 1, 1, 1, 1, 1]);
});

// ---- pooled readers on tiny synthetic tensors, against independent arithmetic ----
const SR = 16000, WIN = 128000, N = 8, D = 4;             // 8 frames of 1 s, dim 4
const geo = { win: WIN, sampleRate: SR, nValid: WIN };
const tapOf = (f) => { const h = new Float32Array(N * D); for (let i = 0; i < N; i++) for (let d = 0; d < D; d++) h[i * D + d] = f(i, d); return h; };
const h = tapOf((i, d) => (i + 1) * 0.5 + d * 0.25 - (i % 3) * 0.125);          // exactly representable in float16
const spec1 = (extra) => ({ tap: "t", ...extra });
const outputs = () => ({ frames: N, dim: D });
const row = (i) => Array.from({ length: D }, (_, d) => h[i * D + d]);
const mean = (rows) => rows[0].map((_, d) => rows.reduce((s, r) => s + r[d], 0) / rows.length);

test("frame centres and the tail-mean feature (valid audio only)", () => {
  assert.equal(frameCenter(0, 8, WIN), 8000); assert.equal(frameCenter(7, 8, WIN), 120000);
  const tail = (nValid, seconds) => pooledFeature({ kind: "tail-mean", seconds }, h, { frames: N, dim: D, ...geo, nValid });
  assert.deepEqual([...tail(WIN, 2)], mean([row(6), row(7)]));
  assert.deepEqual([...tail(3 * SR, 2)], mean([row(6), row(7)]));
  assert.deepEqual([...tail(1 * SR, 2)], row(7));                       // only the last frame holds real audio
  assert.equal(typeof tail(1, 2), "string");                            // 1 real sample: its frame is 7 (centre 120000 >= 127999 is false)
});

test("pair, pair-split and mean-std features need a full window and equal the arithmetic", () => {
  const g = { frames: N, dim: D, ...geo };
  const l2 = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const pair = (a, b) => { const na = l2(a), nb = l2(b); return [...a.map((x, d) => (x / na) * (b[d] / nb)), ...a.map((x, d) => Math.abs(x / na - b[d] / nb))]; };
  const a = mean([row(0), row(1), row(2), row(3)]), b = mean([row(4), row(5), row(6), row(7)]);
  const p = pooledFeature({ kind: "pair", a: { from: 0, to: 4 }, b: { from: 4, to: 8 } }, h, g);
  pair(a, b).forEach((x, i) => assert.ok(Math.abs(p[i] - x) < 1e-12));
  const pre = mean([0, 1, 2, 3, 4, 5].map(row)), last = mean([row(6), row(7)]);
  const ps = pooledFeature({ kind: "pair-split", split: 6 }, h, g);
  assert.equal(ps.length, 4 * D);
  [...pair(pre, last), ...pre, ...last].forEach((x, i) => assert.ok(Math.abs(ps[i] - x) < 1e-12));
  const ms = pooledFeature({ kind: "mean-std" }, h, g), m = mean([...Array(N).keys()].map(row));
  m.forEach((x, d) => { assert.ok(Math.abs(ms[d] - x) < 1e-12); const sd = Math.sqrt([...Array(N).keys()].reduce((s, i) => s + (row(i)[d] - x) ** 2, 0) / N); assert.ok(Math.abs(ms[D + d] - sd) < 1e-12); });
  for (const k of [{ kind: "pair", a: { from: 0, to: 4 }, b: { from: 4, to: 8 } }, { kind: "pair-split", split: 6 }, { kind: "mean-std" }])
    assert.match(pooledFeature(k, h, { frames: N, dim: D, ...geo, nValid: 100000 }), /full window/);
});

const T = (extra) => ({ frames: N, dim: D, ...extra });
test("pooled-linear predicate: float64 standardise, linear, Platt; and the uncalibrated score", () => {
  const coef = [0.5, -1, 2, 0.25], mu = [1.5, 2.25, 0.5, 1], sc = [2, 4, 0.5, 1.25];
  const st = parseSafetensors(safetensors({ coef: { dtype: "F32", shape: [D], data: coef }, intercept: { dtype: "F32", shape: [1], data: [0.125] },
    mean: { dtype: "F64", shape: [D], data: mu }, scale: { dtype: "F64", shape: [D], data: sc }, pc: { dtype: "F32", shape: [1], data: [1.5] }, pi: { dtype: "F32", shape: [1], data: [-0.25] } }));
  assert.equal(st.tensors.mean.f64[1], 2.25);
  const r = createPooledReader("turn", { kind: "pooled-linear", type: "predicate", tap: "t", pooling: { kind: "tail-mean", seconds: 2 }, weights: {}, tensors: { coef: "coef", intercept: "intercept", mean: "mean", scale: "scale" }, calibration: { kind: "platt", coefficient: "pc", intercept: "pi" } }, st, () => outputs());
  const a = r.read({ t: h }, geo);
  const x = mean([row(6), row(7)]);
  const logit = 0.125 + x.reduce((s, v, d) => s + ((v - mu[d]) / sc[d]) * coef[d], 0), raw = logit * 1.5 - 0.25;
  assert.ok(Math.abs(a.raw - raw) < 1e-6 && Math.abs(a.probability - sigmoid(raw)) < 1e-7 && Math.abs(a.uncalibrated - sigmoid(logit)) < 1e-7);
  assert.equal(a.type, "predicate"); assert.equal(a.available, true); assert.equal(a.modality, "audio");
  assert.ok(Math.abs(a.margin - Math.abs(2 * a.probability - 1)) < 1e-12);
  assert.equal(a.decision, a.probability >= 0.5 ? "yes" : "no");
});

test("pooled-linear choice with a temperature; pooled-mlp; shape errors; unavailable windows", () => {
  const B = [0.1, -0.2, 0.3], mu = Array(2 * D).fill(1), sc = Array(2 * D).fill(2);
  const coef = Array.from({ length: 3 * 2 * D }, (_, i) => ((i * 7) % 11) / 11 - 0.4), Tm = 0.8;
  const st = parseSafetensors(safetensors({ coef: { dtype: "F32", shape: [3, 2 * D], data: coef }, intercept: { dtype: "F32", shape: [3], data: B },
    scaler_mean: { dtype: "F32", shape: [2 * D], data: mu }, scaler_scale: { dtype: "F32", shape: [2 * D], data: sc }, temperature: { dtype: "F32", shape: [1], data: [Tm] } }));
  const mk = (labels) => createPooledReader("count", { kind: "pooled-linear", type: "choice", labels, tap: "t", pooling: { kind: "mean-std" }, weights: {}, tensors: { coef: "coef", intercept: "intercept", mean: "scaler_mean", scale: "scaler_scale" }, calibration: { kind: "temperature", tensor: "temperature" } }, st, () => outputs());
  const a = mk(["0", "1", "2+"]).read({ t: h }, geo);
  const x = pooledFeature({ kind: "mean-std" }, h, T(geo));
  const z = B.map((b, c) => b + x.reduce((s, v, d) => s + ((v - 1) / 2) * coef[c * 2 * D + d], 0) / 1);
  const ex = z.map((v) => Math.exp((v - Math.max(...z)) / Tm)), se = ex.reduce((p, q) => p + q, 0);
  a.probabilities.forEach((p, i) => assert.ok(Math.abs(p.probability - ex[i] / se) < 1e-6));
  const sorted = [...a.probabilities.map((p) => p.probability)].sort((p, q) => q - p);
  assert.ok(Math.abs(a.margin - (sorted[0] - sorted[1])) < 1e-12);
  assert.equal(a.choice, a.probabilities.reduce((b, p) => (p.probability > b.probability ? p : b)).value);
  assert.throws(() => mk(["0", "1"]), /rows need/);
  const nanTap = Float32Array.from(h); nanTap[3] = NaN;
  assert.throws(() => mk(["0", "1", "2+"]).read({ t: nanTap }, geo), /not finite/);
  assert.equal(mk(["0", "1", "2+"]).read({ t: h }, { ...geo, nValid: 5 * SR }).available, false);

  // MLP: in = 2*D (pair), hidden 3
  const W0 = Array.from({ length: 2 * D * 3 }, (_, i) => ((i * 5) % 7) / 7 - 0.5), B0 = [0.05, -0.1, 0.2], W1 = [1, -0.5, 0.75];
  const sm = parseSafetensors(safetensors({ w0: { dtype: "F32", shape: [2 * D, 3], data: W0 }, b0: { dtype: "F32", shape: [3], data: B0 }, w1: { dtype: "F32", shape: [3, 1], data: W1 }, b1: { dtype: "F32", shape: [1], data: [0.3] },
    pc: { dtype: "F32", shape: [1], data: [0.9] }, pi: { dtype: "F32", shape: [1], data: [0.1] } }));
  const pool = { kind: "pair", a: { from: 0, to: 4 }, b: { from: 4, to: 8 } };
  const mlp = createPooledReader("chg", { kind: "pooled-mlp", type: "predicate", tap: "t", pooling: pool, weights: {}, tensors: { layer0Weight: "w0", layer0Bias: "b0", layer1Weight: "w1", layer1Bias: "b1" }, calibration: { kind: "platt", coefficient: "pc", intercept: "pi" } }, sm, () => outputs());
  const m = mlp.read({ t: h }, geo), f = pooledFeature(pool, h, T(geo));
  const hid = B0.map((b, j) => Math.max(0, b + f.reduce((s, v, d) => s + v * W0[d * 3 + j], 0)));
  const zz = 0.3 + hid.reduce((s, v, j) => s + v * W1[j], 0), p1 = Math.min(Math.max(sigmoid(zz), 1e-12), 1), lg = Math.log(p1) - Math.log(Math.max(1 - p1, 1e-12));
  assert.ok(Math.abs(m.probability - sigmoid(lg * 0.9 + 0.1)) < 1e-7);
  // wrong feature width
  assert.throws(() => createPooledReader("bad", { kind: "pooled-mlp", type: "predicate", tap: "t", pooling: { kind: "pair-split", split: 6 }, weights: {}, tensors: { layer0Weight: "w0", layer0Bias: "b0", layer1Weight: "w1", layer1Bias: "b1" }, calibration: { kind: "none" } }, sm, () => outputs()).read({ t: h }, geo), /takes/);
  assert.throws(() => createPooledReader("x", { kind: "pooled-linear", type: "predicate", tap: "t", pooling: pool, weights: {}, tensors: { coef: "nope", intercept: "b1" }, calibration: { kind: "none" } }, sm, () => outputs()), /not in the weights file/);
});

// ---- AudioModel with an injected runtime: the flow, the request checks, the cost ----
function fakeModel() {
  const spec = JSON.parse(JSON.stringify(SPEC));
  spec.tower.outputs = { h9: { layer: 9, frames: 8, dim: 4 }, h12: { layer: 12, frames: 8, dim: 4 }, h18: { layer: 18, frames: 8, dim: 4 }, audio_features: { layer: "final", frames: 8, dim: 4 } };
  spec.readers = {
    turn_end: { kind: "pooled-linear", type: "predicate", tap: "h18", pooling: { kind: "tail-mean", seconds: 2 }, weights: {}, tensors: { coef: "coef", intercept: "intercept" }, calibration: { kind: "none" } },
    same_speaker: { kind: "pooled-linear", type: "predicate", tap: "h12", pooling: { kind: "pair", a: { from: 0, to: 4 }, b: { from: 4, to: 8 } }, weights: {}, tensors: { coef: "coef8", intercept: "intercept" }, calibration: { kind: "none" } },
    voice_activity: { kind: "speech-head", type: "predicate", tap: "h9", graph: {}, question: { dim: 2, prompt: "q" }, inputs: { tap: "h9", mask: "m", question: "q", output: "s" }, frames: 4, temperature: 2, threshold: 0.3 },
  };
  const w = parseSafetensors(safetensors({ coef: { dtype: "F32", shape: [1, 4], data: [1, 0, 0, 0] }, coef8: { dtype: "F32", shape: [1, 8], data: [1, 0, 0, 0, 0, 0, 0, 0] }, intercept: { dtype: "F32", shape: [1], data: [0] } }));
  let passes = 0; const speechCalls = [];
  const rt = { backend: "fake", device: "cpu", async tower(mel) { passes++; assert.equal(mel.length, 128 * 800); return { h9: h, h12: h, h18: h, audio_features: h }; },
    async speech(task, tap, mask) { speechCalls.push([task, [...mask]]); return 0.8; }, async dispose() {} };
  const m = new AudioModel({ spec, weights: { turn_end: w, same_speaker: w }, questions: { voice_activity: Float32Array.of(1, 2) }, runtime: rt });
  return { m, passes: () => passes, speechCalls };
}
const sine = (n, f = 440) => Float32Array.from({ length: n }, (_, i) => 0.3 * Math.sin((2 * Math.PI * f * i) / 16000));

test("AudioModel.decide: one tower pass for all questions, typed answers, cost parts, speech head threshold", async () => {
  const { m, passes, speechCalls } = fakeModel();
  const d = await m.decide({ audio: { samples: sine(64000), sampleRate: 16000 }, questions: ["voice_activity", "turn_end", "same_speaker"] });
  assert.equal(passes(), 1); assert.equal(d.cost.passes, 1);
  assert.deepEqual(d.answers.map((a) => a.task), ["voice_activity", "turn_end", "same_speaker"]);
  const [vad, turn, same] = d.answers;
  assert.ok(Math.abs(vad.probability - sigmoid(0.8 / 2)) < 1e-12 && vad.threshold === 0.3 && vad.decision === "yes" && vad.method === "head");
  assert.deepEqual(speechCalls[0][1], [0, 0, 1, 1]);                                   // 4 s of real audio: the last two of four 2 s blocks... half the window
  assert.equal(turn.available, true); assert.equal(same.available, false); assert.match(same.reason, /full window/);
  assert.ok(d.cost.ms > 0 && d.cost.parts.mel > 0 && d.cost.parts.tower >= 0 && d.cost.parts.readers >= 0 && d.cost.device === "cpu");
  // encode once, read many
  const enc = await m.encode({ samples: sine(128000), sampleRate: 16000 });
  const more = await m.read(enc, ["same_speaker", "turn_end"]);
  assert.equal(passes(), 2); assert.equal(more[0].available, true);
  await m.dispose();
  await assert.rejects(() => m.decide({ audio: { samples: sine(64000), sampleRate: 16000 }, questions: ["turn_end"] }), /disposed/);
});

test("AudioModel.decide: request errors say what is wrong", async () => {
  const { m } = fakeModel();
  const A = { samples: sine(64000), sampleRate: 16000 };
  await assert.rejects(() => m.decide({ audio: { samples: sine(64000), sampleRate: 48000 }, questions: ["turn_end"] }), /48000 Hz.*resample/);
  await assert.rejects(() => m.decide({ audio: A, questions: ["language"] }), /unknown audio task "language".*voice_activity/);
  await assert.rejects(() => m.decide({ audio: A, questions: [] }), /non-empty/);
  await assert.rejects(() => m.decide({ questions: ["turn_end"] }), /audio.samples/);
  await assert.rejects(() => m.decide({ audio: { samples: new Float32Array(100), sampleRate: 16000 }, questions: ["turn_end"] }), /too short/);
  await assert.rejects(() => m.decide({ audio: { samples: Array.from(sine(64000)), sampleRate: 16000 }, questions: ["turn_end"] }), /Float32Array/);
  for (const bad of [NaN, Infinity, -Infinity]) {
    const x = sine(64000); x[3000] = bad;
    await assert.rejects(() => m.decide({ audio: { samples: x, sampleRate: 16000 }, questions: ["turn_end"] }), /samples\[3000\] is .*finite/);
  }
  const long = sine(300000); long[10] = NaN;                                               // outside the last 8 s: not read, not an error
  assert.equal((await m.decide({ audio: { samples: long, sampleRate: 16000 }, questions: ["turn_end"] })).answers[0].available, true);
});

// ---- torch references on stored windows ----
const meta = audioMeta();
const fx = (key) => meta?.windows.find((w) => w.key === key);

test("log-mel in JavaScript vs the torch feature extractor on stored FLEURS windows", { skip: !meta && "fixtures not generated (recipes/marimba-audio/node-parity.mjs --write-fixtures)" }, () => {
  const mel = createLogMel(SPEC.preprocess);
  for (const k of meta.windows.map((w) => w.key)) {
    const got = mel(fixtureF32(`${k}.win.f32`)), ref = fixtureF32(`${k}.mel_torch.f32`);
    let mx = 0; for (let i = 0; i < ref.length; i++) mx = Math.max(mx, Math.abs(got[i] - ref[i]));
    assert.ok(mx <= TOL_MEL, `${k}: max abs ${mx}`);
  }
});

// readers + real heads on the stored tower outputs
const wdir = audioWeightsDir();
// Without the weights these two parity tests (readers and speech head against the torch probabilities) do not run: the weights
// are not published. To run them: put the package files in <repo>/.weights/marimba-audio/ (layout: recipes/marimba-audio/README.md)
// or point S1S2_MM_WEIGHTS at the directory that holds marimba-audio/, then `node --test test/media-audio.test.mjs`.
const skipW = (!meta && "fixtures not generated (recipes/marimba-audio/node-parity.mjs --write-fixtures)")
  || (!wdir && "SKIPPED, not passed: the Marimba weights are missing (not published). Put marimba-audio/ in .weights/ or set S1S2_MM_WEIGHTS=<dir holding marimba-audio/>; layout in recipes/marimba-audio/README.md");
test("pooled readers with the real Marimba readouts reproduce the torch probabilities (stored tower outputs)", { skip: skipW }, async () => {
  const files = new Files({ platform: "node", localDirs: { [MANIFEST.repo]: wdir } });
  const run = {};
  for (const [task, r] of Object.entries(SPEC.readers)) {
    if (r.kind === "speech-head") continue;
    const bytes = await files.bytes(MANIFEST.repo, MANIFEST.revision, r.weights);   // sha256 checked against the manifest
    run[task] = createPooledReader(task, r, parseSafetensors(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)), (t) => SPEC.tower.outputs[t]);
  }
  const taps = (k) => ({ h12: fixtureF32(`${k}.h12.f32`), h18: fixtureF32(`${k}.h18.f32`) });
  let checked = 0;
  for (const w of meta.windows) {
    const t = taps(w.key), g = { win: WIN, sampleRate: SR, nValid: w.nValid }, ref = w.ref;
    const turn = run.turn_end.read(t, g);
    assert.ok(Math.abs(turn.probability - ref.turn_p) <= TOL_P, `${w.key} turn_p ${turn.probability} vs ${ref.turn_p}`);
    assert.ok(Math.abs(turn.uncalibrated - ref.turn_score) <= TOL_P, `${w.key} turn_score`);
    assert.equal(turn.decision, ref.turn_p >= 0.5 ? "yes" : "no");
    checked += 2;
    for (const task of ["same_speaker", "speaker_change", "speaker_count"]) {
      const a = run[task].read(t, g);
      if (w.set === "A") { assert.equal(a.available, false, `${task} needs a full window`); continue; }
      if (task === "same_speaker") { assert.ok(Math.abs(a.probability - ref.same_p) <= TOL_P, `same ${a.probability} vs ${ref.same_p}`); assert.equal(a.decision, ref.same_p >= 0.5 ? "yes" : "no"); }
      if (task === "speaker_change") { assert.ok(Math.abs(a.probability - ref.change_p) <= TOL_P, `change ${a.probability} vs ${ref.change_p}`); assert.equal(a.decision, ref.change_p >= 0.5 ? "yes" : "no"); }
      if (task === "speaker_count") a.probabilities.forEach((p, i) => assert.ok(Math.abs(p.probability - ref.count_p[i]) <= TOL_P, `count[${i}] ${p.probability} vs ${ref.count_p[i]}`));
      checked++;
    }
  }
  assert.ok(checked >= 6);
});

test("voice activity: the speech-head graph on the stored h9 reproduces the torch probability (onnxruntime-node)", { skip: skipW }, async () => {
  const ort = await import("onnxruntime-node");
  const r = SPEC.readers.voice_activity, files = new Files({ platform: "node", localDirs: { [MANIFEST.repo]: wdir } });
  const graph = await files.bytes(MANIFEST.repo, MANIFEST.revision, r.graph), qb = await files.bytes(MANIFEST.repo, MANIFEST.revision, r.question);
  const q = new Float32Array(qb.buffer.slice(qb.byteOffset, qb.byteOffset + qb.byteLength)), sess = await ort.InferenceSession.create(graph);
  assert.equal(q.length, r.question.dim);
  for (const w of meta.windows) {
    const out = await sess.run({ h9: new ort.Tensor("float32", fixtureF32(`${w.key}.h9.f32`), [104, 896]), fmask: new ort.Tensor("float32", frameMask(w.nValid, WIN, r.frames), [1, r.frames]), qpool: new ort.Tensor("float32", q, [1, r.dim ?? 256]) });
    const p = sigmoid(Number(out.score.data[0]) / r.temperature);
    assert.ok(Math.abs(p - w.ref.vad_p) <= TOL_P, `${w.key}: ${p} vs ${w.ref.vad_p}`);
    assert.equal(p >= r.threshold, w.ref.vad_p >= r.threshold);
  }
});

test("all 56 windows through the real tower on Node CPU (opt-in: S1S2_AUDIO_FULL=1, heavy; see recipes/marimba-audio/node-parity.mjs)", { skip: process.env.S1S2_AUDIO_FULL !== "1" && "SKIPPED, opt-in: needs the weights and the torch reference run; set S1S2_AUDIO_FULL=1 and S1S2_AUDIO_REF=<reference run directory> (see recipes/marimba-audio/node-parity.mjs)" }, async () => {
  const { execFileSync } = await import("node:child_process");
  const out = path.join(ROOT, ".capturas-multimodal-node-parity.json");
  execFileSync("node", [path.join(ROOT, "recipes/marimba-audio/node-parity.mjs"), process.env.S1S2_AUDIO_REF, "--out", out], { stdio: "inherit" });
  const r = JSON.parse(readFileSync(out, "utf8"));
  for (const [k, v] of Object.entries(r.maxAbsDiff)) assert.ok(v <= (k.startsWith("log-mel") ? TOL_MEL : TOL_P), `${k}: ${v}`);
  for (const v of Object.values(r.agreement)) assert.equal(v, 56);
});

test("the media code is its own entry point: the text API (root, bundle) does not carry it", async () => {
  const root = await import("../dist/index.js"), media = await import("../dist/media.js");
  for (const n of ["loadMedia", "AudioModel", "createLogMel", "createPooledReader", "f16round", "rightAlign", "frameMask", "pinnedProblems", "mediaProblems", "validateMediaManifest"])
    assert.equal(n in root, false, `${n} leaked into the root API`);
  for (const n of ["loadMedia", "AudioModel", "loadAudio", "validateMediaManifest"]) assert.equal(typeof media[n], "function", n);
  const bundle = readFileSync(path.join(ROOT, "dist/index.js"), "utf8");
  for (const code of ["createLogMel", "createPooledReader", "f16roundSoft"]) assert.equal(bundle.includes(code), false, `${code} is bundled into dist/index.js`);
  const pkg = JSON.parse(readFileSync(path.join(ROOT, "package.json"), "utf8"));
  assert.equal(pkg.exports["./media"].default, "./dist/media.js");
});
