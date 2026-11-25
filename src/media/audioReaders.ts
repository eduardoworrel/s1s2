// Audio readers over pooled tower states (no model runtime here: pure maths on Float32Array taps).
//   pooled-linear: float16-round -> pool -> standardise -> W x + b -> Platt | temperature softmax
//   pooled-mlp:    float16-round -> pool -> Linear -> ReLU -> Linear -> sigmoid -> logit -> Platt
// Ported from the reference run's marimba.js (parity with the torch Marimba package: |dp| <= 3.8e-5 on 56 windows). Answers reuse
// predicateFromP / choiceFromScores, so probability, margin and rawMargin mean what they mean for text.

import { predicateFromP, choiceFromScores, sigmoid, type PredicateAnswer } from "../readout.js";
import type { SafeTensors } from "../heads/safetensors.js";
import type { Calibration, Pooling, PooledLinearReader, PooledMlpReader, MediaAnswer, UnavailableAnswer, Span } from "./types.js";

/** Round to the nearest float16 (ties to even) and back; software version, used when Math.f16round is missing. */
export function f16roundSoft(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const s = x < 0 ? -1 : 1, a = Math.abs(x);
  if (a >= 65520) return s * Infinity;           // halfway between the largest half (65504) and 2^16 rounds up to infinity
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--; else if (2 ** (e + 1) <= a) e++;
  if (e < -14) e = -14;                          // subnormals share the exponent of the smallest normal
  const q = 2 ** (e - 10), v = a / q, f = Math.floor(v), d = v - f;
  const r = d > 0.5 || (d === 0.5 && f % 2 === 1) ? f + 1 : f;
  return s * r * q;
}
/** Round to the nearest float16 and back (the readers were fit on float16-stored features). */
export const f16round: (x: number) => number = typeof (Math as any).f16round === "function" ? (Math as any).f16round : f16roundSoft;

/** Centre sample of frame i of n over a window of `win` samples. */
export const frameCenter = (i: number, n: number, win: number) => Math.min(Math.floor(((i + 0.5) * win) / n), win - 1);

export interface FrameGeometry { frames: number; dim: number; /** window samples */ win: number; sampleRate: number; /** real samples at the end of the window */ nValid: number }

/** Mean of the float16-rounded frames whose centre is in [lo, hi) samples (and, with `validOnly`, in real audio). */
function poolSpan(h: Float32Array, g: FrameGeometry, lo: number, hi: number, validOnly = false): Float64Array | null {
  const { frames: n, dim: D, win } = g, o = new Float64Array(D);
  let c = 0;
  for (let i = 0; i < n; i++) {
    const ct = frameCenter(i, n, win);
    if (ct < lo || ct >= hi || (validOnly && ct < win - g.nValid)) continue;
    c++;
    for (let d = 0; d < D; d++) o[d] += f16round(h[i * D + d]);
  }
  if (!c) return null;
  for (let d = 0; d < D; d++) o[d] /= c;
  return o;
}

function pair(a: Float64Array, b: Float64Array): Float64Array {
  let na = 0, nb = 0;
  for (let d = 0; d < a.length; d++) { na += a[d] * a[d]; nb += b[d] * b[d]; }
  na = Math.sqrt(na); nb = Math.sqrt(nb);
  const o = new Float64Array(a.length * 2);
  for (let d = 0; d < a.length; d++) { const x = a[d] / na, y = b[d] / nb; o[d] = x * y; o[a.length + d] = Math.abs(x - y); }
  return o;
}

const spanSamples = (s: Span, g: FrameGeometry): [number, number] => [Math.round(s.from * g.sampleRate), Math.round(s.to * g.sampleRate)];

/** The feature vector of a pooling over one tap, or the reason the window cannot give it. */
export function pooledFeature(p: Pooling, h: Float32Array, g: FrameGeometry): Float64Array | string {
  if (h.length !== g.frames * g.dim) throw new Error(`pooling: tap has ${h.length} values, expected ${g.frames} x ${g.dim}`);
  const full = g.nValid >= g.win;
  const needFull = "needs a full window of audio (" + g.win / g.sampleRate + " s)";
  if (p.kind === "tail-mean") {
    const m = poolSpan(h, g, g.win - p.seconds * g.sampleRate, g.win, true);
    return m ?? `no frame of real audio in the last ${p.seconds} s`;
  }
  if (!full) return needFull;
  if (p.kind === "pair") {
    const a = poolSpan(h, g, ...spanSamples(p.a, g)), b = poolSpan(h, g, ...spanSamples(p.b, g));
    return a && b ? pair(a, b) : "a span holds no frame";
  }
  if (p.kind === "pair-split") {
    const k = Math.round(p.split * g.sampleRate), pre = poolSpan(h, g, 0, k), last = poolSpan(h, g, k, g.win);
    if (!pre || !last) return "a span holds no frame";
    const pc = pair(pre, last), x = new Float64Array(pc.length + pre.length + last.length);
    x.set(pc); x.set(pre, pc.length); x.set(last, pc.length + pre.length);
    return x;
  }
  // mean-std: population std over every frame
  const { frames: n, dim: D } = g, m = new Float64Array(D), sd = new Float64Array(D);
  for (let d = 0; d < D; d++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += f16round(h[i * D + d]);
    m[d] = s / n;
    let v = 0;
    for (let i = 0; i < n; i++) { const e = f16round(h[i * D + d]) - m[d]; v += e * e; }
    sd[d] = Math.sqrt(v / n);
  }
  const x = new Float64Array(2 * D); x.set(m); x.set(sd, D);
  return x;
}

const tensor = (st: SafeTensors, name: string | undefined, what: string) => {
  if (!name) throw new Error(`audio reader: ${what} tensor name missing in the manifest`);
  const t = st.tensors[name];
  if (!t) throw new Error(`audio reader: tensor ${name} (${what}) not in the weights file (has ${Object.keys(st.tensors).join(", ")})`);
  return t;
};
const exact = (t: { data: Float32Array; f64?: Float64Array }) => t.f64 ?? t.data;

export interface ReaderResult { answer: PredicateAnswer | ReturnType<typeof choiceFromScores>; threshold?: number; uncalibrated?: number }

function platt(st: SafeTensors, c: Extract<Calibration, { kind: "platt" }>, logit: number): number {
  return logit * tensor(st, c.coefficient, "Platt coefficient").data[0] + tensor(st, c.intercept, "Platt intercept").data[0];
}

const finite = (name: string, x: number) => { if (!Number.isFinite(x)) throw new Error(`audio reader ${name}: the score is not finite (the tap holds NaN or Infinity)`); return x; };

export interface PooledReaderRun { read(taps: Record<string, Float32Array>, g: Pick<FrameGeometry, "win" | "sampleRate" | "nValid">): MediaAnswer }

/** Build the reader of a manifest entry from its parsed safetensors. Checks every tensor and shape at build time. */
export function createPooledReader(name: string, r: PooledLinearReader | PooledMlpReader, st: SafeTensors, geometry: (tap: string) => { frames: number; dim: number }): PooledReaderRun {
  const tp = geometry(r.tap);
  const t = r.tensors;
  const unavailable = (reason: string): UnavailableAnswer => ({ name, task: name, modality: "audio", available: false, reason });
  if (r.kind === "pooled-linear") {
    const coef = tensor(st, t.coef, "coef"), b = tensor(st, t.intercept, "intercept");
    const C = b.data.length, D = coef.data.length / C;
    if (!Number.isInteger(D)) throw new Error(`${name}: coef has ${coef.data.length} values for ${C} rows`);
    const mean = t.mean ? tensor(st, t.mean, "mean") : null, scale = t.scale ? tensor(st, t.scale, "scale") : null;
    if (!!mean !== !!scale) throw new Error(`${name}: mean and scale go together`);
    if (mean && (mean.data.length !== D || scale!.data.length !== D)) throw new Error(`${name}: mean/scale length ${mean.data.length} != ${D}`);
    if (r.type === "predicate" && C !== 1) throw new Error(`${name}: a predicate reader has one row, the weights have ${C}`);
    if (r.type === "choice" && (!r.labels || r.labels.length !== C)) throw new Error(`${name}: ${C} rows need ${C} labels`);
    const mu = mean ? exact(mean) : null, sc = scale ? exact(scale) : null, W = coef.data, B = b.data;
    const T = r.calibration.kind === "temperature" ? tensor(st, r.calibration.tensor, "temperature").data[0] : 1;
    return {
      read(taps, g) {
        const f = pooledFeature(r.pooling, taps[r.tap], { ...g, frames: tp.frames, dim: tp.dim });
        if (typeof f === "string") return unavailable(f);
        if (f.length !== D) throw new Error(`${name}: pooled feature has ${f.length} values, the reader takes ${D}`);
        const z = new Float64Array(C);
        for (let c = 0; c < C; c++) {
          let s = B[c];
          for (let d = 0; d < D; d++) s += ((mu ? (f[d] - mu[d]) / sc![d] : f[d]) * W[c * D + d]);
          z[c] = finite(name, s);
        }
        if (r.type === "choice") {
          const a = choiceFromScores(name, r.labels!, Array.from(z, (v) => v / T), "head");
          return { ...a, modality: "audio", task: name, decision: a.choice, available: true };
        }
        const raw = r.calibration.kind === "platt" ? platt(st, r.calibration, z[0]) : z[0];
        const a = predicateFromP(name, sigmoid(raw), "head", raw);
        return { ...a, modality: "audio", task: name, decision: a.probability >= 0.5 ? "yes" : "no", threshold: 0.5, uncalibrated: sigmoid(z[0]), available: true } as MediaAnswer;
      },
    };
  }
  // pooled-mlp
  const W0 = tensor(st, t.layer0Weight, "layer0 weight"), B0 = tensor(st, t.layer0Bias, "layer0 bias");
  const W1 = tensor(st, t.layer1Weight, "layer1 weight"), B1 = tensor(st, t.layer1Bias, "layer1 bias");
  const H = B0.data.length, D = W0.data.length / H;
  if (!Number.isInteger(D) || W1.data.length !== H || B1.data.length !== 1) throw new Error(`${name}: MLP shapes ${W0.shape} / ${W1.shape} do not fit`);
  return {
    read(taps, g) {
      const x = pooledFeature(r.pooling, taps[r.tap], { ...g, frames: tp.frames, dim: tp.dim });
      if (typeof x === "string") return unavailable(x);
      if (x.length !== D) throw new Error(`${name}: pooled feature has ${x.length} values, the MLP takes ${D}`);
      const hid = Float64Array.from(B0.data);
      for (let d = 0; d < D; d++) { const xv = x[d], row = d * H; for (let j = 0; j < H; j++) hid[j] += xv * W0.data[row + j]; }
      let z = B1.data[0];
      for (let j = 0; j < H; j++) z += Math.max(hid[j], 0) * W1.data[j];
      finite(name, z);
      const p1 = Math.min(Math.max(sigmoid(z), 1e-12), 1), lg = Math.log(p1) - Math.log(Math.max(1 - p1, 1e-12));
      const raw = r.calibration.kind === "platt" ? platt(st, r.calibration, lg) : lg;
      const a = predicateFromP(name, sigmoid(raw), "head", raw);
      return { ...a, modality: "audio", task: name, decision: a.probability >= 0.5 ? "yes" : "no", threshold: 0.5, uncalibrated: p1, available: true } as MediaAnswer;
    },
  };
}

/** Right-align audio in a window of `win` samples: shorter audio is zero-padded on the left, longer keeps its last `win`. */
export function rightAlign(samples: Float32Array, win: number): { window: Float32Array; nValid: number } {
  if (samples.length >= win) return { window: samples.subarray(samples.length - win), nValid: win };
  const w = new Float32Array(win); w.set(samples, win - samples.length);
  return { window: w, nValid: samples.length };
}

/** Frame mask (1 = the frame holds real audio) for a head that pools the window into `frames` equal blocks. */
export function frameMask(nValid: number, win: number, frames: number): Float32Array {
  const m = new Float32Array(frames), blk = win / frames;
  for (let f = 0; f < frames; f++) m[f] = (f + 1) * blk - 1 >= win - nValid ? 1 : 0;
  return m;
}
