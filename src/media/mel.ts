// Log-mel extractor equal to the WhisperFeatureExtractor of Qwen3-ASR (ported from the lab's mel.js, which matched the
// torch features to 3.2e-5 max abs on 56 windows): periodic Hann window, reflect padding of nFft/2, power spectrum,
// Slaney mel filters, log10 with a floor, a dynamic-range clamp, (x + shift) / divide, last STFT frame dropped.
// Pure JavaScript (float64 inside, float32 out): runs in a browser tab, a worker and Node, with no model file.

import type { LogMelSpec } from "./types.js";

/** Slaney mel scale (transformers.audio_utils.hertz_to_mel / mel_to_hertz, mel_scale="slaney"). */
const hzToMel = (f: number) => (f >= 1000 ? 15 + Math.log(f / 1000) * (27 / Math.log(6.4)) : (3 * f) / 200);
const melToHz = (m: number) => (m >= 15 ? 1000 * Math.exp((Math.log(6.4) / 27) * (m - 15)) : (200 * m) / 3);

/** Mel filter bank [bins, mels] (row-major), Slaney scale and Slaney area norm, as transformers.audio_utils.mel_filter_bank. */
export function melFilters(sampleRate: number, nFft: number, nMels: number, fMin = 0, fMax = sampleRate / 2): Float32Array {
  const bins = nFft / 2 + 1;
  const mm = hzToMel(fMin), mx = hzToMel(fMax);
  const edges = Array.from({ length: nMels + 2 }, (_, i) => melToHz(mm + ((mx - mm) * i) / (nMels + 1)));
  const out = new Float32Array(bins * nMels);
  for (let k = 0; k < bins; k++) {
    const f = (k * (sampleRate / 2)) / (bins - 1);
    for (let b = 0; b < nMels; b++) {
      const lo = edges[b], mid = edges[b + 1], hi = edges[b + 2];
      const down = -(lo - f) / (mid - lo), up = (hi - f) / (hi - mid);
      out[k * nMels + b] = Math.max(0, Math.min(down, up)) * (2 / (hi - lo));
    }
  }
  return out;
}

function tables(nFft: number) {
  const hann = new Float64Array(nFft), re = new Float64Array(nFft), im = new Float64Array(nFft);
  for (let i = 0; i < nFft; i++) {
    hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft);
    re[i] = Math.cos((2 * Math.PI * i) / nFft); im[i] = -Math.sin((2 * Math.PI * i) / nFft);
  }
  return { hann, twRe: re, twIm: im };
}

/** Mixed-radix recursive FFT (nFft = 2^a 5^b in the Qwen3-ASR extractor: 400). Reads with a stride; returns [re, im]. */
function fft(re: Float64Array, im: Float64Array, n: number, stride: number, nFft: number, twRe: Float64Array, twIm: Float64Array): [Float64Array, Float64Array] {
  if (n === 1) return [Float64Array.of(re[0]), Float64Array.of(im[0])];
  let p = 2; while (n % p) p++;
  const m = n / p, subs: [Float64Array, Float64Array][] = [];
  for (let r = 0; r < p; r++) {
    const sr = new Float64Array(m), si = new Float64Array(m);
    for (let j = 0; j < m; j++) { sr[j] = re[(j * p + r) * stride]; si[j] = im[(j * p + r) * stride]; }
    subs.push(fft(sr, si, m, 1, nFft, twRe, twIm));
  }
  const outR = new Float64Array(n), outI = new Float64Array(n), step = nFft / n;
  for (let k = 0; k < m; k++) for (let q = 0; q < p; q++) {
    const kk = k + m * q; let ar = 0, ai = 0;
    for (let r = 0; r < p; r++) {
      const t = ((r * kk) % n) * step, wr = twRe[t], wi = twIm[t], yr = subs[r][0][k], yi = subs[r][1][k];
      ar += yr * wr - yi * wi; ai += yr * wi + yi * wr;
    }
    outR[kk] = ar; outI[kk] = ai;
  }
  return [outR, outI];
}

export interface MelExtractor { (audio: Float32Array): Float32Array; spec: LogMelSpec; filters: Float32Array }

/** Build the extractor of a manifest's `audio.preprocess`. Returns [nMels * T] row-major (row = band), T = frames in the STFT - 1. */
export function createLogMel(spec: LogMelSpec, filters?: Float32Array): MelExtractor {
  if (spec.window !== "hann-periodic" || spec.padding !== "reflect" || spec.filters !== "slaney")
    throw new Error(`log-mel: unsupported window/padding/filters (${spec.window}/${spec.padding}/${spec.filters})`);
  const { nFft, hop, nMels } = spec, bins = nFft / 2 + 1;
  const fb = filters ?? melFilters(spec.sampleRate, nFft, nMels);
  if (fb.length !== bins * nMels) throw new Error(`log-mel: filters have ${fb.length} values, expected ${bins}x${nMels}`);
  const { hann, twRe, twIm } = tables(nFft);
  const fn = ((audio: Float32Array) => {
    const L = audio.length, pad = nFft / 2;
    if (L <= pad) throw new Error(`log-mel: audio of ${L} samples is shorter than the reflect padding (${pad})`);
    const P = new Float64Array(L + 2 * pad);
    for (let i = 0; i < L; i++) P[pad + i] = audio[i];
    for (let i = 1; i <= pad; i++) { P[pad - i] = audio[i]; P[pad + L - 1 + i] = audio[L - 1 - i]; }
    const frames = 1 + Math.floor((P.length - nFft) / hop), T = frames - 1;
    const out = new Float32Array(nMels * T), im0 = new Float64Array(nFft), re = new Float64Array(nFft), pw = new Float64Array(bins);
    let mx = -Infinity;
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < nFft; i++) re[i] = P[t * hop + i] * hann[i];
      const [fr, fi] = fft(re, im0, nFft, 1, nFft, twRe, twIm);
      for (let k = 0; k < bins; k++) pw[k] = fr[k] * fr[k] + fi[k] * fi[k];
      for (let b = 0; b < nMels; b++) {
        let s = 0;
        for (let k = 0; k < bins; k++) s += pw[k] * fb[k * nMels + b];
        const v = Math.log10(Math.max(s, spec.log.floor));
        out[b * T + t] = v; if (v > mx) mx = v;
      }
    }
    const floor = mx - spec.log.dynamicRange;
    for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], floor) + spec.log.shift) / spec.log.divide;
    return out;
  }) as MelExtractor;
  fn.spec = spec; fn.filters = fb;
  return fn;
}
