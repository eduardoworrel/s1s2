// Mel extractor equal to the Qwen3-ASR WhisperFeatureExtractor (128 bands, 16 kHz, n_fft 400, hop 160, periodic Hann window,
// reflect padding of 200 at both ends, power, mel filters, log10 floored at 1e-10, clamp at max-8, (x+4)/4, last frame dropped).
export const N_FFT = 400, HOP = 160, N_MELS = 128;
const b64ToF32 = (b64) => { const s = atob(b64); const u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); };
export function loadFilters(json) { return { data: b64ToF32(json.b64), bins: json.shape[0], mels: json.shape[1] }; } // [201,128]
const hann = new Float64Array(N_FFT); for (let i = 0; i < N_FFT; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N_FFT);
const TW_RE = new Float64Array(N_FFT), TW_IM = new Float64Array(N_FFT);
for (let i = 0; i < N_FFT; i++) { TW_RE[i] = Math.cos(2 * Math.PI * i / N_FFT); TW_IM[i] = -Math.sin(2 * Math.PI * i / N_FFT); }
// Mixed-radix complex FFT (400 = 2^4 * 5^2), recursive, decimation in time.
function fft(re, im, n, stride) { // reads re/im with a stride; returns [re,im] of size n
  if (n === 1) return [Float64Array.of(re[0]), Float64Array.of(im[0])];
  let p = 2; while (n % p) p++;
  const m = n / p, subs = [];
  for (let r = 0; r < p; r++) { const sr = new Float64Array(m), si = new Float64Array(m); for (let j = 0; j < m; j++) { sr[j] = re[(j * p + r) * stride]; si[j] = im[(j * p + r) * stride]; } subs.push(fft(sr, si, m, 1)); }
  const outR = new Float64Array(n), outI = new Float64Array(n), step = N_FFT / n;
  for (let k = 0; k < m; k++) for (let q = 0; q < p; q++) {
    const kk = k + m * q; let ar = 0, ai = 0;
    for (let r = 0; r < p; r++) { const t = ((r * kk) % n) * step, wr = TW_RE[t], wi = TW_IM[t], yr = subs[r][0][k], yi = subs[r][1][k]; ar += yr * wr - yi * wi; ai += yr * wi + yi * wr; }
    outR[kk] = ar; outI[kk] = ai;
  }
  return [outR, outI];
}
// audio: Float32Array (already in the 8 s window = 128000 samples). Returns Float32Array [128*800], row = band.
export function logMel(audio, filters) {
  const L = audio.length, pad = N_FFT / 2, P = new Float64Array(L + 2 * pad);
  for (let i = 0; i < L; i++) P[pad + i] = audio[i];
  for (let i = 1; i <= pad; i++) { P[pad - i] = audio[i]; P[pad + L - 1 + i] = audio[L - 1 - i]; }
  const frames = 1 + Math.floor((P.length - N_FFT) / HOP), T = frames - 1, bins = N_FFT / 2 + 1;
  const fb = filters.data, out = new Float32Array(N_MELS * T), im0 = new Float64Array(N_FFT), re = new Float64Array(N_FFT), pw = new Float64Array(bins);
  let mx = -Infinity;
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < N_FFT; i++) re[i] = P[t * HOP + i] * hann[i];
    const [fr, fi] = fft(re, im0, N_FFT, 1);
    for (let k = 0; k < bins; k++) pw[k] = fr[k] * fr[k] + fi[k] * fi[k];
    for (let b = 0; b < N_MELS; b++) { let s = 0; for (let k = 0; k < bins; k++) s += pw[k] * fb[k * N_MELS + b]; const v = Math.log10(Math.max(s, 1e-10)); out[b * T + t] = v; if (v > mx) mx = v; }
  }
  const floor = mx - 8; for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], floor) + 4) / 4;
  return out;
}
