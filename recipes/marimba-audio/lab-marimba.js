// Leituras do Marimba em JavaScript puro: fim de turno (camada 18, cauda de 2 s, Platt), locutor (camada 12) e voz ativa (máscara + chamada da cabeça ONNX).
const SR = 16000, WIN = 128000;
const f16 = (typeof Math.f16round === "function") ? Math.f16round : (() => { const b = new Float32Array(1), u = new Uint32Array(b.buffer); return (x) => { // arredonda float32 -> float16 -> float32
  b[0] = x; const bits = u[0], s = bits & 0x80000000, e = (bits >>> 23) & 0xff; if (e === 0xff) return x; const ex = e - 127;
  if (ex > 15) return s ? -Infinity : Infinity; if (ex < -25) return s ? -0 : 0;
  const q = ex < -14 ? Math.pow(2, -24) : Math.pow(2, ex - 10); return Math.round(x / q) * q; }; })();
const dec = (o) => { const s = atob(o.b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return o.shape.length ? new Float32Array(u.buffer) : null; };
const dec64 = (o) => { const s = atob(o.b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float64Array(u.buffer); };
export function carregarPesos(j) {
  const w = {}; for (const t of ["turn_end", "same_speaker", "speaker_change", "speaker_count"]) { w[t] = {}; for (const [k, v] of Object.entries(j[t])) w[t][k] = (k.endsWith("64")) ? dec64(v) : dec(v); }
  w.vad = { temperature: j.vad.temperature, tail_temperature: j.vad.tail_temperature, threshold: j.vad.threshold, qpool: dec(j.vad.qpool) }; return w;
}
const sigmoid = (x) => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
const frameCenter = (i, n) => Math.min(Math.floor((i + 0.5) * WIN / n), WIN - 1);
// h: Float32Array [n*896] (n=104), saída crua do encoder. nValid = amostras reais do áudio (alinhado à direita).
export function maskQuadros(nValid, nFrames = 100) { const m = new Float32Array(nFrames), blk = WIN / nFrames; for (let f = 0; f < nFrames; f++) m[f] = ((f + 1) * blk - 1 >= WIN - nValid) ? 1 : 0; return m; }
export function turnEnd(h18, n, nValid, W) { // h18 em fp16 (como no torch)
  const D = 896, acc = new Float64Array(D); let c = 0;
  for (let i = 0; i < n; i++) { const ct = frameCenter(i, n); if (ct >= WIN - nValid && ct >= WIN - 2 * SR) { c++; for (let d = 0; d < D; d++) acc[d] += f16(h18[i * D + d]); } }
  if (!c) throw new Error("sem quadros válidos no fim da janela");
  const t = W.turn_end; let logit = t.intercept[0];
  for (let d = 0; d < D; d++) logit += ((acc[d] / c - t.mean64[d]) / t.scale64[d]) * t.coef[d];
  return { score: sigmoid(logit), p: sigmoid(logit * t.platt_coef[0] + t.platt_intercept[0]), logit };
}
function pool(h, n, lo, hi) { const D = 896, o = new Float64Array(D); let c = 0; for (let i = 0; i < n; i++) { const ct = frameCenter(i, n); if (ct >= lo * SR && ct < hi * SR) { c++; for (let d = 0; d < D; d++) o[d] += f16(h[i * D + d]); } } for (let d = 0; d < D; d++) o[d] /= c; return o; }
function pair(a, b) { let na = 0, nb = 0; for (let d = 0; d < a.length; d++) { na += a[d] * a[d]; nb += b[d] * b[d]; } na = Math.sqrt(na); nb = Math.sqrt(nb); const o = new Float64Array(a.length * 2); for (let d = 0; d < a.length; d++) { const x = a[d] / na, y = b[d] / nb; o[d] = x * y; o[a.length + d] = Math.abs(x - y); } return o; }
const lin = (x, w) => { const D = x.length, C = w.intercept.length, o = new Float64Array(C); for (let c = 0; c < C; c++) { let s = w.intercept[c]; for (let d = 0; d < D; d++) s += ((x[d] - w.scaler_mean[d]) / w.scaler_scale[d]) * w.coef[c * D + d]; o[c] = s; } return o; };
// Só com janela cheia de 8 s (como o pacote original). h12 [n*896].
export function locutor(h12, n, W) {
  const a = pool(h12, n, 0, 4), b = pool(h12, n, 4, 8), same = pair(a, b);
  const ps = sigmoid(lin(same, W.same_speaker)[0] * W.same_speaker.platt_coefficient[0] + W.same_speaker.platt_intercept[0]);
  const pre = pool(h12, n, 0, 6), last = pool(h12, n, 6, 8), pc = pair(pre, last), x = new Float64Array(3584); x.set(pc); x.set(pre, 1792); x.set(last, 2688);
  const c = W.speaker_change, H = 256, hid = new Float64Array(H); for (let j = 0; j < H; j++) hid[j] = c.layer_0_bias[j];
  for (let d = 0; d < 3584; d++) { const xv = x[d], row = d * H; for (let j = 0; j < H; j++) hid[j] += xv * c.layer_0_weight[row + j]; }
  let z = c.layer_1_bias[0]; for (let j = 0; j < H; j++) z += Math.max(hid[j], 0) * c.layer_1_weight[j];
  const p1 = Math.min(Math.max(sigmoid(z), 1e-12), 1), lg = Math.log(p1) - Math.log(Math.max(1 - p1, 1e-12));
  const pch = sigmoid(lg * c.platt_coefficient[0] + c.platt_intercept[0]);
  const m = new Float64Array(896), sd = new Float64Array(896), D = 896; for (let d = 0; d < D; d++) { let s = 0; for (let i = 0; i < n; i++) s += f16(h12[i * D + d]); m[d] = s / n; let v = 0; for (let i = 0; i < n; i++) { const e = f16(h12[i * D + d]) - m[d]; v += e * e; } sd[d] = Math.sqrt(v / n); }
  const xc = new Float64Array(1792); xc.set(m); xc.set(sd, 896);
  const lc = lin(xc, W.speaker_count), mx = Math.max(...lc), ex = Array.from(lc, (v) => Math.exp((v - mx) / W.speaker_count.temperature[0])), se = ex.reduce((a, b) => a + b, 0);
  return { same: ps, change: pch, count: ex.map((v) => v / se) };
}
export const vadProb = (score, W, tail = false) => sigmoid(score / (tail ? W.vad.tail_temperature : W.vad.temperature));
