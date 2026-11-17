// Reference reader of the alias3-web head, vendored VERBATIM for fidelity tests (lines 33-70 and the body of
// AliasWeb.norm, lines 81-88) from aliasrobotics/alias3-web at revision 2ab2e6016a91256fd074144a22ed97d7cea72f6d,
// file aliasweb.mjs (git blob 97fca84231465c6f8c78829071c34ecc342b2676). Copyright Alias Robotics, Apache-2.0
// (https://huggingface.co/aliasrobotics/alias3-web). Only `export` keywords were added and `norm` was lifted out of
// the class as `aliasNorm(normW, h)`; no arithmetic was changed.
export function f16(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : s ? -Infinity : Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
export function loadHead(buf) {
  const dv = new DataView(buf);
  const jl = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jl)));
  let off = 4 + jl;
  const take = (n) => { const f = new Float32Array(n); for (let i = 0; i < n; i++) f[i] = f16(dv.getUint16(off + i * 2, true)); off += n * 2; return f; };
  const d = meta.headDim, h = meta.hiddenSize;
  return { q: take(d * h), qb: take(d), k: take(d * h), kb: take(d), headDim: d, hiddenSize: h,
    temperature: meta.temperature ?? 1.0, scale: meta.scale ?? 1 / Math.sqrt(d), scoreOffset: meta.score_offset ?? 0 };
}
// norm.bin: [u32 lenJSON][json {hiddenSize}][fp32 hiddenSize]
export async function loadNorm(buf) {
  const dv = new DataView(buf);
  const jl = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jl)));
  const n = meta.hiddenSize, off = 4 + jl, f = new Float32Array(n);
  for (let i = 0; i < n; i++) f[i] = dv.getFloat32(off + i * 4, true);
  return f;
}
export function proj(W, b, x, dim, hidden) {
  const y = new Float32Array(dim);
  for (let d = 0; d < dim; d++) { let a = b[d]; const r = d * hidden; for (let i = 0; i < hidden; i++) a += W[r + i] * x[i]; y[d] = a; }
  return y;
}
export const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const headLogit = (hd, hD, hO) => dot(proj(hd.q, hd.qb, hD, hd.headDim, hd.hiddenSize), proj(hd.k, hd.kb, hO, hd.headDim, hd.hiddenSize)) * hd.scale / hd.temperature + (hd.scoreOffset || 0);
export const softmax = (xs) => {
  const mx = Math.max(...xs.filter(Number.isFinite));
  const e = xs.map((x) => (Number.isFinite(x) ? Math.exp(x - mx) : 0));
  const z = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((v) => v / z);
};
export function aliasNorm(normW, h) {
  const self = { normW };
    if (!self.normW) return h;
    let s = 0; for (let i = 0; i < h.length; i++) s += h[i] * h[i];
    const inv = 1 / Math.sqrt(s / h.length);
    const o = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) o[i] = h[i] * inv * self.normW[i];
    return o;
  }
