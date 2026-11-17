// Out-of-graph final norm (spec 0.2 section 6.5). Some published builds export the hidden state BEFORE the decoder's
// final norm while their head was trained AFTER it (aliasrobotics/alias3-web@2ab2e60: "the graph exports hidden_states
// before the final norm; the head was trained on the normalised state"). The runtime then applies the norm itself.

export interface NormSpecLike {
  /** "rmsnorm" (the only kind defined in 0.2). */
  kind?: "rmsnorm";
  /** Added to mean(h^2) under the square root. alias3-web's reader uses none (0); a Qwen/Llama RMSNorm uses 1e-6. */
  eps?: number;
  /** "w": h * inv * w (Llama/Qwen, alias3-web); "1+w": h * inv * (1 + w) (Gemma). */
  weight?: "w" | "1+w";
}

export interface NormWeights { kind: "rmsnorm"; w: Float32Array; eps: number; weight: "w" | "1+w"; hiddenSize: number }

/** norm.bin (alias3-web layout): u32 LE json length, json {kind?, hiddenSize}, fp32 LE weight [hiddenSize]. */
export function loadNorm(buf: ArrayBuffer, spec: NormSpecLike = {}): NormWeights {
  const dv = new DataView(buf);
  const n = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  const H = meta.hiddenSize;
  if (!Number.isInteger(H) || H < 1) throw new Error("norm: bad metadata " + JSON.stringify(meta));
  if (meta.kind !== undefined && !/rmsnorm/.test(String(meta.kind))) throw new Error(`norm: kind ${JSON.stringify(meta.kind)} is not an RMSNorm`);
  if (buf.byteLength !== 4 + n + 4 * H) throw new Error(`norm: ${buf.byteLength} bytes, expected ${4 + n + 4 * H}`);
  const w = new Float32Array(H);
  for (let i = 0; i < H; i++) w[i] = dv.getFloat32(4 + n + 4 * i, true);
  const eps = spec.eps ?? 0;
  if (!(Number.isFinite(eps) && eps >= 0)) throw new Error(`norm: eps ${eps} must be >= 0`);
  const weight = spec.weight ?? "w";
  if (weight !== "w" && weight !== "1+w") throw new Error(`norm: weight ${JSON.stringify(weight)} must be "w" | "1+w"`);
  return { kind: "rmsnorm", w, eps, weight, hiddenSize: H };
}

/** RMSNorm in fp32: h / sqrt(mean(h^2) + eps) * (w | 1 + w). */
export function applyNorm(nw: NormWeights, h: Float32Array): Float32Array {
  if (h.length !== nw.hiddenSize) throw new Error(`norm: hidden size ${h.length} != ${nw.hiddenSize}`);
  let s = 0;
  for (let i = 0; i < h.length; i++) s += h[i] * h[i];
  const inv = 1 / Math.sqrt(s / h.length + nw.eps);
  const o = new Float32Array(h.length);
  const one = nw.weight === "1+w" ? 1 : 0;
  for (let i = 0; i < h.length; i++) o[i] = h[i] * inv * (one + nw.w[i]);
  return o;
}
