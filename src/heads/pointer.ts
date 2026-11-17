// Pointer (bilinear) head stored as safetensors in fp32 (Kev head.pt converted, the model cards' head.safetensors):
// tensors q.weight [D, H], q.bias [D], k.weight [D, H], k.bias [D]. Same math as head.bin (readout.ts headScore):
// z = scale * (Wq h_q + bq) . (Wk h_k + bk) / T + score_offset; no fp16 rounding of the weights.
// Reference: sglang@da614ba python/sglang/srt/layers/pointer_head.py PointerPooler (scale 1/sqrt(head_dim)).

import type { HeadWeights } from "../readout.js";
import { parseSafetensors, need } from "./safetensors.js";

export interface PointerSpecLike { scale?: number; temperature?: number; scoreOffset?: number }

export function loadPointerSafetensors(buf: ArrayBuffer, spec: PointerSpecLike = {}): HeadWeights {
  const st = parseSafetensors(buf);
  const q = need(st, "q.weight"), k = need(st, "k.weight");
  if (q.shape.length !== 2 || k.shape.length !== 2 || q.shape[0] !== k.shape[0] || q.shape[1] !== k.shape[1]) throw new Error(`pointer head: q/k shapes [${q.shape}] / [${k.shape}]`);
  const [D, H] = q.shape;
  const md = st.metadata ?? {};
  const num = (x: any) => (x === undefined ? undefined : Number(x));
  const scale = spec.scale ?? num(md.scale) ?? 1 / Math.sqrt(D);
  const temperature = spec.temperature ?? num(md.temperature) ?? 1;
  const scoreOffset = spec.scoreOffset ?? num(md.score_offset) ?? 0;
  if (!(scale > 0) || !(temperature > 0) || !Number.isFinite(scoreOffset)) throw new Error(`pointer head: scale ${scale}, temperature ${temperature}, score_offset ${scoreOffset}`);
  const bias = (n: string) => (st.tensors[n] ? need(st, n, [D]).data : new Float32Array(D));
  return { q: q.data, qb: bias("q.bias"), k: k.data, kb: bias("k.bias"), headDim: D, hiddenSize: H, temperature, scoreOffset, scale };
}
