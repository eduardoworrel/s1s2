// Linear readout head (spec 0.2 section 6.3): K trained rows over the hidden state at ONE read position,
// z = W h + b, optional softcap c * tanh(z / c), the options of a question scored by the rows of its type in order,
// then z / T per type. Covers pplx-decider / d3 `readout.safetensors` (rows = option codes A, B, ..., all types from
// row 0), JEV / autotrust `head.safetensors` (`proj`, slot ranges per type, optional softcap), and Standard One
// `jev-gpu-readout-v2` classifier mode. Reference: transformers@31d7392 models/jev/modeling_jev.py JevForDecision.forward.

import type { QType } from "../manifest.js";
import { parseSafetensors, need } from "./safetensors.js";

export interface LinearHeadSpecLike {
  /** Tensor names in the safetensors file (default "weight" / "bias"; JEV: "proj.weight" / "proj.bias"). */
  tensors?: { weight?: string; bias?: string | null };
  /** First row (and optional end) of each question type; options of a question read rows start..start+n-1 in order. */
  rows?: Partial<Record<QType | "evidence", { start: number; end?: number }>>;
  /** softcap c: z -> c * tanh(z / c), applied to every row before the rows of a question are taken. */
  softcap?: number | null;
  /** Temperature per question type: p = softmax(z / T). Default 1. */
  temperature?: Partial<Record<QType | "evidence", number>>;
  read?: "decide" | "last";
}

export interface LinearHead {
  kind: "linear";
  W: Float32Array; b: Float32Array | null; rows: number; hiddenSize: number;
  start: Partial<Record<QType | "evidence", { start: number; end: number }>>;
  softcap: number | null;
  temperature: Partial<Record<QType | "evidence", number>>;
  /** Read position: hidden state at the decide marker (default) or at the last prompt token. */
  read: "decide" | "last";
}

export function loadLinearHead(buf: ArrayBuffer, spec: LinearHeadSpecLike = {}): LinearHead {
  const st = parseSafetensors(buf);
  const wn = spec.tensors?.weight ?? "weight", bn = spec.tensors?.bias === undefined ? "bias" : spec.tensors.bias;
  const W = need(st, wn);
  if (W.shape.length !== 2) throw new Error(`linear head: ${wn} must be [rows, hidden], got [${W.shape}]`);
  const [K, H] = W.shape;
  const b = bn !== null && st.tensors[bn] ? need(st, bn, [K]).data : null;
  if (bn !== null && spec.tensors?.bias && !b) throw new Error(`linear head: bias tensor ${bn} missing`);
  const start: LinearHead["start"] = {};
  for (const t of ["predicate", "choice", "score", "evidence"] as const) {
    const r = spec.rows?.[t] ?? (spec.rows ? undefined : { start: 0 });
    if (!r) continue;
    const end = r.end ?? K;
    if (!(Number.isInteger(r.start) && r.start >= 0 && Number.isInteger(end) && end > r.start && end <= K)) throw new Error(`linear head: rows.${t} ${JSON.stringify(r)} outside [0, ${K}]`);
    start[t] = { start: r.start, end };
  }
  const c = spec.softcap ?? null;
  if (c !== null && !(c > 0)) throw new Error(`linear head: softcap ${c} must be > 0 or null`);
  for (const [t, v] of Object.entries(spec.temperature ?? {})) if (!(typeof v === "number" && v > 0)) throw new Error(`linear head: temperature.${t} must be > 0`);
  return { kind: "linear", W: W.data, b, rows: K, hiddenSize: H, start, softcap: c, temperature: { ...(spec.temperature ?? {}) }, read: spec.read ?? "decide" };
}

/** Raw scores of the n options of a question of type `t` (after softcap, before temperature). */
export function linearRaw(hd: LinearHead, h: Float32Array, t: QType | "evidence", n: number): number[] {
  if (h.length !== hd.hiddenSize) throw new Error(`linear head: hidden size ${h.length} != ${hd.hiddenSize}`);
  const r = hd.start[t];
  if (!r) throw new Error(`linear head: no rows for question type ${t}`);
  if (n > r.end - r.start) throw new Error(`linear head: a ${t} question with ${n} options, the head scores at most ${r.end - r.start}`);
  const z: number[] = [];
  for (let i = 0; i < n; i++) {
    const row = (r.start + i) * hd.hiddenSize;
    let a = hd.b ? hd.b[r.start + i] : 0;
    for (let j = 0; j < hd.hiddenSize; j++) a += hd.W[row + j] * h[j];
    z.push(hd.softcap ? hd.softcap * Math.tanh(a / hd.softcap) : a);
  }
  return z;
}

/** Scores the runtime turns into probabilities: raw / T(type). */
export function linearScores(hd: LinearHead, h: Float32Array, t: QType | "evidence", n: number): number[] {
  const T = hd.temperature[t] ?? 1;
  return linearRaw(hd, h, t, n).map((z) => z / T);
}
