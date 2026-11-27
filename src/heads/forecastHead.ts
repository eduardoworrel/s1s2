// Forecast head file (spec/FORECAST.md): safetensors, fp32. `mean`/`std` [in] (training normalisation), then
//   linear: weight [out, in], bias [out]            z = W x + b
//   mlp:    w1 [hidden, in], b1, w2 [out, hidden], b2   z = w2 relu(w1 x + b1) + b2
// with x = (pooled - mean) / std. Float32 storage; sums run in double and are stored back to Float32.

import { parseSafetensors, need } from "./safetensors.js";

export interface ForecastHead {
  type: "linear" | "mlp";
  in: number; out: number; hidden?: number;
  mean: Float32Array; std: Float32Array;
  w1: Float32Array; b1: Float32Array;       // linear: weight / bias; mlp: first layer
  w2?: Float32Array; b2?: Float32Array;     // mlp: second layer
}

export interface ForecastHeadDims { in: number; out: number; hidden?: number }

export function loadForecastHead(buf: ArrayBuffer, type: "linear" | "mlp", dims: ForecastHeadDims, at = "forecast head"): ForecastHead {
  let st;
  try { st = parseSafetensors(buf); } catch (e: any) { throw new Error(`${at}: not a safetensors file: ${e?.message ?? e}`); }
  const t = (name: string, shape: number[]) => {
    try { return need(st, name, shape).data; }
    catch (e: any) { throw new Error(`${at}: ${e?.message ?? e} (manifest dims ${JSON.stringify(dims)})`); }
  };
  const mean = t("mean", [dims.in]), std = t("std", [dims.in]);
  for (let i = 0; i < std.length; i++) if (!(std[i] > 0) || !Number.isFinite(std[i])) throw new Error(`${at}: std[${i}] = ${std[i]} must be a finite number > 0`);
  if (type === "linear") return { type, in: dims.in, out: dims.out, mean, std, w1: t("weight", [dims.out, dims.in]), b1: t("bias", [dims.out]) };
  if (!dims.hidden) throw new Error(`${at}: an mlp head needs dims.hidden`);
  return { type, in: dims.in, out: dims.out, hidden: dims.hidden, mean, std,
    w1: t("w1", [dims.hidden, dims.in]), b1: t("b1", [dims.hidden]), w2: t("w2", [dims.out, dims.hidden]), b2: t("b2", [dims.out]) };
}

/** Raw scores z (before temperature / Platt) of one pooled feature vector. */
export function forecastScores(hd: ForecastHead, pooled: Float32Array): Float32Array {
  if (pooled.length !== hd.in) throw new Error(`forecast head: feature width ${pooled.length} != ${hd.in}`);
  const x = new Float32Array(hd.in);
  for (let i = 0; i < hd.in; i++) x[i] = (pooled[i] - hd.mean[i]) / hd.std[i];
  const dense = (W: Float32Array, b: Float32Array, v: Float32Array, rows: number, cols: number, relu: boolean) => {
    const o = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let a = b[r];
      const base = r * cols;
      for (let j = 0; j < cols; j++) a += W[base + j] * v[j];
      o[r] = relu && a < 0 ? 0 : a;
    }
    return o;
  };
  if (hd.type === "linear") return dense(hd.w1, hd.b1, x, hd.out, hd.in, false);
  return dense(hd.w2!, hd.b2!, dense(hd.w1, hd.b1, x, hd.hidden!, hd.in, true), hd.out, hd.hidden!, false);
}
