// Forecast reader (spec/FORECAST.md): pooling of mid-layer rows of the context, a small head, temperature and Platt.
// The rows come from the SAME prefill as decide(); nothing here calls a backend.

import { FORECAST_BINS, type ForecastSpec, type ForecastPooling } from "./manifest.js";
import { loadForecastHead, forecastScores, type ForecastHead } from "./heads/forecastHead.js";

export interface ForecastBehavior { p: number; logit: number }
export interface ForecastCost {
  /** Tokens of the decide prompt the rows come from (already prefilled for decide()). */
  prefillTokens: number;
  /** Tokens run only for the forecast: always 0, the forecast reads the decide prefill. */
  extraPrefillTokens: number;
  headMs: number;
}
export interface ForecastResult {
  behaviors: Record<string, ForecastBehavior>;
  layer: number;
  pooling: ForecastPooling;
  /** Name of the question whose prompt the rows came from. */
  question: string;
  cost: ForecastCost;
}

/** Token positions of an n-token prompt the pooling reads. */
export function forecastPositions(pooling: ForecastPooling, n: number): number[] {
  if (!(Number.isInteger(n) && n > 0)) throw new Error(`forecast: prompt of ${n} tokens`);
  if (pooling === "last") return [n - 1];
  if (pooling === "bins32" && n < FORECAST_BINS) throw new Error(`forecast: pooling bins32 needs a prompt of at least ${FORECAST_BINS} tokens, got ${n}`);
  return Array.from({ length: n }, (_, i) => i);
}

/** Pool the rows of positions 0..n-1 (all of them for mean / bins32, the last one for "last"). */
export function poolRows(pooling: ForecastPooling, rows: Float32Array[]): Float32Array {
  if (!rows.length) throw new Error("forecast: no rows to pool");
  const H = rows[0].length;
  for (const r of rows) if (r.length !== H) throw new Error(`forecast: rows of different widths (${r.length} vs ${H})`);
  if (pooling === "last") return rows.length === 1 ? rows[0] : rows[rows.length - 1];
  const meanOf = (a: number, b: number, out: Float32Array, off: number) => {
    for (let j = 0; j < H; j++) { let s = 0; for (let i = a; i < b; i++) s += rows[i][j]; out[off + j] = s / (b - a); }
  };
  const n = rows.length;
  if (pooling === "mean") { const o = new Float32Array(H); meanOf(0, n, o, 0); return o; }
  if (n < FORECAST_BINS) throw new Error(`forecast: pooling bins32 needs at least ${FORECAST_BINS} tokens, got ${n}`);
  const o = new Float32Array(FORECAST_BINS * H);
  for (let b = 0; b < FORECAST_BINS; b++) meanOf(Math.floor((b * n) / FORECAST_BINS), Math.floor(((b + 1) * n) / FORECAST_BINS), o, b * H);
  return o;
}

const sigmoid = (x: number) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));

export class ForecastReader {
  constructor(readonly spec: ForecastSpec, readonly head: ForecastHead) {}

  static load(spec: ForecastSpec, bytes: Uint8Array, hiddenSize: number): ForecastReader {
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    const head = loadForecastHead(buf, spec.head.type, spec.head.dims, `forecast ${spec.id}: head ${spec.head.file.path}`);
    const want = (spec.pooling === "bins32" ? FORECAST_BINS : 1) * hiddenSize;
    if (head.in !== want) throw new Error(`forecast ${spec.id}: head input width ${head.in} != ${want} (pooling ${spec.pooling} x hiddenSize ${hiddenSize})`);
    if (head.out !== spec.behaviors.length) throw new Error(`forecast ${spec.id}: head has ${head.out} outputs for ${spec.behaviors.length} behaviors`);
    return new ForecastReader(spec, head);
  }

  get positionsFor() { return (n: number) => forecastPositions(this.spec.pooling, n); }

  /** Rows of the positions `positionsFor(n)` (in that order) -> behavior probabilities. */
  read(rows: Float32Array[], question: string, prefillTokens: number): ForecastResult {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const z = forecastScores(this.head, poolRows(this.spec.pooling, rows));
    const c = this.spec.calibration;
    const at = (v: number | number[], k: number) => (Array.isArray(v) ? v[k] : v);
    const behaviors: Record<string, ForecastBehavior> = {};
    this.spec.behaviors.forEach((b, k) => {
      let l = z[k] / at(c.temperature, k);
      if (c.platt) l = at(c.platt.a, k) * l + at(c.platt.b, k);
      if (!Number.isFinite(l)) throw new Error(`forecast ${this.spec.id}: behavior ${b.name} is not finite (check the layer output of this build)`);
      behaviors[b.name] = { p: sigmoid(l), logit: l };
    });
    const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
    return { behaviors, layer: this.spec.layer, pooling: this.spec.pooling, question, cost: { prefillTokens, extraPrefillTokens: 0, headMs: t1 - t0 } };
  }
}

export type { ForecastHead };
