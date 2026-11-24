// Shared tensor / sampling helpers (ported from ct0-edge backends/decodeOnly.ts).

const F16: any = (globalThis as any).Float16Array;

export function mulberry32(seed: number) {
  let a = seed >>> 0 || 1;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export function halfToNumber(h: number): number {
  const s = (h & 0x8000) ? -1 : 1, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  return e === 0 ? s * Math.pow(2, -14) * (f / 1024)
       : e === 0x1f ? (f ? NaN : s * Infinity)
       : s * Math.pow(2, e - 15) * (1 + f / 1024);
}

export function tensorToF32(data: any, type: string | undefined, off = 0, n = data.length - off): Float32Array {
  const out = new Float32Array(n);
  const isF16Arr = typeof F16 !== "undefined" && data instanceof F16;
  if (data instanceof Float32Array || isF16Arr) { for (let i = 0; i < n; i++) out[i] = Number(data[off + i]); }
  else if (type === "float16" || data instanceof Uint16Array) { for (let i = 0; i < n; i++) out[i] = halfToNumber(data[off + i]); }
  else for (let i = 0; i < n; i++) out[i] = Number(data[off + i]);
  return out;
}

export function lastRow(L: any): Float32Array {
  const V = L.dims.at(-1), off = (L.dims.at(-2) - 1) * V;
  return tensorToF32(L.data, L.type, off, V);
}

export function sample(logits: Float32Array, gen: number[], rand: () => number, o: { temperature: number; topP: number; topK: number; repetitionPenalty: number }): number {
  const { temperature: temp, topP, topK, repetitionPenalty: rep } = o;
  if (rep !== 1) for (const t of new Set(gen)) logits[t] = logits[t] > 0 ? logits[t] / rep : logits[t] * rep;
  if (temp <= 0) { let bi = 0, bv = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; bi = i; } return bi; }
  let idx = Array.from(logits.keys()); idx.sort((a, b) => logits[b] - logits[a]);
  if (topK > 0) idx = idx.slice(0, topK);
  const mx = logits[idx[0]]; let probs = idx.map((i) => Math.exp((logits[i] - mx) / temp));
  const s = probs.reduce((a, b) => a + b, 0); probs = probs.map((p) => p / s);
  if (topP < 1) { let c = 0, cut = idx.length; for (let i = 0; i < idx.length; i++) { c += probs[i]; if (c >= topP) { cut = i + 1; break; } } idx = idx.slice(0, cut); probs = probs.slice(0, cut); const z = probs.reduce((a, b) => a + b, 0); probs = probs.map((p) => p / z); }
  const r = rand(); let acc = 0; for (let i = 0; i < idx.length; i++) { acc += probs[i]; if (r <= acc) return idx[i]; } return idx[idx.length - 1];
}

/** present.* output -> past.* input name (ONNX Runtime GenAI / optimum / custom naming). */
export function pastNameFor(out: string): string | null {
  let m: RegExpMatchArray | null;
  if ((m = out.match(/^present_conv\.(\d+)$/))) return "past_conv." + m[1];
  if ((m = out.match(/^present_recurrent\.(\d+)$/))) return "past_recurrent." + m[1];
  if ((m = out.match(/^present\.(\d+)\.(key|value)$/))) return `past_key_values.${m[1]}.${m[2]}`;
  if ((m = out.match(/^present\.(\d+)\.(conv|recurrent)$/))) return `past.${m[1]}.${m[2]}`;
  if ((m = out.match(/^so(\d+)$/))) return "si" + m[1];
  return null;
}

export function cacheDims(name: string, shape: any[], pastLen: number): number[] {
  const isKV = /^past_key_values\./.test(name) || (shape.length === 4 && !/conv|recurrent/.test(name) && typeof shape[2] !== "number");
  return shape.map((d: any, i: number) => (typeof d === "number" && d >= 0 ? d : isKV && i === 2 ? pastLen : 1));
}

export function inputMeta(session: any): Map<string, any> {
  const meta: any[] = Array.isArray(session.inputMetadata) ? session.inputMetadata
    : Object.entries(session.inputMetadata ?? {}).map(([name, m]: any) => ({ name, ...m }));
  return new Map(meta.map((m) => [m.name, m]));
}

/** The manifest names the hidden output; assert the graph has it (no name scanning). */
export function resolveHidden(name: string, outNames: string[], hiddenOutput: string | null | undefined): string | null {
  if (hiddenOutput == null) return null;
  if (!outNames.includes(hiddenOutput)) throw new Error(`${name}: manifest hiddenOutput ${JSON.stringify(hiddenOutput)} is not an output of the graph (outputs: ${outNames.slice(0, 4).join(", ")}${outNames.length > 4 ? ", ..." : ""})`);
  return hiddenOutput;
}

/** The forecast layer is a raw residual stream in the graph dtype: refuse NaN / Infinity (fp16 overflow) with a clear message. */
export function assertFiniteRow(name: string, output: string, row: Float32Array): void {
  for (let i = 0; i < row.length; i++) if (!Number.isFinite(row[i])) throw new Error(`${name}: forecast output ${output} is not finite at index ${i} (fp16 overflow of the raw residual stream, or a broken graph); use a build in fp32 or another layer`);
}

export function checkOrtVersion(v: string | undefined, min: number[] = [1, 30]): void {
  const m = /^(\d+)\.(\d+)/.exec(v ?? "");
  if (!m) return;
  if (Number(m[1]) < min[0] || (Number(m[1]) === min[0] && Number(m[2]) < min[1]))
    throw new Error(`onnxruntime ${v} is too old; requires >= ${min.join(".")} (WebGPU returns NaN below that; genai needs the LinearAttention/CausalConvWithState kernels). Pass skipVersionCheck to ignore.`);
}
