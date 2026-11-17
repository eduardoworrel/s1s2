// Minimal safetensors reader (https://github.com/huggingface/safetensors: u64 LE header length, JSON header
// {name: {dtype, shape, data_offsets}}, raw little-endian data). F32, F16 and BF16 are widened to Float32Array.

export interface Tensor { dtype: string; shape: number[]; data: Float32Array }
export interface SafeTensors { metadata: Record<string, string>; tensors: Record<string, Tensor> }

function f16(h: number): number {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

export function parseSafetensors(buf: ArrayBuffer): SafeTensors {
  const dv = new DataView(buf);
  const n = Number(dv.getBigUint64(0, true));
  if (!(n > 1 && 8 + n <= buf.byteLength)) throw new Error(`safetensors: bad header length ${n}`);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const base = 8 + n, tensors: Record<string, Tensor> = {};
  for (const [name, t] of Object.entries<any>(head)) {
    if (name === "__metadata__") continue;
    const [a, b] = t.data_offsets as [number, number], count = (t.shape as number[]).reduce((x, y) => x * y, 1);
    if (!(base + b <= buf.byteLength && b >= a)) throw new Error(`safetensors: ${name} outside the file`);
    const out = new Float32Array(count);
    const w = t.dtype === "F32" ? 4 : t.dtype === "F16" || t.dtype === "BF16" ? 2 : 0;
    if (!w) throw new Error(`safetensors: ${name} dtype ${t.dtype} not supported (F32, F16, BF16)`);
    if (b - a !== count * w) throw new Error(`safetensors: ${name} has ${b - a} bytes for ${count} x ${t.dtype}`);
    const o = base + a;
    if (t.dtype === "F32") for (let i = 0; i < count; i++) out[i] = dv.getFloat32(o + 4 * i, true);
    else if (t.dtype === "F16") for (let i = 0; i < count; i++) out[i] = f16(dv.getUint16(o + 2 * i, true));
    else { const u = new Uint32Array(1), f = new Float32Array(u.buffer); for (let i = 0; i < count; i++) { u[0] = dv.getUint16(o + 2 * i, true) << 16; out[i] = f[0]; } }
    tensors[name] = { dtype: t.dtype, shape: t.shape, data: out };
  }
  return { metadata: head.__metadata__ ?? {}, tensors };
}

export function need(st: SafeTensors, name: string, shape?: number[]): Tensor {
  const t = st.tensors[name];
  if (!t) throw new Error(`safetensors: missing tensor ${name} (has ${Object.keys(st.tensors).slice(0, 8).join(", ")}…)`);
  if (shape && (t.shape.length !== shape.length || t.shape.some((d, i) => shape[i] >= 0 && d !== shape[i])))
    throw new Error(`safetensors: ${name} has shape [${t.shape}], expected [${shape.map((d) => (d < 0 ? "*" : d))}]`);
  return t;
}
