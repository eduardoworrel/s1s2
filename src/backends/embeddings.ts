// Token embedding table kept outside the graph (ONNX Runtime GenAI builder option exclude_embeds=true): the graph
// takes inputs_embeds instead of input_ids, and the lib looks the rows up. File layout (manifest file role
// "embeddings"): float16 little-endian, row-major [vocab, hidden], row `id` at byte offset id * hidden * 2 — what
// the lab-s12 build scripts write (embed_tokens.fp16.bin). No scaling is applied here: a model that
// scales its embeddings must do it in the graph.

export interface EmbeddingTable {
  hidden: number;
  vocab: number;
  /** Raw fp16 bits of the rows for `ids`, concatenated ([ids.length * hidden]). */
  rows(ids: number[]): Promise<Uint16Array>;
  close?(): Promise<void>;
}

export type RangeReader = (offset: number, length: number) => Promise<Uint8Array>;

/** Table over a byte-range reader, with a bounded row cache (rows are read once per distinct id). */
export function embeddingTable(read: RangeReader, size: number, hidden: number, opts: { cacheRows?: number; close?: () => Promise<void> } = {}): EmbeddingTable {
  const rowBytes = hidden * 2;
  if (!Number.isInteger(hidden) || hidden <= 0) throw new Error(`embeddings: bad hidden size ${hidden}`);
  if (size % rowBytes !== 0) throw new Error(`embeddings: file size ${size} is not a multiple of hidden * 2 = ${rowBytes} (expected fp16 [vocab, ${hidden}])`);
  const vocab = size / rowBytes;
  const max = opts.cacheRows ?? 8192;
  const cache = new Map<number, Uint16Array>();
  const row = async (id: number): Promise<Uint16Array> => {
    let r = cache.get(id);
    if (r) { cache.delete(id); cache.set(id, r); return r; }
    const b = await read(id * rowBytes, rowBytes);
    if (b.byteLength !== rowBytes) throw new Error(`embeddings: short read for token ${id} (${b.byteLength} of ${rowBytes} bytes)`);
    r = new Uint16Array(hidden);
    new Uint8Array(r.buffer).set(b);
    if (cache.size >= max) cache.delete(cache.keys().next().value!);
    cache.set(id, r);
    return r;
  };
  return {
    hidden, vocab,
    async rows(ids) {
      for (const id of ids) if (!Number.isInteger(id) || id < 0 || id >= vocab) throw new Error(`embeddings: token ${id} outside the table (vocab ${vocab})`);
      const got = new Map<number, Uint16Array>();
      for (const id of new Set(ids)) got.set(id, await row(id));
      const out = new Uint16Array(ids.length * hidden);
      ids.forEach((id, i) => out.set(got.get(id)!, i * hidden));
      return out;
    },
    close: opts.close,
  };
}

/** Whole table in memory (small tables, already sha256-verified bytes). */
export function embeddingTableFromBytes(bytes: Uint8Array, hidden: number): EmbeddingTable {
  return embeddingTable(async (o, n) => bytes.subarray(o, o + n), bytes.byteLength, hidden, { cacheRows: 0 });
}

/** Browser: HTTP Range reads of a large table (the server must answer 206; a full 200 body is refused). */
export function httpRangeTable(url: string, size: number, hidden: number, fetchFn: typeof fetch = fetch): EmbeddingTable {
  return embeddingTable(async (offset, length) => {
    const r = await fetchFn(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
    if (r.status !== 206) throw new Error(`embeddings: ${url} answered HTTP ${r.status} to a Range request; the table is read row by row, so the server must support Range (206)`);
    return new Uint8Array(await r.arrayBuffer());
  }, size, hidden);
}

/** Node: positional reads of a local (verified) file. */
export async function fileTable(path: string, hidden: number): Promise<EmbeddingTable> {
  const spec = "node:fs/promises";
  const fs: any = await import(/* @vite-ignore */ spec);
  const fh = await fs.open(path, "r");
  const { size } = await fh.stat();
  return embeddingTable(async (offset, length) => {
    const b = new Uint8Array(length);
    const { bytesRead } = await fh.read(b, 0, length, offset);
    return b.subarray(0, bytesRead);
  }, size, hidden, { close: () => fh.close() });
}

function h2f(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  return e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
}

/** fp16 bits -> float32 (for graphs whose inputs_embeds is float32). */
export function f16ToF32(b: Uint16Array): Float32Array {
  const out = new Float32Array(b.length);
  for (let i = 0; i < b.length; i++) out[i] = h2f(b[i]);
  return out;
}
