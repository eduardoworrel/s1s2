// ONNX Runtime GenAI graph backend (ported from ct0-edge backends/genai.ts, generalized):
// chunked prefill with KV/conv/recurrent state, hidden state named by the manifest, EOS from
// the manifest, and kept states for shared prefill (prefill/prefillFrom keep + generateFrom).
// Exact reuse: onnxruntime-web 1.31 WebGPU int4 graphs take a different numeric path for every token fed in a block
// shorter than 32 tokens (deterministic, measured on Qwen3-0.6B in Chrome). So no prefill block is ever shorter than `minBlock` (32), and
// a kept state stops `minBlock` tokens before the end it was fed to, so any continuation from it is >= minBlock tokens:
// reuse is then bit-identical to a full prefill. Decode steps (one token) are the same in both paths.

import { createGate, resolveGenerateOptions, checkContinuation, checkPositions, type Backend, type GenerateOptions, type StateHandle, type PrefillResult } from "../backend.js";
import { tensorToF32, mulberry32, sample, pastNameFor, cacheDims, inputMeta, resolveHidden, checkOrtVersion } from "./common.js";
import { f16ToF32, type EmbeddingTable } from "./embeddings.js";

export interface GenaiOptions {
  /** EOS token ids (from the manifest). */
  eos: Set<number>;
  /** Graph output with the post-final-norm hidden state (from the manifest), or null. */
  hiddenOutput: string | null;
  providers?: string[];
  ort?: any;
  chunk?: number;
  verifyEp?: boolean | "force";
  epCache?: { getItem(k: string): string | null; setItem(k: string, v: string): void } | null;
  adapterTag?: string | (() => Promise<string>);
  maxCpuNodes?: number;
  kvHeadDim?: number;
  skipVersionCheck?: boolean;
  /** Token fed once after load (manifest warmup); omitted = no warmup. */
  warmupToken?: number;
  /** Key of the graph for the EP-placement cache (default: the URL, or the byte length). */
  modelKey?: string;
  /** Token embedding table for graphs built with exclude_embeds (input inputs_embeds instead of input_ids). */
  embeddings?: EmbeddingTable;
  /**
   * Shortest prefill block (default 32; 0 = off, the 0.1.2 chunking). ORT-web WebGPU int4 computes a block shorter than
   * 32 tokens on a different numeric path, so continuing from a kept state is exact only when every block is >= 32.
   */
  minBlock?: number;
}

/** Block lengths for feeding `len` tokens: `chunk`-sized, a short tail (< minBlock) merged into the previous block. */
export function blockPlan(len: number, chunk: number, minBlock: number): number[] {
  const out: number[] = [];
  for (let p = 0; p < len; p += chunk) out.push(Math.min(chunk, len - p));
  if (minBlock > 0 && out.length > 1 && out[out.length - 1] < minBlock) out[out.length - 2] += out.pop()!;
  return out;
}

export function parsePlacement(lines: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const l of lines) {
    // "Node(s) placed on [EP]" per EP; onnxruntime 1.31+ prints "All nodes placed on [EP]" when one EP takes the graph.
    const m = /(?:Node\(s\)|All nodes) placed on \[(\w+)\]\.\s*Number of nodes:\s*(\d+)/.exec(l);
    if (m) out[m[1]] = (out[m[1]] ?? 0) + Number(m[2]);
  }
  return out;
}

export function assertWebGpuPlacement(p: Record<string, number>, maxCpuNodes = 32): void {
  const gpu = Object.entries(p).filter(([k]) => /webgpu/i.test(k)).reduce((a, [, n]) => a + n, 0);
  const cpu = p.CPUExecutionProvider ?? 0;
  if (!Object.keys(p).length) throw new Error("genai: could not verify the EP placement (no 'Node(s) placed on' logs)");
  if (gpu === 0) throw new Error(`genai: no node on the WebGPU EP (placement: ${JSON.stringify(p)})`);
  if (cpu > maxCpuNodes) throw new Error(`genai: ${cpu} nodes on CPU (> ${maxCpuNodes}); probably a bundle without the WebGPU kernels. Placement: ${JSON.stringify(p)}`);
}

export type GenaiBackend = Backend & { chunk: number; minBlock: number };

export function genaiFromSession(session: any, ort: { Tensor: new (type: any, data: any, dims: number[], ...r: any[]) => any }, opts: GenaiOptions): GenaiBackend {
  const T = ort.Tensor;
  if (!(opts?.eos instanceof Set) || !opts.eos.size) throw new Error("genai: opts.eos (Set of EOS ids from the manifest) is required");
  const eos = opts.eos;
  const inNames: string[] = session.inputNames, outNames: string[] = session.outputNames;
  const metaBy = inputMeta(session);
  const byEmbeds = !inNames.includes("input_ids") && inNames.includes("inputs_embeds");
  if (!inNames.includes("input_ids") && !byEmbeds) throw new Error("genai: graph without input input_ids (or inputs_embeds)");
  const emb = opts.embeddings;
  let embType = "";
  if (byEmbeds) {
    if (!emb) throw new Error("genai: the graph takes inputs_embeds (built with exclude_embeds) but the build has no embeddings table; add the file role \"embeddings\" (fp16 [vocab, hidden]) to the manifest build");
    const m = metaBy.get("inputs_embeds");
    const h = m?.shape?.at(-1);
    if (typeof h === "number" && h !== emb.hidden) throw new Error(`genai: inputs_embeds hidden ${h} != embeddings table hidden ${emb.hidden}`);
    embType = m?.type ?? "float16";
    if (embType !== "float16" && embType !== "float32") throw new Error(`genai: unsupported inputs_embeds dtype ${embType}`);
  } else if (emb) throw new Error("genai: the build pins an embeddings table but the graph takes input_ids (embedding in the graph)");
  if (!outNames.includes("logits")) throw new Error("genai: graph without a logits output");
  const hasMask = inNames.includes("attention_mask"), hasPos = inNames.includes("position_ids");
  const specs: { name: string; type: string; dims: number[] }[] = [];
  for (const n of inNames) {
    if (n === "input_ids" || n === "inputs_embeds" || n === "attention_mask" || n === "position_ids") continue;
    const m = metaBy.get(n);
    if (!m?.shape) throw new Error("genai: no shape metadata for input " + n);
    const dims = cacheDims(n, m.shape, 0);
    if (/^past_key_values\./.test(n) && typeof m.shape[3] !== "number") {
      if (!opts.kvHeadDim) throw new Error(`genai: KV head dim of ${n} is symbolic; pass kvHeadDim (genai_config head_size)`);
      dims[3] = opts.kvHeadDim;
    }
    specs.push({ name: n, type: m.type, dims });
  }
  const pastOf = new Map<string, string>();
  for (const o of outNames) { const p = pastNameFor(o); if (p && inNames.includes(p)) pastOf.set(o, p); }
  const covered = new Set(pastOf.values());
  const missing = specs.filter((s) => !covered.has(s.name));
  if (missing.length) throw new Error("genai: cache inputs without a present output: " + missing.slice(0, 3).map((s) => s.name).join(", "));
  const hiddenName = resolveHidden("genai", outNames, opts.hiddenOutput);
  const posRows = (() => { const sh = metaBy.get("position_ids")?.shape; return typeof sh?.[0] === "number" ? sh[0] : 3; })();

  const zeros = (s: { type: string; dims: number[] }) => {
    const n = s.dims.reduce((a, b) => a * b, 1);
    if (s.type === "float16") return new T("float16", new Uint16Array(n), s.dims);
    if (s.type === "float32") return new T("float32", new Float32Array(n), s.dims);
    throw new Error(`genai: unsupported cache dtype: ${s.type}`);
  };
  const initState = () => { const st: Record<string, any> = {}; for (const s of specs) st[s.name] = zeros(s); return st; };
  const free = (st: Record<string, any> | null) => { if (st) for (const t of Object.values(st)) if (t?.location === "gpu-buffer") { try { t.dispose?.(); } catch { /* already freed */ } } };

  const gate = createGate("genai");
  const mb = Math.floor(opts.minBlock ?? 32);
  const minBlock = Number.isFinite(mb) && mb > 0 ? mb : 0;

  async function run(chunk: number[], past: number, st: Record<string, any>, wantLogits: boolean, wantHidden: boolean, keepInput: boolean) {
    const S = chunk.length;
    const feed: Record<string, any> = { ...st };
    if (byEmbeds) {
      const raw = await emb!.rows(chunk);
      feed.inputs_embeds = embType === "float16" ? new T("float16", raw, [1, S, emb!.hidden]) : new T("float32", f16ToF32(raw), [1, S, emb!.hidden]);
    } else feed.input_ids = new T("int64", BigInt64Array.from(chunk.map(BigInt)), [1, S]);
    if (hasMask) feed.attention_mask = new T("int64", new BigInt64Array(past + S).fill(1n), [1, past + S]);
    if (hasPos) {
      const pos = new BigInt64Array(posRows * S);
      for (let r = 0; r < posRows; r++) for (let i = 0; i < S; i++) pos[r * S + i] = BigInt(past + i);
      feed.position_ids = new T("int64", pos, [posRows, 1, S]);
    }
    const fetches = [...pastOf.keys()];
    if (wantLogits) fetches.unshift("logits");
    if (wantHidden && hiddenName) fetches.push(hiddenName);
    const r = await session.run(feed, fetches, { logSeverityLevel: 3 });
    const ns: Record<string, any> = {};
    for (const [o, p] of pastOf) ns[p] = r[o];
    if (!keepInput) free(st);
    return { st: ns, out: { logits: wantLogits ? r.logits : undefined, hidden: wantHidden && hiddenName ? r[hiddenName] : undefined } };
  }
  const read = async (t: any) => (typeof t.getData === "function" ? await t.getData(true) : t.data);

  /** Feed ids[from.past..] on top of `from.st` (never freed: it belongs to a kept handle). With `keepAt`, a block
   *  boundary falls there and the state at that position is returned (alive) as `mid`. */
  async function feedAll(ids: number[], wanted: Set<number>, wantLast: boolean, signal?: AbortSignal, from?: { st: Record<string, any>; past: number }, keepAt?: number) {
    let st = from?.st ?? initState(), past = from?.past ?? 0;
    const kept = from?.st;
    let mid: Record<string, any> | undefined;
    const start = past, cut = keepAt !== undefined && keepAt > start && keepAt < ids.length ? keepAt : undefined;
    const plan = cut === undefined ? blockPlan(ids.length - start, self.chunk, self.minBlock)
      : [...blockPlan(cut - start, self.chunk, self.minBlock), ...blockPlan(ids.length - cut, self.chunk, self.minBlock)];
    const hid = new Map<number, Float32Array>();
    let lastLogits: Float32Array | undefined;
    const alive = (s: Record<string, any>) => s === kept || s === mid;
    try {
      for (const len of plan) {
        if (signal?.aborted) { if (mid && mid !== st) free(mid); return { st, past, hid, lastLogits, aborted: true, owned: st !== kept, mid: undefined }; }
        const chunk = ids.slice(past, past + len), last = past + chunk.length >= ids.length;
        const need = [...wanted].filter((p) => p >= past && p < past + chunk.length);
        const r = await run(chunk, past, st, last && wantLast, need.length > 0, alive(st));
        st = r.st;
        if (r.out.hidden) {
          const H = r.out.hidden, D = H.dims.at(-1), d = await read(H);
          for (const p of need) hid.set(p, tensorToF32(d, H.type, (p - past) * D, D));
        }
        if (r.out.logits) {
          const L = r.out.logits, V = L.dims.at(-1), d = await read(L);
          lastLogits = tensorToF32(d, L.type, (L.dims.at(-2) - 1) * V, V);
        }
        past += chunk.length;
        if (past === cut) mid = st;
      }
    } catch (e) { if (st !== kept) free(st); if (mid && mid !== st) free(mid); throw e; }
    return { st, past, hid, lastLogits, aborted: false, owned: st !== kept, mid };
  }

  /** Where a kept state of a prefill over [start, n) stops: n - minBlock (so every continuation is >= minBlock tokens),
   *  provided [start, n - minBlock) is itself >= minBlock; otherwise null (no state is kept: continue by a full prefill). */
  const keepPoint = (start: number, n: number): number | null => {
    const M = self.minBlock;
    if (!M) return n;
    const kp = n - M;
    return kp - start >= M ? kp : null;
  };
  const checkTail = (k: number, n: number) => {
    if (self.minBlock && n - k < self.minBlock) throw new Error(`genai: continuation of ${n - k} tokens past the kept state; shared prefill feeds at least minBlock (${self.minBlock}) tokens so it stays exact on WebGPU. Keep the state earlier, or prefill from scratch`);
  };

  const snaps = new WeakMap<StateHandle, Record<string, any>>();
  const handle = (ids: number[], st: Record<string, any>): StateHandle => {
    let live: Record<string, any> | null = st;
    const h: StateHandle = { ids: ids.slice(), dispose: async () => {
      if (!live) return;
      const s0 = live; live = null; snaps.delete(h);
      let r: (() => void) | null = null;
      try { r = await gate.acquire(); } catch { /* backend disposed: buffers go with the session */ }
      try { free(s0); } finally { r?.(); }
    } };
    snaps.set(h, st);
    return h;
  };
  const finish = (ids: number[], f: Awaited<ReturnType<typeof feedAll>>, positions: number[], keep: boolean | undefined, kp: number | null): PrefillResult => {
    const res: PrefillResult = { hiddens: positions.map((p) => f.hid.get(p)!), lastLogits: f.lastLogits! };
    if (keep && kp === ids.length) { res.state = handle(ids, f.st); return res; }
    if (keep && kp !== null && f.mid) res.state = handle(ids.slice(0, kp), f.mid);
    if (f.owned) free(f.st);
    return res;
  };
  const prefillTo = async (ids: number[], positions: number[], keep: boolean | undefined, from?: { st: Record<string, any>; past: number }) => {
    const kp = keep ? keepPoint(from?.past ?? 0, ids.length) : null;
    const f = await feedAll(ids, new Set(positions), true, undefined, from, kp ?? undefined);
    return finish(ids, f, positions, keep, kp);
  };
  const stateOf = (h: StateHandle) => { const st = snaps.get(h); if (!st) throw new Error("genai: unknown or released state handle"); return st; };
  const needHidden = (positions: number[]) => { if (positions.length && !hiddenName) throw new Error("genai: this build has no hidden output; use method 'logits'"); };

  async function* decodeLoop(ids: number[], o: GenerateOptions, from?: { st: Record<string, any>; past: number }) {
    const g = resolveGenerateOptions(o);
    const rand = mulberry32(g.seed);
    if (g.maxNewTokens === 0) return;
    const rel = await gate.acquire();
    let st: Record<string, any> | null = null, owned = false;
    try {
      if (o.signal?.aborted) return;
      if (from) checkTail(from.past, ids.length);
      const f = await feedAll(ids, new Set(), true, o.signal, from);
      st = f.st; owned = f.owned; if (f.aborted) return;
      let past = f.past, logits = f.lastLogits!;
      const gen: number[] = [];
      for (let s = 0; s < g.maxNewTokens; s++) {
        if (o.signal?.aborted) break;
        const tid = sample(logits, gen, rand, g);
        if (eos.has(tid)) break;
        gen.push(tid); yield tid;
        if (o.signal?.aborted || s + 1 >= g.maxNewTokens) break;
        const r = await run([tid], past, st!, true, false, !owned); st = r.st; owned = true; past++;
        const L = r.out.logits, V = L.dims.at(-1);
        logits = tensorToF32(await read(L), L.type, (L.dims.at(-2) - 1) * V, V);
      }
    } finally { if (owned) free(st); rel(); }
  }

  const self: GenaiBackend = {
    kind: "genai",
    minBlock,
    chunk: Math.max(1, Math.floor(opts.chunk ?? 128) || 1, minBlock),
    hiddenName,
    dispose() { return gate.dispose(async () => { try { await session.release?.(); } finally { await emb?.close?.(); } }); },
    async prefill(ids, want, o = {}) {
      gate.assertLive(); checkPositions("genai", ids, want.positions); needHidden(want.positions);
      const rel = await gate.acquire();
      try { return await prefillTo(ids, want.positions, o.keep); } finally { rel(); }
    },
    async snapshot(ids) {
      gate.assertLive();
      if (!ids.length) throw new Error("genai: snapshot of empty ids");
      if (ids.length < self.minBlock) throw new Error(`genai: snapshot of ${ids.length} tokens; a kept state needs at least minBlock (${self.minBlock}) tokens to stay exact on WebGPU`);
      const rel = await gate.acquire();
      try { return handle(ids, (await feedAll(ids, new Set(), false)).st); } finally { rel(); }
    },
    async prefillFrom(h, ids, want, o = {}) {
      gate.assertLive();
      const st = stateOf(h);
      const k = checkContinuation("genai", h, ids, want.positions); needHidden(want.positions);
      checkTail(k, ids.length);
      const rel = await gate.acquire();
      try { return await prefillTo(ids, want.positions, o.keep, { st, past: k }); } finally { rel(); }
    },
    async *generate(ids, o) {
      gate.assertLive();
      if (!ids.length) throw new Error("generate: empty ids");
      yield* decodeLoop(ids, o);
    },
    async *generateFrom(h, ids, o) {
      gate.assertLive();
      const st = stateOf(h);
      const k = checkContinuation("genai", h, ids, []);
      yield* decodeLoop(ids, o, { st, past: k });
    },
  };
  return self;
}

export function hashStr(x: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < x.length; i++) { h ^= x.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16).padStart(8, "0");
}

async function adapterTag(opts: GenaiOptions): Promise<string> {
  if (typeof opts.adapterTag === "string") return opts.adapterTag;
  if (typeof opts.adapterTag === "function") return opts.adapterTag();
  try {
    const ad = await (globalThis as any).navigator?.gpu?.requestAdapter?.();
    const i = ad?.info ?? (await ad?.requestAdapterInfo?.());
    return i ? hashStr([i.vendor, i.architecture, i.device, i.description].join("|")) : "?";
  } catch { return "?"; }
}

function epStore(opts: GenaiOptions) {
  if (opts.epCache !== undefined) return opts.epCache;
  try { return (globalThis as any).localStorage ?? null; } catch { return null; }
}

async function probePlacement(ort: any, model: string | Uint8Array, sessionOpts: (level: number) => any, maxCpuNodes: number) {
  const captured: string[] = [];
  const names = ["log", "info", "warn", "error", "debug"] as const;
  const saved: Record<string, any> = {};
  const prevLevel = ort.env?.logLevel;
  if (ort.env) ort.env.logLevel = "verbose";
  for (const n of names) { saved[n] = (console as any)[n]; (console as any)[n] = (...a: any[]) => { const s = a.join(" "); if (/(?:Node\(s\)|All nodes) placed on/.test(s)) captured.push(s); else if (!s.includes("[V:onnxruntime") && !/WebGPU Shader code|^\s*$/.test(s)) saved[n].apply(console, a); }; }
  let probe: any;
  try { probe = await ort.InferenceSession.create(model, sessionOpts(0)); }
  finally { for (const n of names) (console as any)[n] = saved[n]; if (ort.env) ort.env.logLevel = prevLevel; }
  try { assertWebGpuPlacement(parsePlacement(captured), maxCpuNodes); }
  finally { await probe.release?.(); }
}

/** Browser: create the session on WebGPU (default) or on the CPU (`providers: ["wasm"]`) from pinned URLs (model + external data). */
export async function createGenai(model: string | Uint8Array, dataUrl: string, dataName: string, opts: GenaiOptions): Promise<GenaiBackend> {
  const modelKey = opts.modelKey ?? (typeof model === "string" ? model : `bytes:${model.byteLength}`);
  const providers = opts.providers ?? ["webgpu"];
  if (providers.length === 1 && providers[0] === "wasm") return createGenaiWasm(model, dataUrl, dataName, opts);
  if (providers[0] !== "webgpu") throw new Error('genai: the browser graph runs on providers ["webgpu", ...] or ["wasm"]');
  const ort: any = opts.ort ?? await import("onnxruntime-web/webgpu");
  const verify = opts.verifyEp ?? false;
  const sessionOpts = (level: number) => ({
    executionProviders: providers.map((n) => (n === "webgpu" ? { name: "webgpu", validationMode: "disabled" } : n)),
    externalData: [{ path: dataName, data: dataUrl }],
    graphOptimizationLevel: "all", enableGraphCapture: false,
    preferredOutputLocation: "gpu-buffer",
    logSeverityLevel: level,
  });
  if (!opts.skipVersionCheck) checkOrtVersion(ort.env?.versions?.web);
  if (verify) {
    const store = epStore(opts);
    const maxCpu = opts.maxCpuNodes ?? 32;
    const key = store ? `s1s2:genai-ep:${ort.env?.versions?.web ?? "?"}:${await adapterTag(opts)}:${modelKey}:${dataUrl}:${maxCpu}` : "";
    let cached = false;
    if (verify !== "force" && store) { try { cached = store.getItem(key) === "ok"; } catch { /* no storage */ } }
    if (!cached) {
      await probePlacement(ort, model, sessionOpts, maxCpu);
      if (store) { try { store.setItem(key, "ok"); } catch { /* no storage */ } }
    }
  }
  const session = await ort.InferenceSession.create(model, sessionOpts(3));
  return wrapSession(session, ort, opts);
}

/**
 * Browser CPU path, as measured (Chrome on an Apple M2 Pro, 120 development items per task): the onnxruntime-web WASM
 * bundle (`onnxruntime-web/wasm`,
 * ort-wasm-simd-threaded.wasm), `executionProviders: ["wasm"]` with the external data, ORT's default threads and
 * optimization level, outputs on the CPU, no EP-placement probe; then the same wrapSession as WebGPU.
 */
async function createGenaiWasm(model: string | Uint8Array, dataUrl: string, dataName: string, opts: GenaiOptions): Promise<GenaiBackend> {
  const ort: any = opts.ort ?? await import("onnxruntime-web/wasm");
  if (!opts.skipVersionCheck) checkOrtVersion(ort.env?.versions?.web);
  let session: any;
  try { session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"], externalData: [{ path: dataName, data: dataUrl }] }); }
  catch (e: any) {
    const msg = String(e?.message ?? e);
    if (/Failed to find kernel/i.test(msg)) {
      const err = new Error(`genai on the browser CPU (WASM): onnxruntime-web has no WASM kernel for an op of this graph (${msg.slice(0, 300)}). This build needs WebGPU.`);
      (err as any).cause = e; throw err;
    }
    throw e;
  }
  return wrapSession(session, ort, opts);
}

/** Wrap an existing session (Node: onnxruntime-node with a local model path) and warm it up. */
export async function wrapSession(session: any, ort: any, opts: GenaiOptions): Promise<GenaiBackend> {
  try {
    const be = genaiFromSession(session, ort, opts);
    if (opts.warmupToken !== undefined) await be.prefill([opts.warmupToken], { positions: [] });
    return be;
  } catch (e) {
    try { await session.release?.(); } catch { /* ignore */ }
    throw e;
  }
}
