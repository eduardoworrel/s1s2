
// WebLLM (MLC) backend (ported from ct0-edge webllm.ts, generalized). Logits are captured with a
// LogitProcessor; hidden states, when a recovery matrix is pinned in the manifest, are recovered
// from the logits (h = P . logits[rows]). WebLLM takes text prompts and keeps no addressable
// state, so it has no shared prefill (sharedPrefill: false).

import { gateFor, resolveGenerateOptions, checkPositions, type Backend, type GenerateOptions } from "../backend.js";
import { halfToNumber } from "./common.js";

export interface LogitCapture {

  processor: { processLogits(l: Float32Array): Float32Array; processSampledToken(t: number): void; resetState(): void };

  arm(): void;

  take(): Float32Array | null;

  drain(): number[];

  reset(): void;
}

export function logitCapture(): LogitCapture {
  let armed = false, cap: Float32Array | null = null, sampled: number[] = [];
  return {
    processor: {
      processLogits(l) { if (armed) { cap = new Float32Array(l); armed = false; } return l; },
      processSampledToken(t) { sampled.push(t); },
      resetState() {  },
    },
    arm() { armed = true; cap = null; },
    take() { const c = cap; cap = null; armed = false; return c; },
    drain() { const s = sampled; sampled = []; return s; },
    reset() { armed = false; cap = null; sampled = []; },
  };
}

export interface Recovery {

  rows: Int32Array;

  P: Float32Array | Uint16Array;
}
interface RecoveryF32 { rows: Int32Array; P: Float32Array; D: number }

function normalizeRecovery(r: Recovery): RecoveryF32 {
  const N = r.rows.length;
  if (!N || r.P.length % N) throw new Error(`webllm: P (${r.P.length}) is not a multiple of rows (${N})`);
  let P: Float32Array;
  if (r.P instanceof Uint16Array) { P = new Float32Array(r.P.length); for (let i = 0; i < P.length; i++) P[i] = halfToNumber(r.P[i]); }
  else P = r.P;
  return { rows: r.rows, P, D: P.length / N };
}

export function recoverHidden(rec: Recovery | RecoveryF32, logits: Float32Array): Float32Array {
  const r = "D" in rec ? rec : normalizeRecovery(rec);
  const N = r.rows.length, sub = new Float32Array(N), h = new Float32Array(r.D);
  for (let n = 0; n < N; n++) sub[n] = logits[r.rows[n]];
  for (let d = 0; d < r.D; d++) { let acc = 0; const off = d * N; for (let n = 0; n < N; n++) acc += r.P[off + n] * sub[n]; h[d] = acc; }
  return h;
}

export async function loadRecovery(rowsUrl: string, pUrl: string, format: "f32" | "f16" = pUrl.endsWith(".f16") ? "f16" : "f32", fetchFn: typeof fetch = fetch): Promise<Recovery> {
  const [rb, pb] = await Promise.all([fetchFn(rowsUrl).then((r) => { if (!r.ok) throw new Error("webllm: " + rowsUrl + " " + r.status); return r.arrayBuffer(); }),
    fetchFn(pUrl).then((r) => { if (!r.ok) throw new Error("webllm: " + pUrl + " " + r.status); return r.arrayBuffer(); })]);
  return { rows: new Int32Array(rb), P: format === "f16" ? new Uint16Array(pb) : new Float32Array(pb) };
}

export async function tryLoadRecovery(base: string, files: { rows: string; P: string }, fetchFn: typeof fetch = fetch):
  Promise<{ recovery?: Recovery; reason?: string }> {
  try {
    const rec = await loadRecovery(new URL(files.rows, base).href, new URL(files.P, base).href, files.P.endsWith(".f16") ? "f16" : "f32", fetchFn);
    const N = rec.rows.length;
    if (!N || rec.P.length % N) throw new Error(`P (${rec.P.length}) is not a multiple of rows (${N})`);
    return { recovery: rec };
  } catch (e: any) { return { reason: e?.message ?? String(e) }; }
}

export interface WebllmBackendOptions {
  /** Tokenizer with decode(ids) (lib Tok); WebLLM only accepts text. */
  tok: { decode(ids: number[], o?: { skipSpecial?: boolean }): string };
  recovery?: Recovery;
  /** EOS token ids (from the manifest). */
  eos: Set<number>;
}

export function webllmBackend(engine: any, capture: LogitCapture, opts: WebllmBackendOptions): Backend {
  if (!opts.tok?.decode) throw new Error("webllm: opts.tok (tokenizer with decode) is required; WebLLM only accepts text");
  if (!(opts.eos instanceof Set) || !opts.eos.size) throw new Error("webllm: opts.eos (Set of EOS ids from the manifest) is required");
  const eos = opts.eos;
  let rec: RecoveryF32 | null = opts.recovery ? normalizeRecovery(opts.recovery) : null;

  const gate = gateFor(engine, "webllm");
  const decode = (ids: number[]) => opts.tok.decode(ids, { skipSpecial: false });

  async function logitsAt(ids: number[], k: number): Promise<Float32Array> {
    await engine.resetChat();
    capture.reset(); capture.arm();
    const r = await engine.completions.create({ prompt: decode(ids.slice(0, k + 1)), max_tokens: 1, temperature: 0 });
    const L = capture.take();
    if (!L) throw new Error("webllm: the LogitProcessor captured no logits (engine created without this capture's logitProcessorRegistry, or a Web Worker?)");
    const n = r?.usage?.prompt_tokens;
    if (n !== undefined && n !== k + 1) throw new Error(`webllm: the prompt truncated at position ${k} re-tokenized to ${n} tokens (expected ${k + 1}); the position read would be wrong`);
    return L;
  }

  return {
    kind: "webllm",
    hiddenName: rec ? "recovered-from-logits" : null,
    dispose() { return gate.dispose(async () => { rec = null; await engine.unload?.(); }); },
    async prefill(ids, want) {
      gate.assertLive();
      checkPositions("webllm", ids, want.positions);
      if (want.positions.length && !rec) throw new Error("webllm: without opts.recovery the backend cannot recover hidden states; use method 'logits'");
      const rel = await gate.acquire();
      try {
        const last = ids.length - 1;
        const lastLogits = await logitsAt(ids, last);
        const at = new Map<number, Float32Array>();
        for (const p of new Set(want.positions)) at.set(p, recoverHidden(rec!, p === last ? lastLogits : await logitsAt(ids, p)));
        return { hiddens: want.positions.map((p) => at.get(p)!.slice()), lastLogits };
      } finally { rel(); }
    },

    async *generate(ids, o: GenerateOptions) {
      gate.assertLive();
      const g = resolveGenerateOptions(o); 
      if (!ids.length) throw new Error("generate: empty ids");
      if (g.maxNewTokens === 0) return;
      const rel = await gate.acquire();
      let iter: AsyncIterator<any> | null = null, finished = false;
      try {
        if (o.signal?.aborted) return;
        await engine.resetChat();
        capture.reset();
        const stream = await engine.completions.create({ prompt: decode(ids), stream: true, max_tokens: g.maxNewTokens,
          temperature: g.temperature, top_p: g.topP, seed: g.seed, repetition_penalty: g.repetitionPenalty });
        iter = stream[Symbol.asyncIterator]();
        let n = 0, stop = false;
        while (!stop) {
          if (o.signal?.aborted) break;
          const nx = await iter!.next();
          const toks = capture.drain();
          if (nx.done) finished = true;
          for (const t of toks) {
            if (eos.has(t)) { stop = true; break; }
            if (n >= g.maxNewTokens) { stop = true; break; }
            n++; yield t;
            if (o.signal?.aborted) { stop = true; break; }
          }
          if (nx.done) break;
        }
      } finally {

        if (iter) {
          try {
            if (!finished) { await engine.interruptGenerate(); for (let i = 0; i < 100000; i++) { const nx = await iter.next(); if (nx.done) break; } }
          } catch {  }
          try { await iter.return?.(); } catch {  }
        }
        capture.reset();
        rel();
      }
    },
  };
}

export interface CreateWebllmOptions extends WebllmBackendOptions {

  modelId: string;

  modelUrl?: string;
  libUrl?: string;
  appConfig?: any;
  initProgressCallback?: (p: any) => void;

  recoveryUrls?: { rows: string; P: string; format?: "f32" | "f16" };
}

export async function createWebllm(webllm: any, opts: CreateWebllmOptions): Promise<Backend & { engine: any; capture: LogitCapture }> {
  const capture = logitCapture();
  const appConfig = opts.appConfig ?? { model_list: [{ model: opts.modelUrl, model_id: opts.modelId, model_lib: opts.libUrl }] };
  const recovery = opts.recovery ?? (opts.recoveryUrls ? await loadRecovery(opts.recoveryUrls.rows, opts.recoveryUrls.P, opts.recoveryUrls.format) : undefined);
  const engine = await webllm.CreateMLCEngine(opts.modelId, { appConfig, logitProcessorRegistry: new Map([[opts.modelId, capture.processor]]),
    initProgressCallback: opts.initProgressCallback });
  return Object.assign(webllmBackend(engine, capture, { tok: opts.tok, recovery, eos: opts.eos }), { engine, capture });
}
