// Backend contract (ported from ct0-edge backend.ts, generalized):
//  - prefill/prefillFrom can KEEP the resulting state (`keep: true`) and return a StateHandle;
//  - generateFrom(handle, ids) continues generation from a kept state (shared prefill).
// Backends without snapshot/prefillFrom/generateFrom fall back to a fresh prefill.

export interface GenerateOptions {
  maxNewTokens?: number; temperature?: number; topP?: number; topK?: number;
  repetitionPenalty?: number; seed?: number; signal?: AbortSignal;
}

export interface ResolvedGenerateOptions {
  maxNewTokens: number; temperature: number; topP: number; topK: number; repetitionPenalty: number; seed: number;
  signal?: AbortSignal;
}

export const GENERATE_DEFAULTS = { maxNewTokens: 512, temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1, seed: 0 } as const;

export function resolveGenerateOptions(o: GenerateOptions = {}): ResolvedGenerateOptions {
  const num = (v: unknown, d: number) => (typeof v === "number" && Number.isFinite(v) ? v : d);
  const D = GENERATE_DEFAULTS;
  const temperature = num(o.temperature, D.temperature);
  return {
    maxNewTokens: Math.max(0, Math.floor(num(o.maxNewTokens, D.maxNewTokens))),
    temperature, topP: num(o.topP, D.topP), topK: num(o.topK, D.topK),
    repetitionPenalty: num(o.repetitionPenalty, temperature <= 0 ? 1 : D.repetitionPenalty),
    seed: num(o.seed, D.seed), signal: o.signal,
  };
}

export interface PrefillResult {
  hiddens: Float32Array[];
  lastLogits: Float32Array;
  /** Present when called with keep: true on a backend that can keep state. */
  state?: StateHandle;
}

export interface StateHandle {
  /** Token ids the state covers (a prefix any continuation must extend). */
  readonly ids: readonly number[];
  dispose(): void | Promise<void>;
}

export interface Backend {
  kind: "genai" | "decode-only" | "transformers.js" | "webllm" | "litert" | (string & {});
  /** Graph output carrying the hidden state; null = logits only. */
  hiddenName: string | null;
  prefill(ids: number[], want: { positions: number[] }, opts?: { keep?: boolean }): Promise<PrefillResult>;
  generate(ids: number[], opts: GenerateOptions): AsyncGenerator<number>;
  dispose(): Promise<void>;
  snapshot?(ids: number[]): Promise<StateHandle>;
  prefillFrom?(h: StateHandle, ids: number[], want: { positions: number[] }, opts?: { keep?: boolean }): Promise<PrefillResult>;
  generateFrom?(h: StateHandle, ids: number[], opts: GenerateOptions): AsyncGenerator<number>;
}

/** True when the backend can keep a prefill and continue from it (shared prefill). */
export function canShareState(b: Backend): boolean {
  return typeof b.snapshot === "function" && typeof b.prefillFrom === "function" && typeof b.generateFrom === "function";
}

export function checkContinuation(name: string, h: StateHandle, ids: number[], positions: number[]): number {
  const k = h.ids.length;
  if (ids.length <= k || h.ids.some((x, i) => ids[i] !== x)) throw new Error(`${name}: ids must extend the kept prefix (${k} tokens) by at least one token`);
  for (const p of positions) if (!Number.isInteger(p) || p < k || p >= ids.length) throw new Error(`${name}: position ${p} outside the continuation [${k},${ids.length})`);
  return k;
}

export function checkPositions(name: string, ids: number[], positions: number[]): void {
  if (!ids.length) throw new Error(`${name}: empty ids`);
  for (const p of positions) if (!Number.isInteger(p) || p < 0 || p >= ids.length) throw new Error(`${name}: position ${p} out of [0,${ids.length})`);
}

export interface Gate {
  readonly disposed: boolean;
  acquire(): Promise<() => void>;
  assertLive(): void;
  dispose(release: () => unknown): Promise<void>;
}

/** Serializes calls on one backend; dispose waits for the running call. */
export function createGate(name: string): Gate {
  let tail: Promise<void> = Promise.resolve();
  let disposed = false;
  let disposing: Promise<void> | null = null;
  const err = () => new Error(`${name}: backend already disposed`);
  const lock = async () => { const prev = tail; let rel!: () => void; tail = new Promise<void>((r) => { rel = r; }); await prev; return rel; };
  return {
    get disposed() { return disposed; },
    assertLive() { if (disposed) throw err(); },
    async acquire() { if (disposed) throw err(); return lock(); },
    dispose(release) {
      if (!disposing) {
        disposed = true;
        disposing = (async () => { const rel = await lock(); try { await release(); } finally { rel(); } })();
      }
      return disposing;
    },
  };
}

const gates = new WeakMap<object, Gate>();
export function gateFor(shared: object, name: string): Gate {
  let g = gates.get(shared);
  if (!g) { g = createGate(name); gates.set(shared, g); }
  return g;
}
