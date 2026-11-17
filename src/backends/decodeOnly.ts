// Decode-only ONNX graph backend (one token per run; ported from ct0-edge backends/decodeOnly.ts).
// State is a plain record of tensors that ONNX Runtime never mutates, so a kept state is just a
// reference: prefill(keep) / prefillFrom(keep) / generateFrom share the decision prefill.

import { createGate, resolveGenerateOptions, checkContinuation, checkPositions, type Backend, type GenerateOptions, type StateHandle, type PrefillResult } from "../backend.js";
import { tensorToF32, lastRow, mulberry32, sample, pastNameFor, cacheDims, inputMeta, resolveHidden, checkOrtVersion } from "./common.js";

const F16: any = (globalThis as any).Float16Array;

export interface DecodeOnlyOptions {
  ort: { Tensor: new (type: any, data: any, dims: number[]) => any; env?: { versions?: { web?: string } } };
  eos: Set<number>;
  hiddenOutput: string | null;
  skipVersionCheck?: boolean;
  pastLen?: number;
}

interface CacheSpec { name: string; type: string; dims: number[] }

export async function createDecodeOnly(session: any, opts: DecodeOnlyOptions): Promise<Backend> {
  if (!(opts?.eos instanceof Set) || !opts.eos.size) throw new Error("decode-only: opts.eos (Set of EOS ids from the manifest) is required");
  if (!opts.skipVersionCheck) checkOrtVersion(opts.ort.env?.versions?.web);
  const T = opts.ort.Tensor, eos = opts.eos;
  const pastLen = opts.pastLen ?? 1;
  const inNames: string[] = session.inputNames, outNames: string[] = session.outputNames;
  if (inNames.includes("attention_mask")) throw new Error("decode-only: a graph with attention_mask is not decode-only");
  const metaBy = inputMeta(session);
  const specs: CacheSpec[] = [];
  for (const n of inNames) {
    if (n === "input_ids" || n === "cache_position") continue;
    const m = metaBy.get(n);
    if (!m?.shape) throw new Error("decode-only: no shape metadata for input " + n);
    specs.push({ name: n, type: m.type, dims: cacheDims(n, m.shape, pastLen) });
  }
  const pastOf = new Map<string, string>();
  for (const o of outNames) { const p = pastNameFor(o); if (p && inNames.includes(p)) pastOf.set(o, p); }
  const missing = specs.filter((s) => ![...pastOf.values()].includes(s.name));
  if (missing.length) throw new Error("decode-only: cache inputs without a matching present output: " + missing.slice(0, 3).map((s) => s.name).join(", "));
  const hiddenName = resolveHidden("decode-only", outNames, opts.hiddenOutput);
  const hasCachePos = inNames.includes("cache_position");

  const zeros = (s: CacheSpec) => {
    const n = s.dims.reduce((a, b) => a * b, 1);
    if (s.type === "float32") return new T("float32", new Float32Array(n), s.dims);
    if (s.type === "float16") return new T("float16", F16 ? new F16(n) : new Uint16Array(n), s.dims);
    throw new Error(`decode-only: unsupported cache dtype: ${s.type} (${s.name})`);
  };
  const initState = () => { const st: Record<string, any> = {}; for (const s of specs) st[s.name] = zeros(s); return st; };

  const fetchBase = ["logits", ...pastOf.keys()];
  async function step(tid: number, pos: number, st: Record<string, any>, wantHidden: boolean) {
    const feed: Record<string, any> = { input_ids: new T("int64", BigInt64Array.from([BigInt(tid)]), [1, 1]), ...st };
    if (hasCachePos) feed.cache_position = new T("int64", BigInt64Array.from([BigInt(pos)]), [1]);
    const names = wantHidden && hiddenName ? [...fetchBase, hiddenName] : fetchBase;
    const r = await session.run(feed, names);
    const ns: Record<string, any> = {};
    for (const [o, p] of pastOf) ns[p] = r[o];
    const hidden = wantHidden && hiddenName ? tensorToF32(r[hiddenName].data, r[hiddenName].type) : undefined;
    return { logits: r.logits, st: ns, hidden };
  }

  const snaps = new WeakMap<StateHandle, Record<string, any>>();
  const handle = (ids: number[], st: Record<string, any>): StateHandle => {
    const h: StateHandle = { ids: ids.slice(), dispose() { snaps.delete(h); } };
    snaps.set(h, st);
    return h;
  };
  async function feed(ids: number[], from: number, st: Record<string, any>, positions: number[], keep?: boolean): Promise<PrefillResult> {
    const wanted = new Set(positions), at = new Map<number, Float32Array>();
    let last: any = null;
    for (let i = from; i < ids.length; i++) {
      const r = await step(ids[i], i, st, wanted.has(i));
      st = r.st; last = r.logits;
      if (r.hidden) at.set(i, r.hidden);
    }
    const res: PrefillResult = { hiddens: positions.map((p) => at.get(p)!), lastLogits: lastRow(last) };
    if (keep) res.state = handle(ids, st);
    return res;
  }
  const checkHidden = (positions: number[]) => {
    if (positions.length && !hiddenName) throw new Error("decode-only: this build has no hidden output; use method 'logits'");
  };
  const stateOf = (h: StateHandle) => { const st = snaps.get(h); if (!st) throw new Error("decode-only: unknown or released state handle"); return st; };

  async function* decodeLoop(ids: number[], from: number, st0: Record<string, any>, o: GenerateOptions) {
    const g = resolveGenerateOptions(o);
    const rand = mulberry32(g.seed);
    if (g.maxNewTokens === 0) return;
    const rel = await gate.acquire();
    try {
      let st = st0, last: any = null, pos = from;
      for (; pos < ids.length; pos++) { if (g.signal?.aborted) return; ({ logits: last, st } = await step(ids[pos], pos, st, false)); }
      const gen: number[] = [];
      for (let s = 0; s < g.maxNewTokens; s++) {
        if (g.signal?.aborted) break;
        const tid = sample(lastRow(last), gen, rand, g);
        if (eos.has(tid)) break;
        gen.push(tid); yield tid;
        if (s + 1 >= g.maxNewTokens || g.signal?.aborted) break;
        ({ logits: last, st } = await step(tid, pos, st, false)); pos++;
      }
    } finally { rel(); }
  }

  const gate = createGate("decode-only");
  return {
    kind: "decode-only",
    hiddenName,
    dispose() { return gate.dispose(() => session.release?.()); },
    async prefill(ids, want, o = {}) {
      gate.assertLive(); checkHidden(want.positions); checkPositions("decode-only", ids, want.positions);
      const rel = await gate.acquire();
      try { return await feed(ids, 0, initState(), want.positions, o.keep); } finally { rel(); }
    },
    async snapshot(ids) {
      gate.assertLive();
      if (!ids.length) throw new Error("decode-only: snapshot of empty ids");
      const rel = await gate.acquire();
      try {
        let st = initState();
        for (let i = 0; i < ids.length; i++) st = (await step(ids[i], i, st, false)).st;
        return handle(ids, st);
      } finally { rel(); }
    },
    async prefillFrom(h, ids, want, o = {}) {
      gate.assertLive();
      const st = stateOf(h);
      const k = checkContinuation("decode-only", h, ids, want.positions);
      checkHidden(want.positions);
      const rel = await gate.acquire();
      try { return await feed(ids, k, st, want.positions, o.keep); } finally { rel(); }
    },
    async *generate(ids, o) {
      gate.assertLive();
      if (!ids.length) throw new Error("generate: empty ids");
      yield* decodeLoop(ids, 0, initState(), o);
    },
    async *generateFrom(h, ids, o) {
      gate.assertLive();
      const st = stateOf(h);
      const k = checkContinuation("decode-only", h, ids, []);
      yield* decodeLoop(ids, k, st, o);
    },
  };
}
