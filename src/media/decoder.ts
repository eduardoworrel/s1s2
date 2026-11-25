// Qwen3-ASR decoder questions: closed-set language scores and the five hypotheses (greedy + beam 4) a term is looked for in.
// Behaviour ported from marimba-audio 0.1.2 (_qwen.py: language_scores, hypotheses) and from the Hugging Face generation code
// it runs on (transformers 4.57.6, `_beam_search` after the 2025 refactor, early_stopping=False, length_penalty=1). Design: spec/MULTIMODAL.md, section 9.
//
// The graph takes `inputs_embeds` (ONNX Runtime GenAI build, exclude_embeds): rows of the token table for the prompt text, the tower's 104 final
// vectors (rounded to float16) in place of the audio-pad tokens. States are immutable ORT tensors, so several beams can continue from one parent.

import { pastNameFor, cacheDims, inputMeta, halfToNumber } from "../backends/common.js";
import { blockPlan } from "../backends/genai.js";
import type { EmbeddingTable } from "../backends/embeddings.js";
import { decodeIds, parseAsrText, type DecoderTokens } from "./decoderText.js";
import type { DecoderSpec } from "./types.js";

// ---- float16 ----------------------------------------------------------------------------------------------------------------------

const F32 = new Float32Array(1), U32 = new Uint32Array(F32.buffer);
/** float32 -> float16 bits, round to nearest even (what torch's .half() does). */
export function f32ToF16Bits(x: number): number {
  F32[0] = x;
  const b = U32[0], sign = (b >>> 16) & 0x8000, e = (b >>> 23) & 0xff, m = b & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  let ne = e - 127 + 15;
  if (ne >= 0x1f) return sign | 0x7c00;
  if (ne <= 0) {
    if (ne < -10) return sign;
    const sh = 14 - ne, full = m | 0x800000, mm = full >>> sh, rem = full & ((1 << sh) - 1), half = 1 << (sh - 1);
    let r = mm; if (rem > half || (rem === half && (r & 1))) r++;
    return sign | r;
  }
  let r = (ne << 10) | (m >>> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (r & 1))) r++;
  return sign | r;
}

let HALF_TABLE: Float32Array | null = null;
function halfTable(): Float32Array {
  if (HALF_TABLE) return HALF_TABLE;
  HALF_TABLE = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) HALF_TABLE[i] = halfToNumber(i);
  return HALF_TABLE;
}

/** The last row of a [1, S, V] logits tensor as float32. */
async function lastRow(L: any): Promise<Float32Array> {
  const V = L.dims.at(-1), S = L.dims.at(-2);
  const d = typeof L.getData === "function" ? await L.getData(true) : L.data;
  const off = (S - 1) * V, out = new Float32Array(V);
  if (d instanceof Float32Array) { out.set(d.subarray(off, off + V)); return out; }
  if (L.type === "float16" || d instanceof Uint16Array) { const t = halfTable(); for (let i = 0; i < V; i++) out[i] = t[(d as any)[off + i]]; return out; }
  for (let i = 0; i < V; i++) out[i] = Number((d as any)[off + i]);   // Float16Array
  return out;
}

// ---- the ORT graph -------------------------------------------------------------------------------------------------------------------

export interface DecoderKv { tensors: Record<string, any>; past: number }

/** What the engine needs from a decoder runtime; tests can supply a fake. */
export interface DecoderBackend {
  hidden: number;
  vocab: number;
  empty(): DecoderKv;
  /** Run `rows` (float16 bits, [S * hidden]) on top of `kv` (kept alive); returns the new state and, if asked, the last row of logits. */
  feed(rows: Uint16Array, kv: DecoderKv, wantLogits: boolean): Promise<{ kv: DecoderKv; logits?: Float32Array }>;
  dispose(kv: DecoderKv): void;
  close(): Promise<void>;
}

export interface OrtDecoderOptions { hidden: number; vocab: number; kvHeadDim: number }

export function ortDecoder(session: any, ort: { Tensor: new (type: any, data: any, dims: number[]) => any }, o: OrtDecoderOptions): DecoderBackend {
  const T = ort.Tensor;
  const inNames: string[] = session.inputNames, outNames: string[] = session.outputNames;
  if (!inNames.includes("inputs_embeds")) throw new Error("decoder: the graph has no input inputs_embeds (it must be a build with exclude_embeds)");
  if (inNames.includes("position_ids")) throw new Error("decoder: graphs with position_ids are not supported (the build derives positions from attention_mask)");
  if (!outNames.includes("logits")) throw new Error("decoder: the graph has no logits output");
  const meta = inputMeta(session);
  const embType = meta.get("inputs_embeds")?.type ?? "float16";
  if (embType !== "float16" && embType !== "float32") throw new Error(`decoder: unsupported inputs_embeds type ${embType}`);
  const specs: { name: string; type: string; dims: number[] }[] = [];
  for (const n of inNames) {
    if (n === "inputs_embeds" || n === "attention_mask") continue;
    const m = meta.get(n);
    if (!m?.shape) throw new Error("decoder: no shape metadata for input " + n);
    const dims = cacheDims(n, m.shape, 0);
    if (typeof m.shape[3] !== "number") dims[3] = o.kvHeadDim;
    specs.push({ name: n, type: m.type, dims });
  }
  const pastOf = new Map<string, string>();
  for (const out of outNames) { const p = pastNameFor(out); if (p && inNames.includes(p)) pastOf.set(out, p); }
  if (specs.some((s) => ![...pastOf.values()].includes(s.name))) throw new Error("decoder: cache inputs without a present output");
  const zeros = (s: { type: string; dims: number[] }) => {
    const n = s.dims.reduce((a, b) => a * b, 1);
    return s.type === "float16" ? new T("float16", new Uint16Array(n), s.dims) : new T("float32", new Float32Array(n), s.dims);
  };
  const freeT = (t: any) => { if (t?.location === "gpu-buffer") { try { t.dispose?.(); } catch { /* already freed */ } } };
  return {
    hidden: o.hidden, vocab: o.vocab,
    empty() { const t: Record<string, any> = {}; for (const s of specs) t[s.name] = zeros(s); return { tensors: t, past: 0 }; },
    async feed(rows, kv, wantLogits) {
      const S = rows.length / o.hidden;
      if (!Number.isInteger(S) || S < 1) throw new Error("decoder: rows must be a whole number of hidden-size rows");
      const feed: Record<string, any> = { ...kv.tensors };
      if (embType === "float16") feed.inputs_embeds = new T("float16", rows, [1, S, o.hidden]);
      else { const t = halfTable(), f = new Float32Array(rows.length); for (let i = 0; i < f.length; i++) f[i] = t[rows[i]]; feed.inputs_embeds = new T("float32", f, [1, S, o.hidden]); }
      feed.attention_mask = new T("int64", new BigInt64Array(kv.past + S).fill(1n), [1, kv.past + S]);
      const fetches = [...pastOf.keys()];
      if (wantLogits) fetches.unshift("logits");
      const r = await session.run(feed, fetches, { logSeverityLevel: 3 });
      const tensors: Record<string, any> = {};
      for (const [out, p] of pastOf) tensors[p] = r[out];
      let logits: Float32Array | undefined;
      if (wantLogits) { logits = await lastRow(r.logits); freeT(r.logits); }
      return { kv: { tensors, past: kv.past + S }, logits };
    },
    dispose(kv) { for (const t of Object.values(kv.tensors)) freeT(t); },
    async close() { await session.release?.(); },
  };
}

// ---- engine --------------------------------------------------------------------------------------------------------------------------

const NEG = -1e9;

/** log_softmax over a float32 row, in float64. */
function logSoftmax(x: Float32Array): Float64Array {
  let mx = -Infinity;
  for (let i = 0; i < x.length; i++) if (x[i] > mx) mx = x[i];
  let s = 0;
  for (let i = 0; i < x.length; i++) s += Math.exp(x[i] - mx);
  const lse = mx + Math.log(s), out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] - lse;
  return out;
}

function argmaxFirst(x: Float32Array): number {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < x.length; i++) if (x[i] > bv) { bv = x[i]; bi = i; }
  return bi;
}

/** The k largest of `lp + base`, sorted by value (descending), then index (ascending). */
function topK(lp: Float64Array, base: number, k: number): { tok: number; v: number }[] {
  const best: { tok: number; v: number }[] = [];
  let floor = -Infinity;
  for (let i = 0; i < lp.length; i++) {
    const v = lp[i] + base;
    if (best.length === k && v <= floor) continue;
    let j = best.length;
    while (j > 0 && best[j - 1].v < v) j--;
    best.splice(j, 0, { tok: i, v });
    if (best.length > k) best.pop();
    if (best.length === k) floor = best[k - 1].v;
  }
  return best;
}

export interface DecoderEngineOptions { chunk?: number; minBlock?: number; prefixRows?: number }

export class DecoderEngine {
  private chunk: number; private minBlock: number; readonly prefixRows: number;
  constructor(readonly be: DecoderBackend, private table: EmbeddingTable, readonly spec: DecoderSpec, private tokens: DecoderTokens, o: DecoderEngineOptions = {}) {
    this.chunk = o.chunk ?? 64; this.minBlock = o.minBlock ?? 32; this.prefixRows = o.prefixRows ?? 64;
    if (table.hidden !== be.hidden) throw new Error(`decoder: embedding table hidden ${table.hidden} != graph hidden ${be.hidden}`);
  }

  /** Rows of the 119-position prompt: token rows, the tower's vectors as float16, token rows. [119 * hidden] */
  async promptRows(audio: Float32Array): Promise<Uint16Array> {
    const H = this.be.hidden, p = this.spec.prompt;
    if (audio.length !== p.audioFrames * H) throw new Error(`decoder: ${audio.length} audio values, expected ${p.audioFrames} x ${H}`);
    const before = await this.table.rows(p.before), after = await this.table.rows(p.after);
    const rows = new Uint16Array((p.before.length + p.audioFrames + p.after.length) * H);
    rows.set(before, 0);
    const a0 = p.before.length * H;
    for (let i = 0; i < audio.length; i++) rows[a0 + i] = f32ToF16Bits(audio[i]);
    rows.set(after, a0 + audio.length);
    return rows;
  }

  async open(audio: Float32Array): Promise<DecoderWindow> {
    return new DecoderWindow(this, await this.promptRows(audio));
  }

  /** Feed rows in blocks of at least `minBlock`; logits of the last block's last row if asked. The input state is kept. */
  async prefill(rows: Uint16Array, kv: DecoderKv, wantLogits: boolean): Promise<{ kv: DecoderKv; logits?: Float32Array }> {
    const H = this.be.hidden, n = rows.length / H, plan = this.minBlock > 0 && n < this.minBlock ? [n] : blockPlan(n, this.chunk, this.minBlock);
    let cur = kv, p = 0, logits: Float32Array | undefined;
    for (let i = 0; i < plan.length; i++) {
      const last = i === plan.length - 1;
      const r = await this.be.feed(rows.subarray(p * H, (p + plan[i]) * H), cur, last && wantLogits);
      if (cur !== kv) this.be.dispose(cur);
      cur = r.kv; logits = r.logits; p += plan[i];
    }
    return { kv: cur, logits };
  }

  async tokenRow(id: number): Promise<Uint16Array> { return this.table.rows([id]); }

  decode(ids: number[]): string { return parseAsrText(decodeIds(ids, this.tokens)); }

  async close(): Promise<void> { try { await this.be.close(); } finally { await this.table.close?.(); } }
}

export interface LanguageScores { codes: string[]; scores: number[] }

/** One window of audio vectors in the decoder: the shared prefix is computed once and both questions continue from it. */
export class DecoderWindow {
  private prefix: DecoderKv | null = null;
  constructor(private e: DecoderEngine, private rows: Uint16Array) {}
  private get H() { return this.e.be.hidden; }

  /** Prompt state up to `prefixRows` positions (never changed afterwards). */
  private async prefixState(): Promise<DecoderKv> {
    if (!this.prefix) {
      const n = this.e.prefixRows;
      this.prefix = (await this.e.prefill(this.rows.subarray(0, n * this.H), this.e.be.empty(), false)).kv;
    }
    return this.prefix;
  }

  /** Sum of log-probabilities of each language's name tokens after "language" (marimba-audio `language_scores`). */
  async language(codes?: string[]): Promise<LanguageScores> {
    const spec = this.e.spec, all = Object.keys(spec.languages), use = codes ?? all;
    for (const c of use) if (!spec.languages[c]) throw new Error(`language: unsupported code ${JSON.stringify(c)} (supported: ${all.join(", ")})`);
    const n = this.e.prefixRows, H = this.H;
    const pre = await this.prefixState();
    const tail = new Uint16Array(this.rows.length - n * H + H);
    tail.set(this.rows.subarray(n * H));
    tail.set(await this.e.tokenRow(spec.prompt.languageToken), this.rows.length - n * H);
    const r1 = await this.e.prefill(tail, pre, true);
    const lp1 = logSoftmax(r1.logits!);
    const scores = new Array<number>(use.length);
    const second = new Map<number, Float64Array>();   // first token of a two-token name -> log-probabilities after it
    try {
      for (let i = 0; i < use.length; i++) {
        const ids = spec.languages[use[i]].ids;
        if (ids.length === 1) { scores[i] = lp1[ids[0]]; continue; }
        if (ids.length !== 2) throw new Error(`language ${use[i]}: names of ${ids.length} tokens are not supported`);
        let lp2 = second.get(ids[0]);
        if (!lp2) {
          const s = await this.e.be.feed(await this.e.tokenRow(ids[0]), r1.kv, true);
          this.e.be.dispose(s.kv);
          lp2 = logSoftmax(s.logits!); second.set(ids[0], lp2);
        }
        scores[i] = lp1[ids[0]] + lp2[ids[1]];
      }
    } finally { this.e.be.dispose(r1.kv); }
    return { codes: use, scores };
  }

  private async promptState(): Promise<{ kv: DecoderKv; logits: Float32Array }> {
    const n = this.e.prefixRows, H = this.H;
    const r = await this.e.prefill(this.rows.subarray(n * H), await this.prefixState(), true);
    return { kv: r.kv, logits: r.logits! };
  }

  /** Greedy decoding (num_beams=1): generated ids, EOS included when it was produced. */
  async greedy(): Promise<number[]> {
    const { eos, term } = this.e.spec, E = new Set(eos), out: number[] = [];
    let { kv, logits } = await this.promptState();
    try {
      for (;;) {
        const t = argmaxFirst(logits);
        out.push(t);
        if (E.has(t) || out.length >= term.maxNewTokens) break;
        const r = await this.e.be.feed(await this.e.tokenRow(t), kv, true);
        this.e.be.dispose(kv); kv = r.kv; logits = r.logits!;
      }
    } finally { this.e.be.dispose(kv); }
    return out;
  }

  /** Beam search, num_beams = beams, returning the `beams` best finished sequences, best first (Hugging Face `_beam_search` semantics). */
  async beam(beams = this.e.spec.term.beams): Promise<number[][]> {
    const { eos, term } = this.e.spec, E = new Set(eos), maxNew = term.maxNewTokens, lpen = term.lengthPenalty, V = this.e.be.vocab;
    const keep = Math.max(2, 1 + eos.length) * beams;
    type Run = { kv: DecoderKv; logits: Float32Array; seq: number[]; score: number };
    type Done = { seq: number[]; score: number; finished: boolean };
    const first = await this.promptState();
    let running: Run[] = Array.from({ length: beams }, (_, i) => ({ kv: first.kv, logits: first.logits, seq: [], score: i === 0 ? 0 : NEG }));
    let done: Done[] = Array.from({ length: beams }, () => ({ seq: [], score: NEG, finished: false }));
    let unsatisfied = true, gen = 0;      // gen = generated tokens so far (cur_len - prompt length)
    const dispose = (rs: Run[]) => { for (const kv of new Set(rs.map((r) => r.kv))) this.e.be.dispose(kv); };
    try {
      for (;;) {
        // top `keep` continuations of the accumulated log-probabilities over (beam, token)
        const cand: { b: number; tok: number; v: number }[] = [];
        const lps = new Map<Float32Array, Float64Array>();
        running.forEach((r, b) => {
          if (r.score <= NEG / 2 && b > 0 && gen === 0) return;      // the three copies of the first beam start at -1e9: never selected while beam 0 has enough tokens
          let lp = lps.get(r.logits); if (!lp) { lp = logSoftmax(r.logits); lps.set(r.logits, lp); }
          for (const c of topK(lp, r.score, keep)) cand.push({ b, tok: c.tok, v: c.v });
        });
        cand.sort((x, y) => y.v - x.v || (x.b * V + x.tok) - (y.b * V + y.tok));
        const top = cand.slice(0, keep);
        const hit = top.map((c) => E.has(c.tok) || gen + 1 >= maxNew);
        // next running beams: the best `beams` after pushing the stopped ones down
        const order = top.map((c, j) => ({ j, v: c.v + (hit[j] ? NEG : 0) })).sort((x, y) => y.v - x.v || x.j - y.j).slice(0, beams);
        const next = order.map(({ j, v }) => ({ parent: top[j].b, tok: top[j].tok, score: v, seq: [...running[top[j].b].seq, top[j].tok] }));
        // finished list: the first `beams` candidates that stopped, scored by sum log-prob / length^penalty
        const entries: Done[] = top.map((c, j) => {
          const fin = hit[j] && j < beams && unsatisfied;
          return { seq: [...running[c.b].seq, c.tok], score: fin ? c.v / Math.pow(gen + 1, lpen) : NEG, finished: fin };
        });
        const merged = [...done, ...entries].map((d, i) => ({ d, i })).sort((x, y) => y.d.score - x.d.score || x.i - y.i).slice(0, beams).map((x) => x.d);
        done = merged;
        gen += 1;
        // early-stop heuristic (early_stopping=False): best running score at the current length against the worst finished one
        const best = next[0].score / Math.pow(gen, lpen);
        const worst = Math.min(...done.map((d) => d.score));
        const improve = done.some((d) => best > (d.finished ? worst : NEG));
        unsatisfied = unsatisfied && improve;
        const cont = unsatisfied && !hit.every(Boolean);
        if (!cont) { dispose(running); break; }
        // forward of the surviving beams (one token each, from their parents' states)
        const nr: Run[] = [];
        for (const n of next) {
          const p = running[n.parent];
          const r = await this.e.be.feed(await this.e.tokenRow(n.tok), p.kv, true);
          nr.push({ kv: r.kv, logits: r.logits!, seq: n.seq, score: n.score });
        }
        dispose(running);
        running = nr;
      }
    } catch (e) { dispose(running); throw e; }
    return done.slice(0, beams).map((d) => d.seq);
  }

  /** The five hypotheses of marimba-audio `hypotheses`: greedy first, then the four beams, as parsed transcripts. */
  async hypotheses(): Promise<string[]> {
    const g = await this.greedy(), b = await this.beam();
    return [g, ...b].map((ids) => this.e.decode(ids));
  }

  async close(): Promise<void> { if (this.prefix) { this.e.be.dispose(this.prefix); this.prefix = null; } }
}
