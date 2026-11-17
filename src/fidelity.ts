// Fidelity harness. Pure computation, no network: compare per-build decisions with reference
// outputs (decision changed %), and measure ORDER INVERSION (reverse the option order of choice
// questions; a flip is a different decided label). A thin runner takes an injected decide().

import type { DecisionsQuestion, DecisionInput } from "./questions.js";
import { decisionOf, type Answer } from "./readout.js";

/** One line of the question set JSONL. */
export interface QuestionCase { id: string; input: DecisionInput; question: DecisionsQuestion }
/** One line of an outputs JSONL (reference or build). `decision`: "yes"/"no", the choice value, or the level label. */
export interface OutputRecord { id: string; type?: string; decision: string; probability?: number; method?: string }

export function parseJsonl<T = any>(text: string): T[] {
  return text.split(/\r?\n/).map((l, i) => [l.trim(), i] as const).filter(([l]) => l && !l.startsWith("//")).map(([l, i]) => {
    try { return JSON.parse(l) as T; } catch (e: any) { throw new Error(`jsonl line ${i + 1}: ${e?.message ?? e}`); }
  });
}

export function toJsonl(rows: unknown[]): string { return rows.map((r) => JSON.stringify(r)).join("\n") + "\n"; }

export function recordOf(id: string, a: Answer): OutputRecord {
  return { id, type: a.type, decision: decisionOf(a as any), method: a.method, ...(a.type === "predicate" ? { probability: a.probability } : {}) };
}

function index(rows: OutputRecord[], what: string): Map<string, OutputRecord> {
  const m = new Map<string, OutputRecord>();
  for (const r of rows) {
    if (!r || typeof r.id !== "string" || typeof r.decision !== "string") throw new Error(`${what}: each record needs string id and decision`);
    if (m.has(r.id)) throw new Error(`${what}: duplicate id ${r.id}`);
    m.set(r.id, r);
  }
  return m;
}

export interface Agreement {
  n: number;
  same: number;
  changed: number;
  /** changed / n, in percent. */
  changedPct: number;
  /** 95% Wilson interval of changedPct. */
  ci95: [number, number];
  changedIds: string[];
  /** Ids in the reference with no build output (not counted in n). */
  missing: string[];
  byType: Record<string, { n: number; changed: number; changedPct: number }>;
}

export function wilson(k: number, n: number, z = 1.959964): [number, number] {
  if (!n) return [0, 0];
  const p = k / n, d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d, h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h) * 100, Math.min(1, c + h) * 100];
}

/** Decision agreement of a build with the reference, over the ids present in both. */
export function agreement(reference: OutputRecord[], build: OutputRecord[]): Agreement {
  const ref = index(reference, "reference"), got = index(build, "build");
  const changedIds: string[] = [], missing: string[] = [];
  const byType: Agreement["byType"] = {};
  let n = 0;
  for (const [id, r] of ref) {
    const b = got.get(id);
    if (!b) { missing.push(id); continue; }
    n++;
    const t = r.type ?? b.type ?? "?";
    const bt = (byType[t] ??= { n: 0, changed: 0, changedPct: 0 });
    bt.n++;
    if (b.decision !== r.decision) { changedIds.push(id); bt.changed++; }
  }
  for (const v of Object.values(byType)) v.changedPct = v.n ? (100 * v.changed) / v.n : 0;
  const changed = changedIds.length;
  return { n, same: n - changed, changed, changedPct: n ? (100 * changed) / n : 0, ci95: wilson(changed, n), changedIds, missing, byType };
}

/** Per-build agreement table. */
export function fidelityReport(reference: OutputRecord[], builds: Record<string, OutputRecord[]>): Record<string, Agreement> {
  return Object.fromEntries(Object.entries(builds).map(([k, v]) => [k, agreement(reference, v)]));
}

/** Reverse the option order of a choice question; null when the question has no option order (predicate, score). */
export function invertQuestion(q: DecisionsQuestion): DecisionsQuestion | null {
  if (q.type !== "choice") return null;
  if (q.choices) return { ...q, choices: [...q.choices].reverse() };
  if (q.options) return { ...q, options: [...q.options].reverse() };
  return null;
}

export interface OrderInversion {
  /** Choice questions measured. */
  n: number;
  flips: number;
  flipPct: number;
  ci95: [number, number];
  flippedIds: string[];
  /** Questions without an option order (predicate/score), not measured. */
  skipped: number;
  /** flipPct <= maxFlipPct. */
  pass: boolean;
  maxFlipPct: number;
}

/** Flips between decisions on the original and on the reversed option order (decision = label, order-free). */
export function orderInversion(original: OutputRecord[], inverted: OutputRecord[], opts: { maxFlipPct?: number; skipped?: number } = {}): OrderInversion {
  const a = index(original, "original"), b = index(inverted, "inverted");
  const flippedIds: string[] = [];
  let n = 0;
  for (const [id, r] of a) {
    const s = b.get(id);
    if (!s) continue;
    n++;
    if (s.decision !== r.decision) flippedIds.push(id);
  }
  const maxFlipPct = opts.maxFlipPct ?? 5;
  const flipPct = n ? (100 * flippedIds.length) / n : 0;
  return { n, flips: flippedIds.length, flipPct, ci95: wilson(flippedIds.length, n), flippedIds, skipped: opts.skipped ?? 0, pass: n > 0 && flipPct <= maxFlipPct, maxFlipPct };
}

export type DecideFn = (req: { input: DecisionInput; questions: DecisionsQuestion[] }) => Promise<{ answers: Answer[]; dispose?: () => unknown }>;

/** Run a question set through decide() (one question per call), returning output records. */
export async function runCases(cases: QuestionCase[], decideFn: DecideFn, onProgress?: (i: number, n: number) => void): Promise<OutputRecord[]> {
  const out: OutputRecord[] = [];
  for (const [i, c] of cases.entries()) {
    const s = await decideFn({ input: c.input, questions: [{ ...c.question, name: c.id }] });
    try { out.push(recordOf(c.id, s.answers[0])); } finally { await s.dispose?.(); }
    onProgress?.(i + 1, cases.length);
  }
  return out;
}

/** Order-inversion run: original order and reversed order, then the flip statistics. */
export async function runOrderInversion(cases: QuestionCase[], decideFn: DecideFn, opts: { maxFlipPct?: number } = {}): Promise<OrderInversion & { original: OutputRecord[]; inverted: OutputRecord[] }> {
  const inv = cases.map((c) => ({ c, q: invertQuestion(c.question) })).filter((x) => x.q);
  const original = await runCases(inv.map((x) => x.c), decideFn);
  const inverted = await runCases(inv.map((x) => ({ ...x.c, question: x.q! })), decideFn);
  return { ...orderInversion(original, inverted, { ...opts, skipped: cases.length - inv.length }), original, inverted };
}

/** Manifest `fidelity` slot from measured numbers (fractions 0..1, as stored by the lab). */
export function fidelitySlot(reference: string, a: Partial<Record<"logits" | "head", Agreement>>, inv: Partial<Record<"logits" | "head", OrderInversion>> = {}) {
  const ns = [...Object.values(a), ...Object.values(inv)].map((x) => x!.n);
  const pick = <T,>(o: Partial<Record<string, T>>, f: (x: T) => number) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, f(v as T)]));
  return {
    reference, n: Math.max(0, ...ns),
    decisionChanged: pick(a, (x: Agreement) => x.changedPct / 100),
    orderInversion: pick(inv, (x: OrderInversion) => x.flipPct / 100),
    measuredAt: new Date().toISOString().slice(0, 10),
  };
}
