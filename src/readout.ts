// Readouts. "head": bilinear pointer head over hidden states (ported from ct0-edge systemone.ts;
// the hidden state is the graph's post-final-norm output, no norm is applied here, exactly as
// ct0-edge). "logits": next-token logits at the last prompt position mapped to labels.

import type { QType } from "./manifest.js";

export interface HeadWeights {
  q: Float32Array; qb: Float32Array; k: Float32Array; kb: Float32Array;
  headDim: number; hiddenSize: number; temperature: number; scoreOffset: number;
  /** Factor on the dot product (spec 6.1). 1 when 1/sqrt(headDim) is folded into q/qb. In-memory heads without it: 1.
   *  loadHead always sets it (header, manifest, or the 1/sqrt(headDim) default of alias3-web). */
  scale?: number;
}

function f16ToF32(h: number): number {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** Reader scale of a head (spec 6.1), alias3-web semantics (aliasrobotics/alias3-web@2ab2e60 `aliasweb.mjs`:
 *  `scale: meta.scale ?? 1 / Math.sqrt(d)`): header `scale`, else 1 when the header carries the marker
 *  `scale_folded_into_q` (the factor was multiplied into q/qb before writing), else the manifest's `heads.<name>.scale`,
 *  else 1/sqrt(headDim). Header and manifest must agree when both state it. A head written with the factor folded and no
 *  key at all (ct0-edge heads) MUST be declared `scale: 1` in the manifest, or it is read 1/sqrt(headDim) too small. */
export function headScale(meta: any, declared?: number): number {
  const pos = (x: any) => typeof x === "number" && Number.isFinite(x) && x > 0;
  if (meta.scale !== undefined && !pos(meta.scale)) throw new Error(`head: header scale ${JSON.stringify(meta.scale)} is not a positive number`);
  const fromHeader = meta.scale !== undefined ? meta.scale : meta.scale_folded_into_q !== undefined ? 1 : undefined;
  if (declared !== undefined && !pos(declared)) throw new Error(`head: manifest scale ${JSON.stringify(declared)} is not a positive number`);
  if (fromHeader !== undefined && declared !== undefined && Math.abs(fromHeader - declared) > 1e-12 * Math.max(1, declared))
    throw new Error(`head: header scale ${fromHeader} != manifest scale ${declared}`);
  return fromHeader ?? declared ?? 1 / Math.sqrt(meta.headDim);
}

/** Where a head's scale came from (spec 6.1); "default" = neither the header nor the manifest states it. */
export function headScaleSource(meta: any, declared?: number): "header" | "folded-marker" | "manifest" | "default" {
  return meta.scale !== undefined ? "header" : meta.scale_folded_into_q !== undefined ? "folded-marker" : declared !== undefined ? "manifest" : "default";
}

/** head.bin: u32 LE json length, json meta {headDim, hiddenSize, temperature, score_offset?, scale? | scale_folded_into_q?},
 *  fp16 q, qb, k, kb. `declaredScale`: the manifest head's `scale` (see headScale). */
export function loadHead(buf: ArrayBuffer, declaredScale?: number): HeadWeights {
  const dv = new DataView(buf);
  const jsonLen = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen)));
  if (!Number.isInteger(meta.headDim) || meta.headDim < 1 || !Number.isInteger(meta.hiddenSize) || !Number.isFinite(meta.temperature))
    throw new Error("head: bad metadata " + JSON.stringify(meta));
  const scale = headScale(meta, declaredScale);
  let off = 4 + jsonLen;
  const take = (n: number) => {
    const avail = Math.floor((buf.byteLength - off) / 2);
    if (n > avail) throw new Error(`head binary too short: needs ${n} fp16, ${avail} available`);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = f16ToF32(dv.getUint16(off + i * 2, true));
    off += n * 2;
    return out;
  };
  const hd: HeadWeights = {
    q: take(meta.headDim * meta.hiddenSize), qb: take(meta.headDim),
    k: take(meta.headDim * meta.hiddenSize), kb: take(meta.headDim),
    headDim: meta.headDim, hiddenSize: meta.hiddenSize, temperature: meta.temperature, scoreOffset: meta.score_offset ?? 0, scale,
  };
  if (off !== buf.byteLength) throw new Error(`head binary has ${buf.byteLength - off} trailing bytes`);
  return hd;
}

function proj(W: Float32Array, b: Float32Array, x: Float32Array, dim: number, hidden: number): Float32Array {
  const y = new Float32Array(dim);
  for (let d = 0; d < dim; d++) {
    let acc = b[d];
    const row = d * hidden;
    for (let i = 0; i < hidden; i++) acc += W[row + i] * x[i];
    y[d] = acc;
  }
  return y;
}

/** scale * (Wq hq + bq) . (Wk hk + bk) / T + offset. */
export function headScore(hd: HeadWeights, hq: Float32Array, hk: Float32Array): number {
  if (hq.length !== hd.hiddenSize || hk.length !== hd.hiddenSize)
    throw new Error(`head: hidden size ${hq.length}/${hk.length} != head hiddenSize ${hd.hiddenSize}`);
  const a = proj(hd.q, hd.qb, hq, hd.headDim, hd.hiddenSize), b = proj(hd.k, hd.kb, hk, hd.headDim, hd.hiddenSize);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return (s * (hd.scale ?? 1)) / hd.temperature + (hd.scoreOffset || 0);
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
export function softmax(xs: number[]): number[] {
  const finite = xs.filter(Number.isFinite);
  if (!finite.length) throw new Error("softmax: no finite value");
  const mx = Math.max(...finite);
  const e = xs.map((x) => (Number.isFinite(x) ? Math.exp(x - mx) : 0));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}
export function argmax(xs: number[]): number {
  let b = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[b]) b = i;
  return b;
}

// ---- Decisions-API answers -------------------------------------------------

/** Which path produced the answer: "s1" = the one-pass read; "escalated" = a margin policy acted on it (see `escalation`). */
export type AnswerPath = "s1" | "escalated";

/** Common fields of every answer. */
interface AnswerBase {
  name: string;
  method: "logits" | "head";
  /**
   * Margin of the decision: probability of the top label minus that of the runner-up (predicate: |2p - 1|).
   * Unit-free, in [0, 1]; the definition used by the escalation tests ("margin = p(1st) - p(2nd)").
   * Score via logits: top-1 minus top-2 level probability (the decided level is the rounded expectation, which can
   * differ from the arg-max level). null for the score head: one scalar, no runner-up.
   */
  margin: number | null;
  /**
   * Same gap in the readout's raw score space (logit gap, or head score gap; predicate |score(yes) - score(no)|).
   * Does not saturate when p does (an exploratory probe on development data ranked by this). Units differ per readout and
   * model: compare only within one (model, build, readout, question type). null for the score head.
   */
  rawMargin: number | null;
  path: AnswerPath;
  /** Present when a margin policy acted on this answer (see src/margin.ts). */
  escalation?: EscalationRecord;
}

/** What a margin policy did to one answer. */
export interface EscalationRecord {
  reason: "margin";
  /** Which field the threshold applied to. */
  metric: "margin" | "rawMargin";
  /** Value of `metric` on the System-1 answer. */
  margin: number;
  threshold: number;
  /** True while the threshold is not backed by a measured result. */
  provisional: boolean;
  /** Where the threshold came from (manifest note, or "call"). */
  source: string;
  to: "build" | "generate";
  /** to "build": the build that answered; `from` is the original System-1 answer it replaced. */
  build?: string;
  from?: Omit<Answer, "escalation">;
  /** to "generate": System-2 text. The typed decision is NOT changed by it. */
  text?: string;
  /** to "generate": true when generation continued from the kept decision prefill; false = re-prefilled. */
  sharedPrefill?: boolean;
  reusedTokens?: number;
  finishReason?: "stop" | "length" | "abort";
}

export interface PredicateAnswer extends AnswerBase { type: "predicate"; probability: number; confidence: number; raw?: number; logits?: Omit<PredicateAnswer, "logits"> }
export interface ChoiceAnswer extends AnswerBase { type: "choice"; choice: string; probabilities: { value: string; probability: number }[]; confidence: number; raw?: number[]; logits?: Omit<ChoiceAnswer, "logits"> }
export interface ScoreAnswer extends AnswerBase {
  type: "score";
  /** 0-based level index; logits: probability-weighted average; head: sigmoid(raw) * (levels - 1). */
  score: number;
  /** Level label nearest to `score`. */
  level: string;
  /** Per-level distribution; null for the head read (the score head gives a scalar, not a distribution). */
  probabilities: { value: number; label: string; probability: number }[] | null;
  /** Highest level probability; null for the head read. */
  confidence: number | null;
  raw?: number | number[]; logits?: Omit<ScoreAnswer, "logits">;
}
export type Answer = PredicateAnswer | ChoiceAnswer | ScoreAnswer;

/** Top-1 minus top-2 of finite scores (null with fewer than two). */
export function topGap(xs: number[]): number | null {
  const f = xs.filter(Number.isFinite).sort((a, b) => b - a);
  return f.length < 2 ? null : f[0] - f[1];
}

/** `raw` = score(yes) - score(no) when known; otherwise rawMargin is |logit(p)| (the same quantity for a sigmoid read). */
export function predicateFromP(name: string, p: number, method: "logits" | "head", raw?: number): PredicateAnswer {
  const rm = raw !== undefined ? Math.abs(raw) : Math.abs(Math.log(p) - Math.log1p(-p));
  return { name, type: "predicate", probability: p, confidence: Math.max(p, 1 - p), method, margin: Math.abs(2 * p - 1),
    rawMargin: Number.isFinite(rm) ? rm : null, path: "s1", ...(raw !== undefined ? { raw } : {}) };
}

export function choiceFromScores(name: string, values: string[], scores: number[], method: "logits" | "head"): ChoiceAnswer {
  const probs = softmax(scores);
  const b = argmax(probs);
  return { name, type: "choice", choice: values[b], probabilities: values.map((value, i) => ({ value, probability: probs[i] })), confidence: probs[b], method, margin: topGap(probs), rawMargin: topGap(scores), path: "s1", raw: scores };
}

export function scoreFromLogits(name: string, labels: string[], scores: number[], method: "logits" | "head" = "logits"): ScoreAnswer {
  const probs = softmax(scores);
  const score = probs.reduce((a, p, i) => a + p * i, 0);
  return { name, type: "score", score, level: labels[Math.min(labels.length - 1, Math.max(0, Math.round(score)))],
    probabilities: labels.map((label, i) => ({ value: i, label, probability: probs[i] })), confidence: Math.max(...probs), method,
    margin: topGap(probs), rawMargin: topGap(scores), path: "s1", raw: scores };
}

export function scoreFromHead(name: string, labels: string[], raw: number): ScoreAnswer {
  const score = sigmoid(raw) * (labels.length - 1);
  return { name, type: "score", score, level: labels[Math.round(score)], probabilities: null, confidence: null, method: "head", margin: null, rawMargin: null, path: "s1", raw };
}

// ---- Evidence selector ----------------------------------------------------

/** Evidence selector output (decide() with `evidence`): which of the passage's sentences the evidence head points to. */
export interface EvidenceAnswer {
  method: "head";
  /** 0-based index of the chosen sentence (argmax; ties go to the first sentence). */
  index: number;
  /** The chosen sentence, as shown in the prompt (whitespace collapsed). */
  sentence: string;
  /** Head probability of the chosen sentence (softmax over the sentences). */
  probability: number;
  /** Per-sentence probabilities, in passage order. */
  probabilities: number[];
  /** The sentences as shown in the prompt, in order. */
  sentences: string[];
  /** p(1st) - p(2nd), same definition as Answer.margin. */
  margin: number | null;
  /** Head score gap between the top two sentences. */
  rawMargin: number | null;
  /** Head scores (pre-softmax), in passage order. */
  raw: number[];
}

export function evidenceFromScores(sentences: string[], scores: number[]): EvidenceAnswer {
  if (scores.length !== sentences.length) throw new Error(`evidence: ${scores.length} scores for ${sentences.length} sentences`);
  if (!scores.every(Number.isFinite)) throw new Error("evidence: head score is not finite");
  const probs = softmax(scores);
  const b = argmax(probs);
  return { method: "head", index: b, sentence: sentences[b], probability: probs[b], probabilities: probs, sentences,
    margin: topGap(probs), rawMargin: topGap(scores), raw: scores };
}

/** Decision label of an answer (used by the fidelity harness and the System-2 continuation). */
export function decisionOf(a: Pick<Answer, "type"> & Partial<{ probability: number; choice: string; level: string }>): string {
  if (a.type === "predicate") return (a.probability as number) >= 0.5 ? "yes" : "no";
  if (a.type === "choice") return a.choice as string;
  return a.level as string;
}

export type { QType };
