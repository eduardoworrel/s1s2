// System 1 (decide) and System 2 (generate) over one Backend.
// decide(): one forward pass per question; questions on the same input share the prefill of
// the common prefix (snapshot + prefillFrom, as ct0-edge askMany). With a state-keeping backend
// each question's full prefill is KEPT, and session.generate() continues from it.

import type { Manifest, HeadRole, Method } from "./manifest.js";
import { canShareState, type Backend, type GenerateOptions, type PrefillResult, type StateHandle } from "./backend.js";
import { encodeQuestion, firstTokens, fill, scoreLayout, letterLayout, appendLead, type Encoded } from "./encoding.js";
import { normalizeQuestions, inputToText, questionEntries, splitSentences, type DecisionRequest, type DecisionsQuestion, type EvidenceRequest, type NQ } from "./questions.js";
import { mergePolicy, manifestPolicy, shouldEscalate, type MarginPolicy } from "./margin.js";
import { escalationProblems, type EscalationSpec } from "./manifest.js";
import {
  headScore, sigmoid, predicateFromP, choiceFromScores, scoreFromLogits, scoreFromHead, decisionOf, evidenceFromScores,
  type Answer, type HeadWeights, type EscalationRecord, type EvidenceAnswer,
} from "./readout.js";
import type { Tok } from "./tok.js";
import { forecastPositions, type ForecastReader, type ForecastResult } from "./forecast.js";
import { applyNorm, type NormWeights } from "./heads/norm.js";
import { isLinear, isJoint, type AnyHead } from "./heads/index.js";
import { linearScores } from "./heads/linear.js";

export interface Engine {
  manifest: Manifest;
  tok: Tok;
  backend: Backend;
  heads: Partial<Record<HeadRole, AnyHead>>;
  maxSafeTokens: number;
  /** Out-of-graph final norm of this build (spec 6.5): applied to every hidden state before a head reads it. */
  norm?: NormWeights;
  /** Build this engine runs (its manifest `escalation` level applies). */
  build?: string;
  /** Forecast reader of this build (manifest `forecast` + the build's `forecastOutput`), when it can run. */
  forecast?: ForecastReader;
  /** Why `forecast` is absent ("forecast unsupported: ..."), for the error decide({ forecast: true }) throws. */
  forecastWhy?: string;
  /** Resolves another build of the manifest for `to: { build }` escalation. */
  resolveBuild?: (build: string) => Promise<EscalationTarget>;
}

/** Anything that can re-read questions: another S1S2 (another build), or a test double. */
export interface EscalationTarget {
  decide(req: DecisionRequest, opts?: DecideOptions): Promise<{ answers: Answer[]; dispose(): Promise<void> }>;
}

export interface DecideOptions {
  /** Also read the forecast layer of each question's prompt, in the same prefill, so session.forecast() can run (default
   *  false: S1 is unchanged). Needs the manifest's `forecast` and a build with `forecastOutput`. */
  forecast?: boolean;
  /** Readout for every question that does not set its own (default: manifest defaults.method per type). */
  method?: Method;
  /** Keep each question's prefill so session.generate() continues from it (default true). */
  keepState?: boolean;
  /**
   * Margin policy. Off by default. `true`: the manifest's policy (manifest < build < task). An object: merged on top
   * (`below` per type). `target`: the engine used for `to: { build }` instead of loading that build.
   * Thresholds are provisional until measured (see src/margin.ts); answers record which path answered.
   */
  escalate?: boolean | (EscalationSpec & { target?: EscalationTarget });
}

interface Plan { q: NQ; enc: Encoded; positions: number[]; read(r: PrefillResult): Answer }

/** Token ids the logits readout reads for question `q` (`ctx` = the rendered prompt), in the order of q.values. */
export function logitIds(e: Engine, q: NQ, ctx?: string, labels: string[] = q.values): number[] {
  const m = e.manifest;
  if (q.labelIds) {
    if (q.labelIds.length !== (q.type === "predicate" ? 2 : q.values.length)) throw new Error(`question ${q.name}: task ${q.task} pins ${q.labelIds.length} label tokens for ${q.values.length} labels`);
    return q.labelIds;
  }
  if (q.type === "predicate") return [m.logits.predicate.no.id, m.logits.predicate.yes.id];
  // Letters map: the option keys as the model writes them after the cue (`labels` from letterLayout: " A" or "A").
  if (q.type === "choice" && m.logits.choice.map === "letters") return firstTokens(e.tok, labels === q.values ? letterLayout(e.tok, q, ctx).labels : labels, "choice", ctx);
  // Task labels without pinned tokens: first token of each label after the prompt (whatever the manifest's global map, letters aside).
  if (q.task !== undefined && (q.type === "choice" || (q.type === "score" && m.tasks?.[q.task]?.levels))) return firstTokens(e.tok, q.type === "score" ? labels : q.values, q.type, ctx);
  if (q.type === "choice") {
    if (m.logits.choice.map === "tokens") {
      const t = m.logits.choice.tokens!;
      return q.values.map((v) => {
        const r = t.find((x) => x.label === v);
        if (!r) throw new Error(`question ${q.name}: choice ${JSON.stringify(v)} has no pinned logits token in this manifest (logits.choice.tokens: ${t.map((x) => JSON.stringify(x.label)).join(", ")}); use those labels or method "head"`);
        return r.id;
      });
    }
    return firstTokens(e.tok, q.values, "choice", ctx);
  }
  if (m.logits.score.map === "tokens") {
    const t = m.logits.score.tokens!;
    if (t.length !== q.values.length) throw new Error(`question ${q.name}: the logits score readout of this model reads ${t.length} level tokens (${t.map((x) => x.text).join(",")}); got ${q.values.length} levels`);
    return t.map((x) => x.id);
  }
  return firstTokens(e.tok, labels, "score", ctx);
}

export function planQuestion(e: Engine, q: NQ, state: string, method0?: Method): Plan {
  const m = e.manifest;
  const method: Method = q.method ?? method0 ?? m.defaults.method[q.type];
  let enc = encodeQuestion(e.tok, m, q, state);
  // Score by label logits: read the levels after the lead (the space the model writes after the cue), heads unmoved.
  // Choice by letters: read the keys as the model writes them after the cue (" A" with Qwen-style tokenizers).
  const lay = method === "head" ? { lead: "", labels: q.values }
    : q.type === "score" ? scoreLayout(e.tok, m, q, enc.text)
    : q.type === "choice" && !q.labelIds && m.logits.choice.map === "letters" ? letterLayout(e.tok, q, enc.text)
    : { lead: "", labels: q.values };
  enc = appendLead(e.tok, enc, lay.lead);
  if (enc.ids.length > e.maxSafeTokens) throw new Error(`question ${q.name}: prompt has ${enc.ids.length} tokens, over maxSafeTokens (${e.maxSafeTokens})`);

  const ids = method === "head" ? [] : logitIds(e, q, enc.text, lay.labels);
  const fromLogits = (L: Float32Array): Answer => {
    for (const id of ids) if (id >= L.length) throw new Error(`logits readout: token ${id} outside the vocabulary (${L.length})`);
    const T = m.logits.temperature?.[q.type] ?? 1;
    if (q.type === "predicate") { const raw = (L[ids[1]] - L[ids[0]]) / T; return predicateFromP(q.name, sigmoid(raw), "logits", raw); }
    const s = ids.map((i) => L[i] / T);
    return q.type === "choice" ? choiceFromScores(q.name, q.values, s, "logits") : scoreFromLogits(q.name, q.values, s);
  };
  if (method === "logits") return { q, enc, positions: [], read: (r) => fromLogits(r.lastLogits) };
  if (method !== "head" && method !== "both") throw new Error(`question ${q.name}: unknown method ${JSON.stringify(method)}`);

  if (!e.backend.hiddenName) throw new Error(`question ${q.name}: method "${method}" needs a hidden-state output, which this build does not have; use "logits"`);
  const anyHead = e.heads[q.type];
  if (!anyHead) throw new Error(`question ${q.name}: method "${method}" needs a head for ${q.type}, which this build/manifest does not provide`);
  const withLogits = (a: Answer, L: Float32Array): Answer => (method === "both" ? { ...a, logits: fromLogits(L) } as Answer : a);
  if (isJoint(anyHead)) throw new Error(`question ${q.name}: the ${q.type} head is a joint head (spec 6.4); it reads all questions of one prompt: use jointScores() / encodeJointSchema()`);
  if (isLinear(anyHead)) {
    // Linear readout (spec 6.3): rows of the question's type over the hidden state at one position.
    const n = q.type === "predicate" ? 2 : q.values.length;
    const pos = anyHead.read === "last" ? enc.last : enc.decide;
    return { q, enc, positions: [pos], read: ({ hiddens, lastLogits }) => {
      const z = linearScores(anyHead, hiddens[0], q.type, n);
      if (q.type === "predicate") { const c = choiceFromScores(q.name, ["no", "yes"], z, "head"); return withLogits(predicateFromP(q.name, c.probabilities[1].probability, "head", z[1] - z[0]), lastLogits); }
      if (q.type === "choice") return withLogits(choiceFromScores(q.name, q.values, z, "head"), lastLogits);
      return withLogits(scoreFromLogits(q.name, q.values, z, "head"), lastLogits);
    } };
  }
  const hd: HeadWeights = anyHead;
  const optionRead = q.type === "choice" || (q.type === "predicate" && m.readout.predicate === "options") || (q.type === "score" && m.readout.score === "option");
  if (optionRead) {
    if (enc.options.length < 2) throw new Error(`question ${q.name}: head readout found ${enc.options.length} option markers`);
    return { q, enc, positions: [enc.decide, ...enc.options], read: ({ hiddens, lastLogits }) => {
      const scores = hiddens.slice(1).map((h) => headScore(hd, hiddens[0], h));
      if (q.type === "choice") return withLogits(choiceFromScores(q.name, q.values, scores, "head"), lastLogits);
      if (q.type === "score") return withLogits(scoreFromLogits(q.name, q.values, scores, "head"), lastLogits); // levels as options (Kev)
      const c = choiceFromScores(q.name, ["no", "yes"], scores, "head");
      return withLogits(predicateFromP(q.name, c.probabilities[1].probability, "head", scores[1] - scores[0]), lastLogits);
    } };
  }
  return { q, enc, positions: [enc.decide, enc.last], read: ({ hiddens, lastLogits }) => {
    const raw = headScore(hd, hiddens[0], hiddens[1]);
    if (!Number.isFinite(raw)) throw new Error(`question ${q.name}: head score is not finite`);
    const a = q.type === "predicate" ? predicateFromP(q.name, sigmoid(raw), "head", raw) : scoreFromHead(q.name, q.values, raw);
    return withLogits(a, lastLogits);
  } };
}

interface EvidencePlan { enc: Encoded; positions: number[]; read(r: PrefillResult): EvidenceAnswer }

/** Whitespace collapsed, as the evidence layout shows each sentence on one option line. */
const oneLine = (s: string) => s.split(/\s+/).filter(Boolean).join(" ");

/**
 * Evidence selector plan: the manifest's choice layout with one option per sentence, read by the build's "evidence"
 * head (query at the decide marker, one key per option marker; softmax over the sentences; ties -> first sentence).
 */
export function planEvidence(e: Engine, ev: EvidenceRequest, state: string): EvidencePlan {
  const m = e.manifest;
  if (!ev || typeof ev !== "object") throw new Error("decide: evidence must be an object {requirement?, instructions?, sentences?}");
  if (!e.backend.hiddenName) throw new Error("evidence: needs a hidden-state output, which this build does not have");
  const hd0 = e.heads.evidence;
  if (!hd0) throw new Error('evidence: this build/manifest has no head of type "evidence"');
  if (isLinear(hd0) || isJoint(hd0)) throw new Error("evidence: the evidence selector is a pointer head (spec 6.7)");
  const hd: HeadWeights = hd0;
  let instructions: string;
  if (ev.instructions !== undefined) {
    if (typeof ev.instructions !== "string" || !ev.instructions.trim()) throw new Error("evidence.instructions: non-empty string");
    instructions = ev.instructions;
  } else {
    if (!m.evidence) throw new Error("evidence: the manifest has no evidence.instructions; pass evidence.instructions");
    if (m.evidence.instructions.includes("{requirement}") && (typeof ev.requirement !== "string" || !ev.requirement.trim()))
      throw new Error("evidence.requirement: required by the manifest's evidence.instructions ({requirement})");
    instructions = fill(m.evidence.instructions, { requirement: ev.requirement ?? "" }).text;
  }
  const src = ev.sentences ?? splitSentences(state);
  if (!Array.isArray(src)) throw new Error("evidence.sentences: string[]");
  const sentences = src.map((x, i) => {
    if (typeof x !== "string" || !oneLine(x)) throw new Error(`evidence.sentences[${i}]: empty or non-string sentence`);
    return oneLine(x);
  });
  if (sentences.length < 2) throw new Error(`evidence: needs at least 2 sentences, got ${sentences.length}`);
  const pre = m.evidence?.sentencePrefix ?? "";
  const q: NQ = { name: "evidence", type: "choice", instructions, values: sentences.map((_, i) => String(i)), texts: sentences.map((x) => pre + x) };
  const enc = encodeQuestion(e.tok, m, q, state);
  if (enc.ids.length > e.maxSafeTokens) throw new Error(`evidence: prompt has ${enc.ids.length} tokens, over maxSafeTokens (${e.maxSafeTokens})`);
  if (enc.options.length !== sentences.length) throw new Error(`evidence: found ${enc.options.length} option markers for ${sentences.length} sentences`);
  return { enc, positions: [enc.decide, ...enc.options], read: ({ hiddens }) =>
    evidenceFromScores(sentences, hiddens.slice(1).map((h) => headScore(hd, hiddens[0], h))) };
}

/** Longest common token prefix that stays before every read position and the last token of each plan. */
export function sharedPrefix(plans: { enc: Encoded; positions: number[] }[]): number {
  let k = Math.min(...plans.map((p) => p.enc.ids.length - 1), ...plans.flatMap((p) => p.positions));
  const a = plans[0].enc.ids;
  for (const p of plans.slice(1)) { let i = 0; while (i < k && p.enc.ids[i] === a[i]) i++; k = i; }
  return k;
}

/**
 * Shared-prefix length on a backend with a minimum block M (genai on WebGPU: 32): the snapshot must hold >= M tokens and
 * every continuation from it must feed >= M (2M when that prefill keeps a state, which stops M before its end), so the
 * reuse is bit-identical to a full prefill. `limits[i]` = the largest prefix plan i allows. Below M: 0 (no sharing).
 */
export function backOff(k: number, limits: number[], minBlock: number): number {
  if (!minBlock) return k;
  const kk = Math.min(k, ...limits);
  return kk >= minBlock ? kk : 0;
}

export interface GenerateResult {
  text: string;
  tokenIds: number[];
  /** True when generation continued from a kept prefill instead of prefilling again. */
  sharedPrefill: boolean;
  /** Prompt tokens served from the kept state (0 without shared prefill). */
  reusedTokens: number;
  finishReason: "stop" | "length" | "abort";
}

export interface SessionGenerateOptions extends GenerateOptions {
  /** Which question's prefill to continue from (default: the first). */
  question?: string;
  think?: boolean;
  onToken?: (text: string, id: number) => void;
}

async function collect(e: Engine, gen: AsyncGenerator<number>, max: number, signal: AbortSignal | undefined, onToken?: (t: string, id: number) => void) {
  const ids: number[] = [];
  let text = "";
  for await (const id of gen) {
    ids.push(id);
    const full = e.tok.decode(ids, { skipSpecial: true });
    const stable = full.endsWith("�") ? text : full;
    if (stable.length > text.length || !stable.startsWith(text)) { onToken?.(stable.slice(text.length), id); text = stable; }
  }
  text = e.tok.decode(ids, { skipSpecial: true });
  const finishReason: GenerateResult["finishReason"] = signal?.aborted ? "abort" : ids.length >= max ? "length" : "stop";
  return { text, tokenIds: ids, finishReason };
}

const thinkText = (m: Manifest, on: boolean | undefined) => (m.generation.think ? (on ? m.generation.think.on : m.generation.think.off) : "");

export class DecisionSession {
  readonly answers: Answer[];
  /** True when the decision prefill was kept and generate() continues from it. */
  readonly sharedPrefill: boolean;
  /** Evidence selector output when the request asked for `evidence`, else null. */
  evidence: EvidenceAnswer | null = null;
  private states: (StateHandle | undefined)[];
  private disposed = false;
  /** Forecast-layer rows of each question's prompt (set by decide when forecast was requested). */
  rows: (Float32Array[] | undefined)[] = [];
  constructor(private e: Engine, private plans: Plan[], answers: Answer[], states: (StateHandle | undefined)[]) {
    this.answers = answers; this.states = states;
    this.sharedPrefill = states.length > 0 && states.every(Boolean);
  }

  /** The forecast of a question's prompt (default: the first), read from the rows captured by the decide prefill:
   *  no backend call, no second prefill. Needs decide(req, { forecast: true }). */
  forecast(opts: { question?: string } = {}): ForecastResult {
    if (this.disposed) throw new Error("session disposed");
    const fx = this.e.forecast;
    if (!fx) throw new Error(this.e.forecastWhy ?? "forecast unsupported: this model has no forecast reader");
    if (!this.plans.length) throw new Error("session.forecast: this session has no questions (evidence only)");
    const i = opts.question === undefined ? 0 : this.plans.findIndex((p) => p.q.name === opts.question);
    if (i < 0) throw new Error(`session.forecast: no question named ${JSON.stringify(opts.question)}`);
    const rows = this.rows[i];
    if (!rows) throw new Error("session.forecast: this session did not read the forecast layer; call decide(req, { forecast: true })");
    return fx.read(rows, this.plans[i].q.name, this.plans[i].enc.ids.length);
  }

  answer(name: string): Answer | undefined { return this.answers.find((a) => a.name === name); }

  /** System 2 after question `i`'s decision prompt WITHOUT appending the decided answer (margin escalation):
   *  manifest generation.escalate, else generation.continue with an empty answer. Same kept-state rule as generate(). */
  async reconsider(i: number, prompt: string, opts: SessionGenerateOptions = {}): Promise<GenerateResult> {
    if (this.disposed) throw new Error("session disposed");
    const m = this.e.manifest, plan = this.plans[i];
    const tpl = m.generation.escalate ?? m.generation.continue;
    const cont = fill(tpl, { answer: "", prompt, think: thinkText(m, opts.think) }).text;
    return this.continueWith(i, [...plan.enc.ids, ...this.e.tok.encode(cont)], opts);
  }

  private async continueWith(i: number, ids: number[], opts: SessionGenerateOptions): Promise<GenerateResult> {
    const h = this.states[i], be = this.e.backend;
    const max = Math.max(0, Math.floor(opts.maxNewTokens ?? 512));
    if (h && be.generateFrom) {
      const r = await collect(this.e, be.generateFrom(h, ids, opts), max, opts.signal, opts.onToken);
      return { ...r, sharedPrefill: true, reusedTokens: h.ids.length };
    }
    const r = await collect(this.e, be.generate(ids, opts), max, opts.signal, opts.onToken);
    return { ...r, sharedPrefill: false, reusedTokens: 0 };
  }

  /** System 2: continue after the decision (the answer, then a new user turn with `prompt`). */
  async generate(prompt: string, opts: SessionGenerateOptions = {}): Promise<GenerateResult> {
    if (this.disposed) throw new Error("session disposed");
    if (!this.plans.length) throw new Error("session.generate: this session has no questions (evidence only); use generate()");
    const i = opts.question === undefined ? 0 : this.plans.findIndex((p) => p.q.name === opts.question);
    if (i < 0) throw new Error(`session.generate: no question named ${JSON.stringify(opts.question)}`);
    const m = this.e.manifest, plan = this.plans[i], a = this.answers[i];
    const answer = a.type === "predicate" ? m.generation.answers.predicate[decisionOf(a) === "yes" ? 1 : 0] : decisionOf(a);
    const cont = fill(m.generation.continue, { answer, prompt, think: thinkText(m, opts.think) }).text;
    return this.continueWith(i, [...plan.enc.ids, ...this.e.tok.encode(cont)], opts);
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    const s = this.states; this.states = [];
    for (const h of s) await h?.dispose();
  }
}

export async function decide(e: Engine, req: DecisionRequest, opts: DecideOptions = {}): Promise<DecisionSession> {
  if (!req || typeof req !== "object") throw new Error("decide: expected { input, questions }");
  const state = inputToText(req.input);
  const noQuestions = req.questions === undefined || (Array.isArray(req.questions) && req.questions.length === 0);
  if (noQuestions && req.evidence === undefined) throw new Error("decide: questions must be a non-empty array or object (or pass evidence)");
  const qs = noQuestions ? [] : normalizeQuestions(req.questions, e.manifest);
  const plans = qs.map((q) => planQuestion(e, q, state, opts.method ?? req.method));
  const evPlan = req.evidence !== undefined ? planEvidence(e, req.evidence, state) : null;
  const all: { enc: Encoded; positions: number[] }[] = evPlan ? [...plans, evPlan] : plans;
  const be = e.backend;
  if (opts.forecast && !e.forecast) throw new Error(e.forecastWhy ?? "forecast unsupported: this model has no forecast reader");
  const fx = opts.forecast ? e.forecast : undefined;
  const extras: (number[] | undefined)[] = all.map((p, i) => (fx && i < plans.length ? forecastPositions(fx.spec.pooling, plans[i].enc.last + 1) : undefined));
  const rows: (Float32Array[] | undefined)[] = [];
  const keep = (opts.keepState ?? true) && canShareState(be);
  const answers: Answer[] = [], states: (StateHandle | undefined)[] = [];
  let evidence: EvidenceAnswer | null = null;
  const k = all.length > 1 && be.snapshot && be.prefillFrom ? backOff(sharedPrefix(all.map((p, i) => ({ ...p, positions: [...p.positions, ...(extras[i] ?? [])] }))), all.map((p, i) => p.enc.ids.length - (keep && i < plans.length ? 2 : 1) * (be.minBlock ?? 0)), be.minBlock ?? 0) : 0;
  // The evidence pass keeps no state: System 2 continues from a question's prefill, never from the evidence prompt.
  const take = (i: number, r0: PrefillResult) => {
    const r = e.norm && r0.hiddens.length ? { ...r0, hiddens: r0.hiddens.map((h) => applyNorm(e.norm!, h)) } : r0;
    if (i < plans.length) { states.push(r.state); rows[i] = r.extra; answers.push(plans[i].read(r)); } else evidence = evPlan!.read(r);
  };
  const wantOf = (i: number) => (extras[i] ? { positions: all[i].positions, extra: { positions: extras[i]! } } : { positions: all[i].positions });
  try {
    if (k < 1) {
      for (let i = 0; i < all.length; i++) take(i, await be.prefill(all[i].enc.ids, wantOf(i), { keep: keep && i < plans.length }));
    } else {
      const h = await be.snapshot!(all[0].enc.ids.slice(0, k));
      try {
        for (let i = 0; i < all.length; i++) take(i, await be.prefillFrom!(h, all[i].enc.ids, wantOf(i), { keep: keep && i < plans.length }));
      } finally { await h.dispose(); }
    }
  } catch (err) {
    for (const s of states) await s?.dispose();
    throw err;
  }
  const session = new DecisionSession(e, plans, answers, keep ? states : []);
  session.evidence = evidence;
  session.rows = rows;
  if (opts.escalate && qs.length) {
    try { await escalate(e, req, qs, session, opts.escalate); }
    catch (err) { await session.dispose(); throw err; }
  }
  return session;
}

/** Apply the margin policy to a fresh session's answers (in place). */
async function escalate(e: Engine, req: DecisionRequest, qs: NQ[], session: DecisionSession, spec: true | (EscalationSpec & { target?: EscalationTarget })): Promise<void> {
  const call = spec === true ? undefined : spec;
  if (call) {
    const { target: _t, ...rest } = call;
    const problems: string[] = [];
    escalationProblems(rest, "escalate", e.manifest, problems);
    if (problems.length) throw new Error("decide: invalid escalate option:\n  - " + problems.join("\n  - "));
  }
  const base = manifestPolicy(e.manifest, e.build);
  const policies: MarginPolicy[] = qs.map((q) => mergePolicy(...base, q.escalation, call));
  const answers = session.answers;
  const hits: { i: number; threshold: number; value: number; p: MarginPolicy }[] = [];
  qs.forEach((q, i) => {
    const p = policies[i];
    if (Object.keys(p.below).length && p.to === null) throw new Error(`question ${q.name}: margin policy has thresholds but no target ("to": "generate" or { build })`);
    const h = shouldEscalate(answers[i], p);
    if (h) hits.push({ i, threshold: h.threshold, value: h.value, p });
  });
  if (!hits.length) return;
  const record = (h: { threshold: number; value: number; p: MarginPolicy }): Omit<EscalationRecord, "to"> =>
    ({ reason: "margin", metric: h.p.metric, margin: h.value, threshold: h.threshold, provisional: h.p.provisional, source: h.p.source });

  // to: { build } - grouped per build, one re-read per build on that engine (fresh prefill there).
  const byBuild = new Map<string, typeof hits>();
  for (const h of hits) if (h.p.to !== "generate" && h.p.to) byBuild.set(h.p.to.build, [...(byBuild.get(h.p.to.build) ?? []), h]);
  const raw = questionEntries(req.questions);
  for (const [build, hs] of byBuild) {
    const target = call?.target ?? (e.resolveBuild ? await e.resolveBuild(build) : null);
    if (!target) throw new Error(`margin policy: no engine for build "${build}" (load one, or pass escalate.target)`);
    const sub: DecisionsQuestion[] = hs.map(({ i }) => ({ ...raw[i][1], name: qs[i].name }));
    const t = await target.decide({ input: req.input, questions: sub, ...(req.method ? { method: req.method } : {}) }, { keepState: false, escalate: false });
    try {
      hs.forEach((h, j) => {
        const i = h.i;
        const a = t.answers[j], s1 = answers[i];
        if (!a || a.name !== s1.name || a.type !== s1.type) throw new Error(`margin policy: build "${build}" answered ${a?.name}/${a?.type} for ${s1.name}/${s1.type}`);
        const { escalation: _drop, ...from } = s1;
        answers[i] = { ...a, path: "escalated", escalation: { ...record(h), to: "build", build, from } } as Answer;
      });
    } finally { await t.dispose(); }
  }

  // to: "generate" - System 2 on the same weights, from the kept decision prefill when the backend keeps it.
  for (const h of hits) {
    const { i, p } = h;
    if (p.to !== "generate") continue;
    const q = qs[i];
    const prompt = fill(p.prompt, { options: (q.type === "predicate" ? ["yes", "no"] : q.values).join(", ") }).text;
    const r = await session.reconsider(i, prompt, { maxNewTokens: p.maxNewTokens, temperature: 0 });
    answers[i] = { ...answers[i], path: "escalated", escalation: { ...record(h), to: "generate", text: r.text, sharedPrefill: r.sharedPrefill, reusedTokens: r.reusedTokens, finishReason: r.finishReason } } as Answer;
  }
}

export interface GenerateOpts extends GenerateOptions {
  system?: string;
  think?: boolean;
  onToken?: (text: string, id: number) => void;
}

/** System 2 on its own: one user turn through the manifest chat template. */
export async function generate(e: Engine, prompt: string, opts: GenerateOpts = {}): Promise<GenerateResult> {
  const m = e.manifest;
  const system = opts.system !== undefined ? fill(m.generation.system ?? "{system}", { system: opts.system }).text : "";
  const text = fill(m.generation.turn, { prompt, system, think: thinkText(m, opts.think) }).text;
  const r = await collect(e, e.backend.generate(e.tok.encode(text), opts), Math.max(0, Math.floor(opts.maxNewTokens ?? 512)), opts.signal, opts.onToken);
  return { ...r, sharedPrefill: false, reusedTokens: 0 };
}
