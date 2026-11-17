// Request normalization. Accepts the OpenAI Decisions API shape
//   { input, questions: [{ name, type: "predicate"|"choice"|"score", instructions, choices?, levels? }] }
// with "noul" as an alias of "predicate", and open-jev's question objects
//   { type: "choice"|"score"|"noul", instructions, options, descriptions? }
// (array or keyed object). Output is one internal form, NQ.

import type { Manifest, QType, Method, EscalationSpec, TaskSpec, LabelSpec } from "./manifest.js";

export type InputPart =
  | { type: "input_text"; text: string }
  | { type: "input_json"; data: unknown };
export type InputMessage = { role: "user" | "system" | "developer" | "assistant"; content: string | InputPart[] };
export type DecisionInput = string | InputMessage[];

export type ChoiceSpec = string | { value: string; description?: string };
export type LevelSpec = string | { label: string; description?: string };

export interface DecisionsQuestion {
  name?: string;
  /** Required unless `task` is given. */
  type?: "predicate" | "noul" | "choice" | "score";
  /** Name of a manifest task (manifest.tasks): type, instructions, labels and pinned label tokens come from it;
   *  `instructions` and `method` here override the task's. */
  task?: string;
  instructions?: string;
  /** Decisions API (choice). */
  choices?: ChoiceSpec[];
  /** Decisions API (score), first = lowest. */
  levels?: LevelSpec[];
  /** open-jev shape: choice options or score levels. */
  options?: string[];
  /** open-jev shape: option -> description. */
  descriptions?: Record<string, string>;
  /** Per-question readout override. */
  method?: Method;
}

/** Evidence selector request (needs a build head of type "evidence"; see Manifest.evidence). */
export interface EvidenceRequest {
  /** Fills {requirement} in the manifest's evidence.instructions (the question the head was trained with). */
  requirement?: string;
  /** Full question text instead of the manifest's (the head was trained on one wording; others are untested). */
  instructions?: string;
  /** The passage's sentences, in order. Each is shown on one option line with whitespace collapsed. Default:
   *  splitSentences(input), a convenience splitter (not the segmentation any lab measurement used). */
  sentences?: string[];
}

export interface DecisionRequest {
  input: DecisionInput;
  /** Question objects, or task names (a string = { task: name, name }). May be omitted when `evidence` is given. */
  questions?: (DecisionsQuestion | string)[] | Record<string, DecisionsQuestion | string>;
  method?: Method;
  /** Also select the evidence sentence (one more forward pass sharing the input prefix); result in session.evidence. */
  evidence?: EvidenceRequest;
}

/** Convenience sentence splitter: breaks after . ! ? … (plus closing quotes/brackets) followed by whitespace, trims,
 *  drops empties. No abbreviation list; pass `evidence.sentences` for a segmentation you control. */
export function splitSentences(text: string): string[] {
  const out: string[] = [];
  const re = /[.!?…]+["'”’)\]]*(?=\s)/g;
  let start = 0, m: RegExpExecArray | null;
  while ((m = re.exec(text))) { out.push(text.slice(start, m.index + m[0].length).trim()); start = m.index + m[0].length; }
  out.push(text.slice(start).trim());
  return out.filter(Boolean);
}

/** Normalized question. `values` are what the answer reports; `texts` are what the prompt shows. */
export interface NQ {
  name: string;
  type: QType;
  instructions: string;
  values: string[];
  texts: string[];
  method?: Method;
  /** Task the question came from. */
  task?: string;
  /** Logits readout: token ids pinned by the task, in the order of `values` (predicate: [no, yes]). */
  labelIds?: number[];
  /** Task-level margin policy. */
  escalation?: EscalationSpec;
}

export function inputToText(input: DecisionInput): string {
  if (typeof input === "string") return input;
  if (!Array.isArray(input)) throw new Error("decide: input must be a string or an array of messages");
  return input.map((m, i) => {
    if (!m || typeof m !== "object" || typeof m.role !== "string") throw new Error(`decide: input[${i}] must be {role, content}`);
    const body = typeof m.content === "string" ? m.content : (Array.isArray(m.content) ? m.content : []).map((p: any, j) => {
      if (p?.type === "input_text" || p?.type === "output_text") return String(p.text ?? "");
      if (p?.type === "input_json") return JSON.stringify(p.data);
      throw new Error(`decide: input[${i}].content[${j}]: unsupported part type ${JSON.stringify(p?.type)} (text and JSON only; no images)`);
    }).join("\n");
    return input.length === 1 && m.role === "user" ? body : `${m.role}: ${body}`;
  }).join("\n");
}

const TYPE_ALIAS: Record<string, QType> = { predicate: "predicate", noul: "predicate", choice: "choice", score: "score" };

/** Expand a manifest task into a plain question (labels with descriptions, task name kept). */
function fromTask(q: DecisionsQuestion, m: Manifest, at: string): DecisionsQuestion {
  const t: TaskSpec | undefined = m.tasks?.[q.task as string];
  if (!t) throw new Error(`${at}: unknown task ${JSON.stringify(q.task)}${m.tasks ? ` (manifest tasks: ${Object.keys(m.tasks).join(", ")})` : " (this manifest declares no tasks)"}`);
  if (q.type !== undefined && (TYPE_ALIAS[q.type] ?? q.type) !== t.type) throw new Error(`${at}: type ${JSON.stringify(q.type)} differs from task ${q.task} (${t.type})`);
  if (q.choices !== undefined || q.levels !== undefined || q.options !== undefined) throw new Error(`${at}: a task question takes its labels from the manifest task`);
  const lab = (l: string | LabelSpec) => (typeof l === "string" ? l : { value: l.value, description: l.description });
  return {
    type: t.type, instructions: q.instructions ?? t.instructions, method: q.method ?? t.method,
    ...(t.type === "choice" ? { choices: t.choices!.map(lab) } : {}),
    ...(t.type === "score" && t.levels ? { levels: t.levels.map((l) => { const x = lab(l); return typeof x === "string" ? x : { label: x.value, description: x.description }; }) } : {}),
  };
}

/** Task-pinned logits ids in value order, if the task pins them. */
function taskLabelIds(t: TaskSpec): number[] | undefined {
  if (t.type === "predicate") return t.tokens ? [t.tokens.no.id, t.tokens.yes.id] : undefined;
  const src = (t.type === "choice" ? t.choices : t.levels) ?? [];
  const ids = src.map((l) => (typeof l === "string" ? undefined : l.token?.id));
  return ids.length && ids.every((x) => x !== undefined) ? ids as number[] : undefined;
}

/** [name, question] pairs in request order (a string question is a task name). */
export function questionEntries(qs: DecisionRequest["questions"]): [string | undefined, DecisionsQuestion][] {
  if (qs === undefined) return [];
  const obj = (q: DecisionsQuestion | string): DecisionsQuestion => (typeof q === "string" ? { task: q } : q);
  const entries: [string | undefined, DecisionsQuestion][] = Array.isArray(qs)
    ? qs.map((q0) => { const q = obj(q0); return [q?.name ?? (typeof q0 === "string" ? q0 : undefined), q]; })
    : qs && typeof qs === "object" ? Object.entries(qs).map(([k, q0]) => {
      const q = obj(q0);
      if (q?.name !== undefined && q.name !== k) throw new Error(`decide: question key "${k}" and name "${q.name}" differ`);
      return [k, q];
    }) : [];
  return entries;
}

export function normalizeQuestions(qs: DecisionRequest["questions"], m: Manifest): NQ[] {
  const entries = questionEntries(qs);
  if (!entries.length) throw new Error("decide: questions must be a non-empty array or object");
  const names = new Set<string>();
  return entries.map(([name00, q0], i) => {
    const at = `question ${name00 ?? i}`;
    if (!q0 || typeof q0 !== "object") throw new Error(`${at}: must be an object`);
    const task = q0.task;
    if (task !== undefined && typeof task !== "string") throw new Error(`${at}: task must be a string`);
    const q = task !== undefined ? fromTask(q0, m, at) : q0;
    const name0 = name00 ?? task;
    const type = TYPE_ALIAS[q.type as string];
    if (!type) throw new Error(`${at}: unknown type ${JSON.stringify(q.type)}; use "predicate" (alias "noul"), "choice" or "score"`);
    const name = name0 ?? `q${i}`;
    if (typeof name !== "string" || !name) throw new Error(`${at}: name must be a non-empty string`);
    if (names.has(name)) throw new Error(`decide: duplicate question name "${name}"`);
    names.add(name);
    const instructions = q.instructions ?? m.defaults.instructions?.[type];
    if (typeof instructions !== "string" || !instructions.trim()) throw new Error(`${at}: instructions required (no manifest default for ${type})`);
    if (q.method !== undefined && !["logits", "head", "both"].includes(q.method)) throw new Error(`${at}: method must be "logits", "head" or "both"`);

    let values: string[] = [], texts: string[] = [];
    if (type === "choice") {
      if (q.levels !== undefined) throw new Error(`${at}: choice takes choices (or options), not levels`);
      if (q.choices !== undefined && q.options !== undefined) throw new Error(`${at}: pass choices or options, not both`);
      const src: ChoiceSpec[] = q.choices ?? q.options ?? [];
      if (!Array.isArray(src) || src.length < 2) throw new Error(`${at}: choice needs at least 2 choices`);
      for (const c of src) {
        const value = typeof c === "string" ? c : c?.value;
        const desc = typeof c === "string" ? q.descriptions?.[c] : c?.description;
        if (typeof value !== "string" || !value.trim()) throw new Error(`${at}: empty or non-string choice ${JSON.stringify(c)}`);
        values.push(value); texts.push(desc ? `${value}: ${desc}` : value);
      }
    } else if (type === "score") {
      if (q.choices !== undefined) throw new Error(`${at}: score takes levels (or options), not choices`);
      if (q.levels !== undefined && q.options !== undefined) throw new Error(`${at}: pass levels or options, not both`);
      const src: LevelSpec[] = q.levels ?? q.options ?? m.defaults.levels ?? [];
      if (!Array.isArray(src) || src.length < 2) throw new Error(`${at}: score needs at least 2 levels (no manifest default)`);
      for (const l of src) {
        const label = typeof l === "string" ? l : l?.label;
        const desc = typeof l === "string" ? undefined : l?.description;
        if (typeof label !== "string" || !label.trim()) throw new Error(`${at}: empty or non-string level ${JSON.stringify(l)}`);
        values.push(label); texts.push(desc ? `${label}: ${desc}` : label);
      }
    } else if (q.choices !== undefined || q.levels !== undefined || (q.options !== undefined && q.options.length)) {
      throw new Error(`${at}: predicate takes no choices/levels/options`);
    }
    if (new Set(values).size !== values.length) throw new Error(`${at}: duplicate ${type === "score" ? "level" : "choice"} values`);
    const forbidden = m.encoding.kind === "tagged" ? m.encoding.forbiddenInLabels ?? [] : [];
    for (const t of texts) for (const f of forbidden) if (t.includes(f)) throw new Error(`${at}: label ${JSON.stringify(t)} contains the reserved string ${JSON.stringify(f)}`);
    const t = task !== undefined ? m.tasks![task] : undefined;
    const labelIds = t ? taskLabelIds(t) : undefined;
    return { name, type, instructions, values, texts, method: q.method,
      ...(task !== undefined ? { task } : {}), ...(labelIds ? { labelIds } : {}), ...(t?.escalation ? { escalation: t.escalation } : {}) };
  });
}
