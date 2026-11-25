// Per-model MANIFEST: every model-specific constant lives here (repo, pinned revision,
// file sha256s, hidden size, prompt template, marker token ids + strings, readout tokens,
// head files, per-build fidelity numbers). Validated at load; mismatches fail loudly.

import { mediaProblems } from "./media/manifest.js";
import type { MediaManifest } from "./media/types.js";

export const MANIFEST_SCHEMA = "s1s2.manifest/1";

export type QType = "predicate" | "choice" | "score";
export type RuntimeName = "genai" | "decode-only" | "transformers.js" | "webllm" | "litert";
export const RUNTIMES: readonly RuntimeName[] = ["genai", "decode-only", "transformers.js", "webllm", "litert"];
export type Method = "logits" | "head" | "both";

/** A token named both by id and by its exact decoded string; both are checked against the tokenizer. */
export interface TokenRef { id: number; text: string }
/** Logits read token pinned for one choice label (`label` is matched exactly against the question's choice text). */
export interface ChoiceTokenRef extends TokenRef { label: string }

/** A file pinned by repo-relative path, sha256 (hex) and optionally byte size. */
export interface PinnedFile { path: string; sha256: string; size?: number }

export interface Marker {
  /** Marker string as it appears in the rendered prompt (tagged encoding) or the special token (kev). */
  text: string;
  /** Tokens accepted AT the read position (asserted for every prompt). BPE can merge the marker's last
   *  character with what follows (ct0: "</opt><opt>" reads "><", the last option reads "></"), so this is a set. */
  tokens: TokenRef[];
  /** "last-token": the token covering the last character of `text`; "token": `text` is one special token. */
  at: "last-token" | "token";
}

export interface TaggedEncoding {
  kind: "tagged";
  /** Full decision prompt; placeholders {state} and {question}. Includes the chat template verbatim. */
  template: string;
  /** Question text per type; placeholders {instructions} and {options}. */
  question: Record<QType, string>;
  /** One option; placeholders {label}, {key} (A, B, ...), {index} (1-based). */
  option: string;
  optionSeparator?: string;
  /** Strings that must not appear in option / level labels (e.g. the option delimiters). */
  forbiddenInLabels?: string[];
  /** Rewrite special-token look-alikes in caller text (<|x|> -> <¦x¦>, as open-jev) so text cannot inject structure. */
  escapeSpecial?: boolean;
  /** Literal strings neutralized in caller text (after escapeSpecial), in order: every occurrence of each
   *  (trimmed) literal gets its first character replaced by "‹" (the escape for plain-text
   *  markers: "<option>" -> "‹option>"), so caller text cannot forge a marker. */
  escapeLiterals?: string[];
}

export interface KevEncoding {
  kind: "kev";
  /** Delimiters are special tokens; each is resolved by text and asserted by id. Port of open-jev's KEV layout. */
  delimiters: { state: TokenRef; question: TokenRef; optionStart: TokenRef; optionEnd: TokenRef; decide: TokenRef };
  /** Optional raw text before/after (e.g. a chat-template wrapper); tokenized separately. */
  prefix?: string;
  suffix?: string;
  /** Predicate questions are encoded as a two-option branch [no, yes] (open-jev's noul). */
  predicateOptions?: [string, string];
}

/** What a head reads: a question type, or "evidence" (the evidence selector: a pointer over the passage's sentences). */
export type HeadRole = QType | "evidence";
export const HEAD_ROLES: readonly HeadRole[] = ["predicate", "choice", "score", "evidence"];

export interface HeadSpec {
  file: PinnedFile;
  /** Question types this head reads, or "evidence" for the evidence selector (see Manifest.evidence). */
  types: HeadRole[];
  /** Factor on the head's dot product when head.bin's header does not state it (spec 6.1); must agree with the header
   *  when both do. 1 = 1/sqrt(headDim) folded into q/qb (s1s2, ct0-edge); 1/sqrt(headDim) = not folded (alias3-web). */
  scale?: number;
  /** Head type (spec 0.2 section 6): "pointer" (bilinear q.k over marker positions; default), "linear" (trained rows
   *  over the hidden state at one read position), "joint" (one pass over every question of one prompt, Clef). */
  kind?: HeadKind;
  /** File format: "head.bin" (pointer, fp16, alias3-web layout; default for pointer) or "safetensors" (any kind). */
  format?: "head.bin" | "safetensors";
  /** pointer (safetensors): temperature / score offset when the file's metadata does not state them. */
  temperature?: number | Partial<Record<HeadRole, number>>;
  scoreOffset?: number;
  /** linear: tensor names, rows per question type, softcap, and where it reads ("decide" marker or "last" token). */
  tensors?: { weight?: string; bias?: string | null };
  rows?: Partial<Record<HeadRole, { start: number; end?: number }>>;
  softcap?: number | null;
  read?: "decide" | "last";
  /** Layer the head reads (spec 6.6): "final" (default). Integer layers are describable; s1s2 0.1 runs "final" only. */
  layer?: "final" | number;
  /** joint: the head's own config (Clef joint_head_config.json). */
  config?: Record<string, unknown>;
  /** Format the head was converted from (spec 9.1), e.g. "kev-head.pt", "gguf-kev", "jev-judge-config". */
  source?: string;
}

export type HeadKind = "pointer" | "linear" | "joint";
export const HEAD_KINDS: readonly HeadKind[] = ["pointer", "linear", "joint"];

/**
 * Evidence selector (optional). A pointer head over the passage's sentences: the prompt is the choice layout of this
 * manifest's encoding with one option per sentence (whitespace collapsed), and the head reads the query at the decide
 * marker and one key at each option marker (citation as a one-pass decision, measured on SmolLM3-3B, S5-nano and Jurema-7B). The head file is a build head
 * whose `types` include "evidence".
 */
export interface EvidenceSpec {
  /** Question text the head was trained with; placeholder {requirement} (filled from the request). */
  instructions: string;
  /** Prepended to each (whitespace-collapsed) sentence inside its option, when the head was trained with a separator the
   *  option template does not carry (plain-text markers: "<option> " + sentence, while the template is
   *  " <option>{label} </option> \n" because its decision labels already start with a space). Default "". */
  sentencePrefix?: string;
  /** Free-text provenance (e.g. "evidence head, epoch chosen by dev loss"). */
  source?: string;
}

export interface BuildSpec {
  runtime: RuntimeName;
  repo: string;
  /** Pinned commit sha of `repo`; files are fetched from /resolve/<revision>/. */
  revision: string;
  /** Role -> pinned file. Roles per runtime: genai {model, data, genaiConfig?, embeddings?}; decode-only {model, data}; transformers.js {model, data...};
   *  webllm {lib?, config, recoveryRows?, recoveryP?}; litert {manifest, graph, weights, embed}.
   *  genai embeddings: token embedding table of an exclude_embeds build (graph input inputs_embeds), fp16
   *  little-endian row-major [vocab, hiddenSize]; size must be pinned. */
  files: Record<string, PinnedFile>;
  /** Name of the graph output carrying the hidden state (post final norm). null = logits only. */
  hiddenOutput: string | null;
  /** Head files in this build's repo (sha256 pinned). */
  heads?: Record<string, HeadSpec>;
  /** Out-of-graph final norm (spec 6.5): `hiddenOutput` carries the state BEFORE the decoder's final norm and the
   *  runtime applies this RMSNorm before any head reads it (alias3-web's norm.bin). Absent = hiddenOutput is post-norm. */
  norm?: { file: PinnedFile; kind?: "rmsnorm"; eps?: number; weight?: "w" | "1+w" };
  /** Attention of the decision pass the heads were trained with (spec 7): "causal" (default) or "noncausal"
   *  (pplx-decider v1.1 / d3). Described only: s1s2 0.1 refuses to load a "noncausal" build. */
  attention?: "causal" | "noncausal";
  /** Where the build was checked to run: "webgpu" (browser GPU), "wasm" (browser CPU, onnxruntime-web), "cpu" (Node
   *  onnxruntime-node), "cuda". For genai, "wasm" and "cpu" are opt-in: listed only after measuring that path. */
  devices: string[];
  dtype?: string;
  modelId?: string;
  /** genai: KV head dim when the graph leaves it symbolic (read from genaiConfig when pinned). */
  kvHeadDim?: number;
  maxSafeTokens?: number;
  /** Per-build fidelity numbers, filled by the lab (null until measured). */
  fidelity: Fidelity | null;
  /** Margin policy for this build (thresholds differ per build: an int4 graph's margins are not an int8 graph's). */
  escalation?: EscalationSpec;
}

/**
 * Margin policy (System-1 escalation), as stated in a manifest, a build, a task or a call. Every field is optional
 * at each level; levels merge as: library defaults < manifest < build < task < call.
 * Thresholds apply to Answer.margin (probability gap) or Answer.rawMargin (see readout.ts); there is no library default
 * threshold: a number only exists once someone measured it for that model/build/readout.
 */
export interface EscalationSpec {
  /** Answer field the thresholds apply to: "margin" (probability gap p(1st) - p(2nd); default) or "rawMargin". */
  metric?: "margin" | "rawMargin";
  /** Escalate answers whose metric is strictly below this, per question type. A type without a threshold is never escalated. */
  below?: Partial<Record<QType, number>>;
  /** "generate": System 2 on the same weights, continuing from the decision's prefill where the backend keeps it
   *  (otherwise the same ids are prefilled again). {build}: re-read the question on another build of the manifest
   *  (e.g. an int8 graph after an int4 one). */
  to?: "generate" | { build: string };
  /** to "generate": the user turn appended after the decision prompt; placeholder {options} (the labels, comma-separated). */
  prompt?: string;
  maxNewTokens?: number;
  /** True until a measured result backs the thresholds. */
  provisional?: boolean;
  /** Free-text provenance of the thresholds (e.g. "15% quantile of the dev margins"). */
  source?: string;
}

/** A labelled task: a question type with its own instructions, labels and (optionally) pinned label tokens. */
export interface LabelSpec { value: string; description?: string; token?: TokenRef }
export interface TaskSpec {
  type: QType;
  instructions: string;
  /** choice: the label set. Pinned `token`s (all labels or none) are read by the logits readout instead of the first
   *  token of the label after the prompt. */
  choices?: (string | LabelSpec)[];
  /** score: levels, first = lowest; same token rule as choices. */
  levels?: (string | LabelSpec)[];
  /** predicate: tokens read for yes/no instead of logits.predicate. */
  tokens?: { yes: TokenRef; no: TokenRef };
  method?: Exclude<Method, "both">;
  escalation?: EscalationSpec;
}

export interface Fidelity {
  /** Reference the build was compared with (e.g. "pytorch-bf16"). */
  reference: string;
  /** Number of questions in the measured set. */
  n: number;
  /** Share of decisions that differ from the reference, per readout. */
  decisionChanged?: Partial<Record<"logits" | "head", number>>;
  /** Share of answers that flip when the option order is swapped. */
  orderInversion?: Partial<Record<"logits" | "head", number>>;
  measuredAt?: string;
  notes?: string;
}

export interface Manifest {
  /** `"s1s2.manifest/1"`. */
  schema: typeof MANIFEST_SCHEMA;
  id: string;
  base: { repo: string; revision: string };
  hiddenSize: number;
  /** Which hidden state the head reads. ct0: the graph output after the final norm; no JS norm. */
  layer: { read: "final"; norm: "in-graph" };
  tokenizer: { repo: string; revision: string; files: { tokenizer: PinnedFile; config: PinnedFile } };
  eos: TokenRef[];
  /** Token fed once after load to warm the graph. */
  warmup?: TokenRef;
  encoding: TaggedEncoding | KevEncoding;
  markers: { decide: Marker; option: Marker; last?: Marker };
  /** Where the head reads its key vector: "last" = final prompt token, "option" = option markers. */
  readout: { predicate: "last" | "options"; choice: "option"; score: "last" | "option" };
  logits: {
    predicate: { yes: TokenRef; no: TokenRef };
    /** "first-token": first token of each option label after the rendered prompt; "letters": of A, B, ...;
     *  "tokens": the id pinned for each option label (`tokens[].label` = the exact choice text). Use "tokens" when the
     *  token read at the cue is not the option text's first token (lab kev markers: option "Yes", cue reads " Yes"). */
    choice: { map: "first-token" | "letters" | "tokens"; tokens?: ChoiceTokenRef[] };
    /** "tokens": read `tokens[i]` for level i (count must equal the levels); "first-token": first token of each level label.
     *  `lead`: text appended after the rendered prompt before the level tokens are read by the logits readout (the
     *  hidden-state read positions of heads do not move). Absent: " " when the prompt does not end in whitespace and the
     *  tokenizer writes the space before a level as its own token (SmolLM2/3: " 1" = [" ", "1"]; reading the digit right
     *  after "Answer:" puts it where the model writes a space, label mass ~1e-5 on SST-5 / B2W with SmolLM3-3B); "" disables it. */
    score: { map: "tokens" | "first-token"; tokens?: TokenRef[]; lead?: string };
    /** Softmax temperature of the logits readout per question type (default 1): p = softmax(L / T); rawMargin is the
     *  gap of L / T. Predicate: p = sigmoid((L[yes] - L[no]) / T). */
    temperature?: Partial<Record<QType, number>>;
  };
  defaults: {
    method: Record<QType, Exclude<Method, "both">>;
    instructions?: Partial<Record<QType, string>>;
    levels?: string[];
    maxSafeTokens?: number;
  };
  generation: {
    /** User turn; placeholders {prompt}, {system}, {think}. */
    turn: string;
    system?: string;
    think?: { on: string; off: string };
    /** Continuation after a decision prompt (System 1 -> System 2); placeholders {answer}, {prompt}, {think}. */
    continue: string;
    /** Optional: continuation after a decision prompt for margin escalation to System 2, WITHOUT the decided answer;
     *  placeholders {prompt}, {think}. Default: `continue` with an empty {answer}. */
    escalate?: string;
    answers: { predicate: [string, string] };
  };
  builds: Record<string, BuildSpec>;
  /** Named labelled tasks; questions refer to them by name ("sentiment" or { task: "sentiment" }). Optional. */
  tasks?: Record<string, TaskSpec>;
  /** Margin policy defaults for this model. Optional; see EscalationSpec. */
  escalation?: EscalationSpec;
  /** Evidence selector question (optional; see EvidenceSpec). */
  evidence?: EvidenceSpec;
  /** SPDX license expression of the package as distributed (weights, heads, graphs). Optional; see spec/s1s2-package.md. */
  license?: string;
  /** Where the package came from (base model, data, recipe, tools). Optional; see Provenance. */
  provenance?: Provenance;
  /** Confidence definition answers report (spec 5.1). Reserved hook until a comparison of confidence definitions is done; ignored by s1s2 0.1. */
  confidence?: { definition: "C1" | "C3" | "C4" | "C5" | "C6" | "C7" | "tbd-007"; source?: string; [k: string]: unknown };
}

/** Package provenance (S1S2 package spec 0.1-draft, section "Provenance and license"). Free-form beyond these keys. */
export interface Provenance {
  /** Tool that wrote the package (e.g. {name: "s1s2-convert", version: "0.1.0"}). */
  generator: { name: string; version: string; commit?: string; [k: string]: unknown };
  /** ISO 8601 timestamp. */
  createdAt?: string;
  /** Format the package was converted from, declared by the converter (spec 9; never guessed from file names). */
  source?: { format: string; repo?: string; revision?: string; files?: string[]; [k: string]: unknown };
  /** Base checkpoint the package derives from; `license` is its SPDX id. */
  base?: { repo: string; revision: string; license?: string; [k: string]: unknown };
  /** Training data: sha256 of the dataset file as read, row count, optional name/license/splits. */
  dataset?: { sha256: string; rows: number; name?: string; license?: string; splits?: Record<string, number>; [k: string]: unknown };
  /** Training recipe: free-text recipe reference, seed, hyperparameters, selection rule, measured metrics. */
  training?: { recipe: string; seed?: number; [k: string]: unknown };
  /** Per build name: the tool that produced its graph, version, source (fork/commit), patches and arguments. */
  builds?: Record<string, { tool: string; version: string; source?: string; patches?: { path: string; sha256: string; description?: string }[]; args?: string[]; [k: string]: unknown }>;
  notes?: string;
  [k: string]: unknown;
}

const HEX64 = /^[0-9a-f]{64}$/;
const SHA1 = /^[0-9a-f]{40}$/;

export class ManifestError extends Error {
  constructor(public readonly problems: string[]) {
    super("manifest invalid:\n  - " + problems.join("\n  - "));
    this.name = "ManifestError";
  }
}

/** Validate a parsed manifest; throws ManifestError listing every problem. Returns it typed. */
export function validateManifest(m: any): Manifest {
  const p: string[] = [];
  const need = (c: boolean, msg: string) => { if (!c) p.push(msg); };
  const isObj = (x: any) => !!x && typeof x === "object" && !Array.isArray(x);
  const str = (x: any) => typeof x === "string" && x.length > 0;
  const int = (x: any) => Number.isInteger(x) && x >= 0;
  const tokenRef = (x: any, at: string) => need(isObj(x) && int(x.id) && typeof x.text === "string" && x.text.length > 0, `${at}: expected {id: int, text: string}`);
  const pinned = (x: any, at: string) => {
    if (!isObj(x)) return p.push(`${at}: expected {path, sha256}`);
    need(str(x.path) && (/^https:\/\//.test(x.path) || (!x.path.startsWith("/") && !/^[a-z]+:/i.test(x.path))) && !x.path.split("/").includes(".."),
      `${at}.path: repo-relative path or absolute https URL required`);
    need(typeof x.sha256 === "string" && HEX64.test(x.sha256), `${at}.sha256: 64 lowercase hex chars required`);
    if (x.size !== undefined) need(int(x.size), `${at}.size: non-negative integer`);
  };
  const rev = (x: any, at: string) => need(typeof x === "string" && SHA1.test(x), `${at}: pinned 40-hex commit sha required (not a branch)`);

  if (!isObj(m)) throw new ManifestError(["manifest: expected an object"]);
  // Media packages (spec/MULTIMODAL.md): a manifest without "text" in `modalities` has no text sections and goes to
  // validateMediaManifest / loadMedia. With "text" it is a text manifest and its media blocks are checked below.
  if (m.modalities !== undefined && Array.isArray(m.modalities) && !m.modalities.includes("text"))
    throw new ManifestError(["modalities: this manifest has no \"text\"; it is a media package: use loadMedia() / validateMediaManifest() (spec/MULTIMODAL.md)"]);
  need(m.schema === MANIFEST_SCHEMA, `schema: unsupported schema id ${JSON.stringify(m.schema)}; expected "${MANIFEST_SCHEMA}"`);
  need(str(m.id), "id: non-empty string");
  need(isObj(m.base) && str(m.base?.repo), "base.repo: string");
  rev(m.base?.revision, "base.revision");
  need(int(m.hiddenSize) && m.hiddenSize > 0, "hiddenSize: positive integer");
  need(isObj(m.layer) && m.layer.read === "final" && m.layer.norm === "in-graph", 'layer: {read: "final", norm: "in-graph"} (the only readout supported)');
  if (!isObj(m.tokenizer)) p.push("tokenizer: object required");
  else {
    need(str(m.tokenizer.repo), "tokenizer.repo: string");
    rev(m.tokenizer.revision, "tokenizer.revision");
    pinned(m.tokenizer.files?.tokenizer, "tokenizer.files.tokenizer");
    pinned(m.tokenizer.files?.config, "tokenizer.files.config");
  }
  need(Array.isArray(m.eos) && m.eos.length > 0, "eos: non-empty array");
  (m.eos ?? []).forEach((t: any, i: number) => tokenRef(t, `eos[${i}]`));
  if (m.warmup !== undefined) tokenRef(m.warmup, "warmup");

  const e = m.encoding;
  if (!isObj(e)) p.push("encoding: object required");
  else if (e.kind === "tagged") {
    need(str(e.template) && e.template.includes("{state}") && e.template.includes("{question}"), "encoding.template: must contain {state} and {question}");
    if (str(e.template)) need(e.template.indexOf("{state}") < e.template.indexOf("{question}"), "encoding.template: {state} must come before {question} (shared prefix)");
    for (const t of ["predicate", "choice", "score"]) need(typeof e.question?.[t] === "string", `encoding.question.${t}: string`);
    need(typeof e.question?.choice === "string" && e.question.choice.includes("{options}"), "encoding.question.choice: must contain {options}");
    need(str(e.option) && e.option.includes("{label}"), "encoding.option: must contain {label}");
    if (e.forbiddenInLabels !== undefined) need(Array.isArray(e.forbiddenInLabels) && e.forbiddenInLabels.every(str), "encoding.forbiddenInLabels: string[]");
    if (e.escapeSpecial !== undefined) need(typeof e.escapeSpecial === "boolean", "encoding.escapeSpecial: boolean");
    if (e.escapeLiterals !== undefined) need(Array.isArray(e.escapeLiterals) && e.escapeLiterals.every((x: any) => str(x) && x.trim().length > 1), "encoding.escapeLiterals: string[] (each at least 2 characters after trim)");
  } else if (e.kind === "kev") {
    for (const k of ["state", "question", "optionStart", "optionEnd", "decide"]) tokenRef(e.delimiters?.[k], `encoding.delimiters.${k}`);
    if (e.predicateOptions !== undefined) need(Array.isArray(e.predicateOptions) && e.predicateOptions.length === 2 && e.predicateOptions.every(str), "encoding.predicateOptions: [no, yes]");
  } else p.push(`encoding.kind: "tagged" or "kev", got ${JSON.stringify(e.kind)}`);

  for (const k of ["decide", "option"]) {
    const mk = m.markers?.[k];
    need(isObj(mk) && str(mk.text) && (mk.at === "last-token" || mk.at === "token") && Array.isArray(mk.tokens) && mk.tokens.length > 0,
      `markers.${k}: {text, tokens: [{id, text}], at: "last-token"|"token"}`);
    (Array.isArray(mk?.tokens) ? mk.tokens : []).forEach((t: any, i: number) => tokenRef(t, `markers.${k}.tokens[${i}]`));
  }
  const lastMk = m.markers?.last;
  if (lastMk !== undefined) {
    need(isObj(lastMk) && Array.isArray(lastMk.tokens) && lastMk.tokens.length > 0, "markers.last: {text, tokens, at}");
    (Array.isArray(lastMk?.tokens) ? lastMk.tokens : []).forEach((t: any, i: number) => tokenRef(t, `markers.last.tokens[${i}]`));
  }
  if (isObj(e) && e.kind === "kev" && isObj(m.markers?.decide) && isObj(m.markers?.option)) {
    const only = (mk: any, id: any) => Array.isArray(mk.tokens) && mk.tokens.length === 1 && mk.tokens[0]?.id === id;
    need(only(m.markers.decide, e.delimiters?.decide?.id) && only(m.markers.option, e.delimiters?.optionEnd?.id),
      "markers: for kev, the decide/option marker tokens must be exactly the decide/optionEnd delimiters");
  }
  need(isObj(m.readout) && ["last", "options"].includes(m.readout.predicate) && m.readout.choice === "option" && ["last", "option"].includes(m.readout.score),
    'readout: {predicate: "last"|"options", choice: "option", score: "last"|"option"}');
  if (m.readout?.score === "option") need(isObj(e) && e.kind === "kev", 'readout.score "option" requires the kev encoding (levels as options, Kev)');
  if (m.readout?.predicate === "options") need(isObj(e) && e.kind === "kev", 'readout.predicate "options" requires the kev encoding (predicate as a [no, yes] branch)');

  if (!isObj(m.logits)) p.push("logits: object required");
  else {
    tokenRef(m.logits.predicate?.yes, "logits.predicate.yes"); tokenRef(m.logits.predicate?.no, "logits.predicate.no");
    if (m.logits.predicate?.yes?.id === m.logits.predicate?.no?.id) p.push("logits.predicate: yes and no must be different tokens");
    const ch = m.logits.choice;
    need(["first-token", "letters", "tokens"].includes(ch?.map), 'logits.choice.map: "first-token" | "letters" | "tokens"');
    if (ch?.map === "tokens") {
      need(Array.isArray(ch.tokens) && ch.tokens.length >= 2, "logits.choice.tokens: at least 2 tokens");
      (ch.tokens ?? []).forEach((t: any, i: number) => { tokenRef(t, `logits.choice.tokens[${i}]`); need(str(t?.label), `logits.choice.tokens[${i}].label: non-empty string`); });
      if (Array.isArray(ch.tokens)) {
        need(new Set(ch.tokens.map((t: any) => t?.id)).size === ch.tokens.length, "logits.choice.tokens: ids must be distinct");
        need(new Set(ch.tokens.map((t: any) => t?.label)).size === ch.tokens.length, "logits.choice.tokens: labels must be distinct");
      }
    } else need(ch?.tokens === undefined, 'logits.choice.tokens: only with map "tokens"');
    if (m.logits.temperature !== undefined) {
      if (!isObj(m.logits.temperature)) p.push("logits.temperature: { predicate?, choice?, score? } of positive numbers");
      else for (const [t, v] of Object.entries<any>(m.logits.temperature))
        need(["predicate", "choice", "score"].includes(t) && typeof v === "number" && Number.isFinite(v) && v > 0, `logits.temperature.${t}: positive number for a question type`);
    }
    const sc = m.logits.score;
    need(["tokens", "first-token"].includes(sc?.map), 'logits.score.map: "tokens" | "first-token"');
    if (sc?.lead !== undefined) need(typeof sc.lead === "string" && sc.lead.trim() === "", "logits.score.lead: a whitespace-only string");
    if (sc?.map === "tokens") {
      need(Array.isArray(sc.tokens) && sc.tokens.length >= 2, "logits.score.tokens: at least 2 tokens");
      (sc.tokens ?? []).forEach((t: any, i: number) => tokenRef(t, `logits.score.tokens[${i}]`));
      if (Array.isArray(sc.tokens)) need(new Set(sc.tokens.map((t: any) => t?.id)).size === sc.tokens.length, "logits.score.tokens: ids must be distinct");
    }
  }
  const d = m.defaults;
  if (!isObj(d)) p.push("defaults: object required");
  else {
    for (const t of ["predicate", "choice", "score"]) need(["logits", "head"].includes(d.method?.[t]), `defaults.method.${t}: "logits" | "head"`);
    if (d.levels !== undefined) need(Array.isArray(d.levels) && d.levels.length >= 2 && d.levels.every(str), "defaults.levels: string[] (>= 2)");
    if (m.logits?.score?.map === "tokens" && Array.isArray(d.levels) && Array.isArray(m.logits.score.tokens))
      need(d.levels.length === m.logits.score.tokens.length, "defaults.levels: count must equal logits.score.tokens");
  }
  const g = m.generation;
  if (!isObj(g)) p.push("generation: object required");
  else {
    need(str(g.turn) && g.turn.includes("{prompt}"), "generation.turn: must contain {prompt}");
    need(str(g.continue) && g.continue.includes("{prompt}") && g.continue.includes("{answer}"), "generation.continue: must contain {answer} and {prompt}");
    need(Array.isArray(g.answers?.predicate) && g.answers.predicate.length === 2, "generation.answers.predicate: [noText, yesText]");
    if (g.escalate !== undefined) need(str(g.escalate) && g.escalate.includes("{prompt}") && !g.escalate.includes("{answer}"), "generation.escalate: must contain {prompt} and not {answer}");
    if (g.think !== undefined) need(typeof g.think.on === "string" && typeof g.think.off === "string", "generation.think: {on, off}");
  }
  if (!isObj(m.builds) || !Object.keys(m.builds).length) p.push("builds: at least one build");
  else for (const [name, b] of Object.entries<any>(m.builds)) {
    const at = `builds.${name}`;
    if (!isObj(b)) { p.push(`${at}: object`); continue; }
    need(RUNTIMES.includes(b.runtime), `${at}.runtime: one of ${RUNTIMES.join(", ")}`);
    need(str(b.repo), `${at}.repo: string`);
    rev(b.revision, `${at}.revision`);
    if (!isObj(b.files)) p.push(`${at}.files: object`);
    else for (const [role, f] of Object.entries(b.files)) pinned(f, `${at}.files.${role}`);
    const roles = REQUIRED_ROLES[b.runtime as RuntimeName] ?? [];
    for (const r of roles) need(isObj(b.files?.[r]), `${at}.files.${r}: required for runtime ${b.runtime}`);
    if (isObj(b.files?.embeddings)) {
      need(b.runtime === "genai", `${at}.files.embeddings: only for runtime genai (exclude_embeds builds)`);
      need(int(b.files.embeddings.size) && int(m.hiddenSize) && b.files.embeddings.size % (m.hiddenSize * 2) === 0,
        `${at}.files.embeddings.size: pinned, a multiple of hiddenSize * 2 (fp16 [vocab, hiddenSize])`);
    }
    need(b.hiddenOutput === null || str(b.hiddenOutput), `${at}.hiddenOutput: string or null`);
    need(Array.isArray(b.devices) && b.devices.length > 0, `${at}.devices: non-empty array`);
    need(b.fidelity === null || isObj(b.fidelity), `${at}.fidelity: object or null (slot filled by the lab)`);
    if (isObj(b.heads)) for (const [hn, h] of Object.entries<any>(b.heads)) {
      pinned(h?.file, `${at}.heads.${hn}.file`);
      need(Array.isArray(h?.types) && h.types.length > 0 && h.types.every((t: any) => (HEAD_ROLES as readonly string[]).includes(t)), `${at}.heads.${hn}.types: ("predicate" | "choice" | "score" | "evidence")[]`);
      const kind = h?.kind ?? "pointer";
      need((HEAD_KINDS as readonly string[]).includes(kind), `${at}.heads.${hn}.kind: "pointer" | "linear" | "joint"`);
      need(h?.format === undefined || h.format === "head.bin" || h.format === "safetensors", `${at}.heads.${hn}.format: "head.bin" | "safetensors"`);
      need(kind === "pointer" || h?.format !== "head.bin", `${at}.heads.${hn}: head.bin holds pointer heads only`);
      need(h?.layer === undefined || h.layer === "final" || (Number.isInteger(h.layer) && h.layer >= 0), `${at}.heads.${hn}.layer: "final" | layer index`);
      need(h?.read === undefined || h.read === "decide" || h.read === "last", `${at}.heads.${hn}.read: "decide" | "last"`);
      need(h?.softcap === undefined || h.softcap === null || (typeof h.softcap === "number" && h.softcap > 0), `${at}.heads.${hn}.softcap: positive number | null`);
      if (kind === "joint") need(isObj(h?.config), `${at}.heads.${hn}.config: the joint head config`);
      if (h?.scale !== undefined) need(typeof h.scale === "number" && Number.isFinite(h.scale) && h.scale > 0, `${at}.heads.${hn}.scale: positive number`);
    }
    need(b.attention === undefined || b.attention === "causal" || b.attention === "noncausal", `${at}.attention: "causal" | "noncausal"`);
    if (b.norm !== undefined) {
      if (!isObj(b.norm)) p.push(`${at}.norm: {file, kind?, eps?, weight?}`);
      else {
        pinned(b.norm.file, `${at}.norm.file`);
        need(b.norm.kind === undefined || b.norm.kind === "rmsnorm", `${at}.norm.kind: "rmsnorm"`);
        need(b.norm.eps === undefined || (typeof b.norm.eps === "number" && Number.isFinite(b.norm.eps) && b.norm.eps >= 0), `${at}.norm.eps: number >= 0`);
        need(b.norm.weight === undefined || b.norm.weight === "w" || b.norm.weight === "1+w", `${at}.norm.weight: "w" | "1+w"`);
        need(b.hiddenOutput !== null, `${at}.norm: needs a hiddenOutput`);
      }
    }
    if (isObj(b.heads)) {
      const seen = new Map<string, string>();
      for (const [hn, h] of Object.entries<any>(b.heads)) for (const t of h?.types ?? []) {
        if (seen.has(t)) p.push(`${at}.heads: type ${t} read by both ${seen.get(t)} and ${hn}`);
        seen.set(t, hn);
      }
    }
    if (b.kvHeadDim !== undefined) need(int(b.kvHeadDim) && b.kvHeadDim > 0, `${at}.kvHeadDim: positive integer`);
    if (b.maxSafeTokens !== undefined) need(int(b.maxSafeTokens) && b.maxSafeTokens > 0, `${at}.maxSafeTokens: positive integer`);
  }
  if (m.escalation !== undefined) escalationProblems(m.escalation, "escalation", m, p);
  if (isObj(m.builds)) for (const [name, b] of Object.entries<any>(m.builds)) if (isObj(b) && b.escalation !== undefined) escalationProblems(b.escalation, `builds.${name}.escalation`, m, p);
  if (m.evidence !== undefined) {
    if (!isObj(m.evidence)) p.push("evidence: object {instructions, source?}");
    else {
      need(str(m.evidence.instructions) && m.evidence.instructions.trim().length > 0, "evidence.instructions: non-empty string");
      if (m.evidence.source !== undefined) need(typeof m.evidence.source === "string", "evidence.source: string");
      if (m.evidence.sentencePrefix !== undefined) need(typeof m.evidence.sentencePrefix === "string" && !/\n/.test(m.evidence.sentencePrefix), "evidence.sentencePrefix: string without newlines");
    }
  }
  if (m.license !== undefined) need(str(m.license) && m.license.trim().length > 0, "license: non-empty SPDX expression");
  if (m.provenance !== undefined) provenanceProblems(m.provenance, p);
  if (m.modalities !== undefined || m.audio !== undefined) p.push(...mediaProblems(m, false));
  if (m.confidence !== undefined) need(isObj(m.confidence) && ["C1", "C3", "C4", "C5", "C6", "C7", "tbd-007"].includes(m.confidence.definition), 'confidence.definition: "C1" | "C3" | "C4" | "C5" | "C6" | "C7" | "tbd-007" (spec 5.1)');
  if (m.tasks !== undefined) {
    if (!isObj(m.tasks)) p.push("tasks: object of named tasks");
    else for (const [name, t] of Object.entries<any>(m.tasks)) taskProblems(t, `tasks.${name}`, m, p, tokenRef);
  }
  if (p.length) throw new ManifestError(p);
  return m as Manifest;
}

const QTYPES = ["predicate", "choice", "score"];
const isObj_ = (x: any) => !!x && typeof x === "object" && !Array.isArray(x);

/** Problems of an EscalationSpec (manifest, build, task or call level). */
export function escalationProblems(e: any, at: string, m: any, p: string[]): void {
  if (!isObj_(e)) { p.push(`${at}: object`); return; }
  const known = ["metric", "below", "to", "prompt", "maxNewTokens", "provisional", "source"];
  for (const k of Object.keys(e)) if (!known.includes(k)) p.push(`${at}.${k}: unknown key (known: ${known.join(", ")})`);
  if (e.below !== undefined) {
    if (!isObj_(e.below)) p.push(`${at}.below: { predicate?, choice?, score? } of numbers`);
    else for (const [t, v] of Object.entries<any>(e.below)) {
      if (!QTYPES.includes(t)) p.push(`${at}.below.${t}: not a question type`);
      else if (typeof v !== "number" || !Number.isFinite(v) || v < 0) p.push(`${at}.below.${t}: finite number >= 0`);
    }
  }
  if (e.to !== undefined) {
    if (e.to === "generate") { /* ok */ }
    else if (isObj_(e.to) && typeof e.to.build === "string" && e.to.build) {
      if (isObj_(m?.builds) && !m.builds[e.to.build]) p.push(`${at}.to.build: unknown build "${e.to.build}"`);
    } else p.push(`${at}.to: "generate" or { build: "<build name>" }`);
  }
  if (e.metric !== undefined && !["margin", "rawMargin"].includes(e.metric)) p.push(`${at}.metric: "margin" | "rawMargin"`);
  if (e.prompt !== undefined && (typeof e.prompt !== "string" || !e.prompt.trim())) p.push(`${at}.prompt: non-empty string`);
  if (e.maxNewTokens !== undefined && !(Number.isInteger(e.maxNewTokens) && e.maxNewTokens > 0)) p.push(`${at}.maxNewTokens: positive integer`);
  if (e.provisional !== undefined && typeof e.provisional !== "boolean") p.push(`${at}.provisional: boolean`);
  if (e.source !== undefined && typeof e.source !== "string") p.push(`${at}.source: string`);
}

/** Problems of the optional provenance block (only the keys the spec names are typed; others are free-form). */
export function provenanceProblems(v: any, p: string[]): void {
  const at = "provenance";
  if (!isObj_(v)) { p.push(`${at}: object`); return; }
  const s = (x: any) => typeof x === "string" && x.length > 0;
  if (!isObj_(v.generator) || !s(v.generator.name) || !s(v.generator.version)) p.push(`${at}.generator: {name, version} strings`);
  if (v.createdAt !== undefined && !(s(v.createdAt) && !Number.isNaN(Date.parse(v.createdAt)))) p.push(`${at}.createdAt: ISO 8601 date-time`);
  if (v.base !== undefined) {
    if (!isObj_(v.base) || !s(v.base.repo) || !/^[0-9a-f]{40}$/.test(v.base.revision ?? "")) p.push(`${at}.base: {repo, revision (40-hex)}`);
    else if (v.base.license !== undefined && !s(v.base.license)) p.push(`${at}.base.license: string`);
  }
  if (v.dataset !== undefined) {
    if (!isObj_(v.dataset) || !/^[0-9a-f]{64}$/.test(v.dataset.sha256 ?? "") || !(Number.isInteger(v.dataset.rows) && v.dataset.rows >= 0))
      p.push(`${at}.dataset: {sha256 (64 hex), rows (integer)}`);
  }
  if (v.source !== undefined && (!isObj_(v.source) || !s(v.source.format))) p.push(`${at}.source: {format: string}`);
  if (v.training !== undefined && (!isObj_(v.training) || !s(v.training.recipe))) p.push(`${at}.training: {recipe: string}`);
  if (v.builds !== undefined) {
    if (!isObj_(v.builds)) p.push(`${at}.builds: object`);
    else for (const [n, b] of Object.entries<any>(v.builds)) {
      if (!isObj_(b) || !s(b.tool) || !s(b.version)) { p.push(`${at}.builds.${n}: {tool, version}`); continue; }
      if (b.patches !== undefined && !(Array.isArray(b.patches) && b.patches.every((x: any) => isObj_(x) && s(x.path) && /^[0-9a-f]{64}$/.test(x.sha256 ?? ""))))
        p.push(`${at}.builds.${n}.patches: [{path, sha256}]`);
      if (b.args !== undefined && !(Array.isArray(b.args) && b.args.every((x: any) => typeof x === "string"))) p.push(`${at}.builds.${n}.args: string[]`);
    }
  }
}

function taskProblems(t: any, at: string, m: any, p: string[], tokenRef: (x: any, at: string) => void): void {
  if (!isObj_(t)) { p.push(`${at}: object`); return; }
  if (!QTYPES.includes(t.type)) { p.push(`${at}.type: "predicate" | "choice" | "score"`); return; }
  if (typeof t.instructions !== "string" || !t.instructions.trim()) p.push(`${at}.instructions: non-empty string`);
  if (t.method !== undefined && !["logits", "head"].includes(t.method)) p.push(`${at}.method: "logits" | "head"`);
  if (t.escalation !== undefined) escalationProblems(t.escalation, `${at}.escalation`, m, p);
  const labelsKey = t.type === "choice" ? "choices" : t.type === "score" ? "levels" : null;
  for (const k of ["choices", "levels", "tokens"]) {
    const allowed = k === labelsKey || (k === "tokens" && t.type === "predicate");
    if (t[k] !== undefined && !allowed) p.push(`${at}.${k}: not used by a ${t.type} task`);
  }
  if (t.type === "predicate") {
    if (t.tokens !== undefined) {
      if (!isObj_(t.tokens)) p.push(`${at}.tokens: { yes, no }`);
      else { tokenRef(t.tokens.yes, `${at}.tokens.yes`); tokenRef(t.tokens.no, `${at}.tokens.no`);
        if (t.tokens.yes?.id === t.tokens.no?.id) p.push(`${at}.tokens: yes and no must be different tokens`); }
    }
    return;
  }
  const src = t[labelsKey!];
  if (labelsKey === "levels" && src === undefined && Array.isArray(m?.defaults?.levels)) return; // manifest default levels
  if (!Array.isArray(src) || src.length < 2) { p.push(`${at}.${labelsKey}: at least 2 labels`); return; }
  const values: string[] = [];
  let pinned = 0;
  src.forEach((l: any, i: number) => {
    const v = typeof l === "string" ? l : l?.value;
    if (typeof v !== "string" || !v.trim()) p.push(`${at}.${labelsKey}[${i}]: non-empty string or { value, description?, token? }`);
    else values.push(v);
    if (isObj_(l) && l.description !== undefined && typeof l.description !== "string") p.push(`${at}.${labelsKey}[${i}].description: string`);
    if (isObj_(l) && l.token !== undefined) { pinned++; tokenRef(l.token, `${at}.${labelsKey}[${i}].token`); }
  });
  if (new Set(values).size !== values.length) p.push(`${at}.${labelsKey}: duplicate labels`);
  if (pinned && pinned !== src.length) p.push(`${at}.${labelsKey}: pin a token for every label or for none`);
  if (pinned === src.length) {
    const ids = src.map((l: any) => l.token?.id);
    if (new Set(ids).size !== ids.length) p.push(`${at}.${labelsKey}: pinned token ids must be distinct`);
  }
}

export const REQUIRED_ROLES: Record<RuntimeName, string[]> = {
  genai: ["model", "data"],
  "decode-only": ["model", "data"],
  "transformers.js": ["model"],
  webllm: ["config"],
  litert: ["manifest", "graph", "weights", "embed"],
};

/** Head spec for a question type in a build, if any. */
export function headFor(b: BuildSpec, t: HeadRole): { name: string; spec: HeadSpec } | null {
  for (const [name, spec] of Object.entries(b.heads ?? {})) if (spec.types.includes(t)) return { name, spec };
  return null;
}

/** URL of a pinned file: https://huggingface.co/<repo>/resolve/<revision>/<path> (or under baseUrl). */
export function fileUrl(repo: string, revision: string, path: string, baseUrl?: string): string {
  if (/^https:\/\//.test(path)) return path;
  if (baseUrl) return new URL(path, baseUrl.endsWith("/") ? baseUrl : baseUrl + "/").href;
  return `https://huggingface.co/${repo}/resolve/${revision}/${path}`;
}

/** Validate a media package manifest (no text sections; `modalities` lists "audio"). Throws ManifestError. */
export function validateMediaManifest(m: any): MediaManifest {
  const p: string[] = [];
  if (!m || typeof m !== "object" || Array.isArray(m)) throw new ManifestError(["manifest: expected an object"]);
  if (m.schema !== MANIFEST_SCHEMA) p.push(`schema: unsupported schema id ${JSON.stringify(m.schema)}; expected "${MANIFEST_SCHEMA}"`);
  if (typeof m.id !== "string" || !m.id) p.push("id: non-empty string");
  p.push(...mediaProblems(m, true));
  if (m.provenance !== undefined) provenanceProblems(m.provenance, p);
  if (p.length) throw new ManifestError(p);
  return m as MediaManifest;
}
