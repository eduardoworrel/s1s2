// Prompt encodings and read positions.
//
// "tagged" (generalized from ct0-edge positions.ts): a text template (chat template included)
//   with {state} and {question}; options rendered by a per-option template. Read positions are
//   found STRUCTURALLY (character offsets computed while rendering, never by searching caller
//   text), mapped to tokens, and the token id at every read position is asserted against the
//   manifest marker id.
// "kev" (ported from open-jev src/encoding.ts, MIT, Nico Martin): token-level delimiters
//   <|fim_prefix|> state <|fim_middle|> question (<|box_start|> option <|box_end|>)* <|fim_suffix|>.
//   Each question is its own branch after the shared state prefix.

import type { Manifest, TaggedEncoding, KevEncoding, Marker, QType, TokenRef } from "./manifest.js";
import type { NQ } from "./questions.js";
import { tokenOffsets, tokenAt, assertTokenRef, type Tok } from "./tok.js";

export interface Encoded {
  ids: number[];
  /** Read position of the decide marker (head query vector). */
  decide: number;
  /** Final prompt position (logits readout; head key for predicate/score). */
  last: number;
  /** Read positions of the option markers, one per option, in order. */
  options: number[];
  /** Tokens shared by every question about the same state (upper bound for prefill reuse). */
  prefixLen: number;
  /** Rendered prompt text (tagged encodings): label tokens for the logits readout are taken in its context. */
  text?: string;
}

/** open-jev escape: caller text can never produce a special-token string (<|name|> -> <¦name¦>). */
export function escapeSpecial(text: string): string {
  return text.replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>");
}

/** Fill a template; returns the text and the [start, end) span of the first occurrence of each placeholder. */
export function fill(template: string, values: Record<string, string>): { text: string; spans: Record<string, [number, number]> } {
  let text = "";
  const spans: Record<string, [number, number]> = {};
  const re = /\{([a-z]+)\}/g;
  let last = 0, mt: RegExpExecArray | null;
  while ((mt = re.exec(template))) {
    const key = mt[1];
    if (!(key in values)) continue;
    text += template.slice(last, mt.index);
    const s = text.length;
    text += values[key];
    if (!spans[key]) spans[key] = [s, text.length];
    last = mt.index + mt[0].length;
  }
  text += template.slice(last);
  return { text, spans };
}

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
export function optionKey(i: number): string {
  if (i >= LETTERS.length) throw new Error("letters mapping supports at most 26 options");
  return LETTERS[i];
}

export interface RenderedTagged {
  text: string;
  /** Character index of the last character of the decide marker. */
  decideChar: number;
  /** Character index of the last character of each option marker. */
  optionChars: number[];
  /** Character index where the per-question part starts (end of the shared state prefix). */
  questionStart: number;
}

/** The lab-s12 escape for plain-text markers: each trimmed literal found in `text` gets its first character
 *  replaced by "‹" (all occurrences), in the given order. */
export function escapeLiterals(text: string, literals: readonly string[]): string {
  let t = text;
  for (const raw of literals) {
    const m = raw.trim();
    if (m && t.includes(m)) t = t.split(m).join("‹" + m.slice(1));
  }
  return t;
}

/** Caller-text escape of a tagged encoding (escapeSpecial, then escapeLiterals). */
export function taggedEscaper(enc: TaggedEncoding): (s: string) => string {
  const lits = enc.escapeLiterals ?? [];
  return (s: string) => escapeLiterals(enc.escapeSpecial ? escapeSpecial(s) : s, lits);
}

export function renderTagged(enc: TaggedEncoding, markers: { decide: Marker; option: Marker }, q: NQ, state: string): RenderedTagged {
  const esc = taggedEscaper(enc);
  const optT = enc.option, sep = enc.optionSeparator ?? "";
  // Options: render each, and locate the option marker in the literal text AFTER {label}.
  let options = "";
  const optMarkRel: number[] = [];
  const isChoiceLike = q.type === "choice" || (q.type === "score" && enc.question.score.includes("{options}"));
  if (isChoiceLike) q.texts.forEach((t, i) => {
    if (i) options += sep;
    const o = fill(optT, { label: esc(t), key: optionKey(i), index: String(i + 1) });
    const after = o.text.slice(o.spans.label[1]);
    const k = after.indexOf(markers.option.text);
    if (k < 0) throw new Error(`encoding: option marker ${JSON.stringify(markers.option.text)} not found after {label} in the option template`);
    optMarkRel.push(options.length + o.spans.label[1] + k + markers.option.text.length - 1);
    options += o.text;
  });
  const qt = fill(enc.question[q.type], { instructions: esc(q.instructions), options });
  const optionsAt = qt.spans.options?.[0] ?? 0;
  const full = fill(enc.template, { state: esc(state), question: qt.text });
  const [qs, qe] = full.spans.question;
  const afterQ = full.text.slice(qe);
  const k = afterQ.indexOf(markers.decide.text);
  if (k < 0) throw new Error(`encoding: decide marker ${JSON.stringify(markers.decide.text)} not found after {question} in the template`);
  return {
    text: full.text,
    decideChar: qe + k + markers.decide.text.length - 1,
    optionChars: isChoiceLike ? optMarkRel.map((r) => qs + optionsAt + r) : [],
    questionStart: full.spans.state[1],
  };
}

function checkMarker(ids: number[], pos: number, mk: Marker, what: string): void {
  if (!mk.tokens.some((t) => t.id === ids[pos]))
    throw new Error(`encoding: token at the ${what} read position ${pos} is ${ids[pos]}, manifest marker ${JSON.stringify(mk.text)} accepts ${mk.tokens.map((t) => t.id).join("|")}`);
}

const isPrefix = (a: number[], b: number[]) => a.length <= b.length && a.every((x, i) => x === b[i]);

/**
 * Token position of the text span [a, b) inside the encoding `total` of `text`, by prefix encoding (as the lab-s12
 * prompt builder does): encode(text[:a]) must be a prefix of encode(text[:b]), which must be a prefix of
 * `total`, and the span must add at least one token. Returns the index of the span's last token. Works for any
 * tokenizer (byte-level BPE or SentencePiece), unlike offsets from incremental decoding.
 */
export function spanLastToken(tok: Tok, text: string, total: number[], a: number, b: number, what: string): number {
  const pa = tok.encode(text.slice(0, a)), pb = tok.encode(text.slice(0, b));
  if (!isPrefix(pa, pb) || !isPrefix(pb, total) || pb.length === pa.length)
    throw new Error(`encoding: the ${what} marker (characters ${a}..${b}) does not have a stable token boundary in this prompt (it merges with the neighbouring text)`);
  return pb.length - 1;
}

/** Character offsets of each token when decoding reconstructs the prompt exactly (byte-level BPE); null otherwise
 *  (SentencePiece: decode inserts a '▁' space after special tokens, so offsets from decoding are not reliable). */
function offsetsIfInvertible(tok: Tok, text: string, ids: number[]): [number, number][] | null {
  try {
    const t = tokenOffsets(tok, text);
    return isPrefix(t.ids, ids) && t.ids.length === ids.length ? t.offsets : null;
  } catch { return null; }
}

/**
 * Read positions of a tagged prompt. When decoding reconstructs the prompt, the read token is the token covering the
 * marker's last character (character offsets; it may straddle the next text, e.g. '><'). Otherwise (SentencePiece)
 * positions come from prefix encoding exactly as lab-s12 measures them (spanLastToken), which needs a stable boundary
 * at the marker's end. In both cases the token id at every read position is asserted against the manifest marker.
 */
export function encodeTagged(tok: Tok, m: Manifest, q: NQ, state: string): Encoded {
  const enc = m.encoding as TaggedEncoding;
  const r = renderTagged(enc, m.markers, q, state);
  const ids = tok.encode(r.text);
  const offs = offsetsIfInvertible(tok, r.text, ids);
  const at = (lastChar: number, mk: Marker, what: string) => offs ? tokenAt(offs, lastChar)
    : spanLastToken(tok, r.text, ids, lastChar - mk.text.length + 1, lastChar + 1, what);
  const decide = at(r.decideChar, m.markers.decide, "decide");
  checkMarker(ids, decide, m.markers.decide, "decide");
  const options = r.optionChars.map((c, i) => at(c, m.markers.option, `option ${i}`));
  options.forEach((p, i) => checkMarker(ids, p, m.markers.option, `option ${i}`));
  const last = ids.length - 1;
  if (m.markers.last) checkMarker(ids, last, m.markers.last, "last");
  // Tokens fully before the question part are shared across questions on the same state.
  let prefixLen = 0;
  if (offs) while (prefixLen < offs.length && offs[prefixLen][1] <= r.questionStart) prefixLen++;
  else {
    const pq = tok.encode(r.text.slice(0, r.questionStart));
    while (prefixLen < pq.length && pq[prefixLen] === ids[prefixLen]) prefixLen++;
    if (prefixLen < pq.length && prefixLen > 0) prefixLen--; // the last common token may still merge with the question
  }
  return { ids, decide, last, options, prefixLen, text: r.text };
}

/** KEV encoding of one question branch after the shared state (open-jev encodeKevSequence, one branch). */
export function encodeKev(tok: Tok, m: Manifest, q: NQ, state: string): Encoded {
  const enc = m.encoding as KevEncoding, d = enc.delimiters;
  const e = (s: string) => tok.encode(escapeSpecial(s));
  const ids: number[] = [];
  if (enc.prefix) ids.push(...tok.encode(enc.prefix));
  ids.push(d.state.id, ...e(state));
  const prefixLen = ids.length;
  ids.push(d.question.id, ...e(q.instructions));
  const texts = q.type === "predicate" && m.readout.predicate === "options" ? (enc.predicateOptions ?? ["no", "yes"]) : q.type === "predicate" ? [] : q.texts;
  const options: number[] = [];
  for (const t of texts) { ids.push(d.optionStart.id, ...e(t), d.optionEnd.id); options.push(ids.length - 1); }
  ids.push(d.decide.id);
  const decide = ids.length - 1;
  if (enc.suffix) ids.push(...tok.encode(enc.suffix));
  const last = ids.length - 1;
  if (m.markers.last) checkMarker(ids, last, m.markers.last, "last");
  return { ids, decide, last, options, prefixLen };
}

export function encodeQuestion(tok: Tok, m: Manifest, q: NQ, state: string): Encoded {
  return m.encoding.kind === "kev" ? encodeKev(tok, m, q, state) : encodeTagged(tok, m, q, state);
}

/** Load-time checks of manifest token refs against the tokenizer (fail loudly on mismatch).
 *  EOS, warmup, logits.predicate and kev delimiters must encode alone to exactly their id. Markers of a tagged
 *  encoding are checked where they are read: in the context of a rendered probe prompt (a SentencePiece piece such
 *  as '>' without '▁' only exists after other text, so encoding the marker token alone proves nothing). */
export function assertManifestTokens(tok: Tok, m: Manifest): void {
  for (const t of m.eos) assertTokenRef(tok, t, "eos");
  if (m.warmup) assertTokenRef(tok, m.warmup, "warmup");
  assertTokenRef(tok, m.logits.predicate.yes, "logits.predicate.yes");
  assertTokenRef(tok, m.logits.predicate.no, "logits.predicate.no");
  for (const t of m.logits.score.tokens ?? []) assertTokenRef(tok, t, "logits.score.tokens");
  for (const t of m.logits.choice.tokens ?? []) assertTokenRef(tok, t, `logits.choice.tokens (${JSON.stringify(t.label)})`);
  if (m.encoding.kind === "kev") for (const [k, t] of Object.entries(m.encoding.delimiters)) assertTokenRef(tok, t, `encoding.delimiters.${k}`);
  // Task label tokens are read after the prompt, so (like markers) they are checked by decoding, not by encoding alone.
  for (const t of taskTokens(m)) {
    const d = tok.decode([t.id]);
    if (d !== t.text) throw new Error(`manifest/tokenizer mismatch: ${t.at} token ${t.id} decodes to ${JSON.stringify(d)}, manifest says ${JSON.stringify(t.text)}`);
  }
  for (const k of ["decide", "option", "last"] as const) {
    const mk = m.markers[k];
    if (!mk) continue;
    if (m.encoding.kind === "kev") for (const t of mk.tokens) assertTokenRef(tok, t, `markers.${k}`);
    else for (const t of mk.tokens) {
      const d = tok.decode([t.id]);
      if (d !== t.text) throw new Error(`manifest/tokenizer mismatch: markers.${k} token ${t.id} decodes to ${JSON.stringify(d)}, manifest says ${JSON.stringify(t.text)}`);
    }
    if (mk.at === "token" && (mk.tokens.length !== 1 || mk.tokens[0].text !== mk.text)) throw new Error(`markers.${k}: at "token" needs exactly the token ${JSON.stringify(mk.text)}`);
    if (mk.at === "last-token" && !mk.tokens.some((t) => t.text.includes(mk.text.slice(-1))))
      throw new Error(`markers.${k}: no accepted token contains the marker's last character ${JSON.stringify(mk.text.slice(-1))}`);
  }
  if (m.encoding.kind === "tagged") {
    // Probe prompt: every read position (decide, each option, last) must hold an accepted marker token in context.
    const probe = (q: NQ) => {
      try { encodeTagged(tok, m, q, "Probe state."); }
      catch (e: any) { throw new Error(`manifest/tokenizer mismatch (probe prompt, ${q.type}): ${e?.message ?? e}`); }
    };
    probe({ name: "probe", type: "choice", instructions: "Probe question?", values: ["a", "b"], texts: ["first", "second"] });
    probe({ name: "probe", type: "predicate", instructions: "Probe question?", values: [], texts: [] });
  }
}

/** Every token pinned by a manifest task, with where it was declared. */
export function taskTokens(m: Manifest): (TokenRef & { at: string })[] {
  const out: (TokenRef & { at: string })[] = [];
  for (const [n, t] of Object.entries(m.tasks ?? {})) {
    if (t.tokens) out.push({ ...t.tokens.yes, at: `tasks.${n}.tokens.yes` }, { ...t.tokens.no, at: `tasks.${n}.tokens.no` });
    for (const k of ["choices", "levels"] as const) (t[k] ?? []).forEach((l, i) => { if (typeof l !== "string" && l.token) out.push({ ...l.token, at: `tasks.${n}.${k}[${i}].token` }); });
  }
  return out;
}

/** First token of each label (logits readout); throws when two labels share it.
 *  With `context` (the rendered prompt the logits are read after), the label token is the first token that
 *  encode(context + label) adds after encode(context), as the lab-s12 label contract does; the boundary must be
 *  stable. Without it, the label is encoded alone (kev encodings). */
export function firstTokens(tok: Tok, labels: string[], type: QType, context?: string): number[] {
  const base = context === undefined ? null : tok.encode(context);
  const ids = labels.map((l) => {
    if (base) {
      const b = tok.encode(context + l);
      if (!isPrefix(base, b)) throw new Error(`logits readout: label ${JSON.stringify(l)} merges with the end of the prompt (unstable token boundary)`);
      if (b.length === base.length) throw new Error(`logits readout: label ${JSON.stringify(l)} encodes to no token after the prompt`);
      return b[base.length];
    }
    const t = tok.encode(l);
    if (!t.length) throw new Error(`logits readout: label ${JSON.stringify(l)} encodes to no token`);
    return t[0];
  });
  const seen = new Map<number, string>();
  ids.forEach((id, i) => {
    if (seen.has(id)) throw new Error(`logits readout (${type}): labels ${JSON.stringify(seen.get(id))} and ${JSON.stringify(labels[i])} start with the same token ${id}; the logits cannot tell them apart. Rename the labels or use method "head".`);
    seen.set(id, labels[i]);
  });
  return ids;
}
