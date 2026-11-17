// Tokenizer adapter. Any tokenizer exposing encode/decode works (transformers.js
// PreTrainedTokenizer is wrapped by `wrapTokenizer`). No special tokens are added.

export interface Tok {
  encode(text: string): number[];
  decode(ids: number[], opts?: { skipSpecial?: boolean }): string;
}

/** Wrap a transformers.js tokenizer (or anything with encode(text, {add_special_tokens}) / decode). */
export function wrapTokenizer(t: any): Tok {
  if (typeof t?.encode !== "function" || typeof t?.decode !== "function") throw new Error("tokenizer: expected encode() and decode()");
  return {
    encode: (text) => Array.from(t.encode(text, { add_special_tokens: false }) as ArrayLike<number | bigint>, (x) => Number(x)),
    decode: (ids, o = {}) => t.decode(ids, { skip_special_tokens: !!o.skipSpecial, clean_up_tokenization_spaces: false }),
  };
}

/** Build a transformers.js tokenizer from already-verified tokenizer.json / tokenizer_config.json. */
export function tokenizerFromJson(tjs: any, tokenizerJson: any, tokenizerConfig: any): Tok {
  const cls = tjs[tokenizerConfig?.tokenizer_class] ?? tjs[String(tokenizerConfig?.tokenizer_class ?? "").replace(/Fast$/, "")] ?? tjs.PreTrainedTokenizer;
  if (typeof cls !== "function") throw new Error("tokenizer: transformers.js module has no PreTrainedTokenizer");
  return wrapTokenizer(new cls(tokenizerJson, tokenizerConfig));
}

/**
 * ids + per-token character offsets of `text` (ported from ct0-edge positions.ts).
 * Offsets come from incremental decoding; throws if the decode does not reconstruct the text.
 */
export function tokenOffsets(tok: Tok, text: string): { ids: number[]; offsets: [number, number][] } {
  const ids = tok.encode(text);
  const offsets: [number, number][] = [];
  let prev = 0, full = "";
  for (let i = 0; i < ids.length; i++) {
    full = tok.decode(ids.slice(0, i + 1));
    let end = full.length;
    while (end > prev && full.charCodeAt(end - 1) === 0xfffd) end--;
    offsets.push([prev, end]);
    prev = end;
  }
  if (full !== text) throw new Error("tokenOffsets: decode(encode(prompt)) != prompt; character offsets are not reliable for this tokenizer/prompt");
  return { ids, offsets };
}

/** Index of the token covering character `c`. */
export function tokenAt(offsets: [number, number][], c: number): number {
  const i = offsets.findIndex(([s, e]) => s <= c && c < e);
  if (i < 0) throw new Error(`tokenAt: no token covers character ${c}`);
  return i;
}

/** Check that `ref.text` encodes to exactly [ref.id]. */
export function assertTokenRef(tok: Tok, ref: { id: number; text: string }, what: string): void {
  const got = tok.encode(ref.text);
  if (got.length !== 1 || got[0] !== ref.id)
    throw new Error(`manifest/tokenizer mismatch: ${what} ${JSON.stringify(ref.text)} encodes to [${got.join(",")}], manifest says [${ref.id}]`);
}
