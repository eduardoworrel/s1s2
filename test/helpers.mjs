// Shared test helpers. The real CT0 tokenizer is fetched once (pinned revision, sha256-checked)
// into the lib cache; tests that need it skip when it is unavailable (offline).
import { readFileSync } from "node:fs";
import { Files, tokenizerFromJson, validateManifest } from "../dist/index.js";

export const CT0 = validateManifest(JSON.parse(readFileSync(new URL("../manifests/ct0-alias3-micro-1026.json", import.meta.url), "utf8")));
export const clone = (x) => JSON.parse(JSON.stringify(x));
export const fixture = (n) => JSON.parse(readFileSync(new URL(`./fixtures/${n}`, import.meta.url), "utf8"));

let tokP;
export function ct0Tokenizer() {
  tokP ??= (async () => {
    try {
      const tjs = await import("@huggingface/transformers");
      const files = new Files({ platform: "node" });
      const t = CT0.tokenizer;
      return tokenizerFromJson(tjs, await files.json(t.repo, t.revision, t.files.tokenizer), await files.json(t.repo, t.revision, t.files.config));
    } catch (e) { console.log("# tokenizer unavailable:", e?.message ?? e); return null; }
  })();
  return tokP;
}

// Toy SentencePiece-like tokenizer (test/sentencepiece.test.mjs, test/embeddings.test.mjs): '▁' is prepended at the start
// of the text and after every special token; greedy longest match; decode keeps the '▁' after specials (not invertible).
export const TOY_SPECIALS = ["<s>", "<|im_start|>", "<|im_end|>"];
export const TOY_PIECES = [...TOY_SPECIALS, "▁<", "▁</", "decide", "option", "question", "state", "▁Yes", "▁No", "▁Answer", "▁",
  ...Array.from({ length: 95 }, (_, i) => String.fromCharCode(32 + i)).filter((c) => c !== " "), "\n", "▁>", "‹", "¦"];
export const TOY_ID = new Map(TOY_PIECES.map((p, i) => [p, i]));

/** Greedy longest-match pieces; '▁' prefix at the start of the text and after every special token (legacy SP). */
export const toy = {
  encode(text) {
    const out = [];
    let i = 0, seg = "", fresh = true;
    const flush = () => {
      if (!seg) return;
      let s = (fresh ? "▁" : "") + seg.replace(/ /g, "▁");
      while (s.length) {
        let k = s.length;
        while (k > 0 && !TOY_ID.has(s.slice(0, k))) k--;
        if (!k) throw new Error(`toy: no piece for ${JSON.stringify(s[0])}`);
        out.push(TOY_ID.get(s.slice(0, k))); s = s.slice(k);
      }
      seg = ""; fresh = false;
    };
    while (i < text.length) {
      const sp = TOY_SPECIALS.find((x) => text.startsWith(x, i));
      if (sp) { flush(); out.push(TOY_ID.get(sp)); i += sp.length; fresh = true; continue; }
      seg += text[i++];
    }
    flush();
    return out;
  },
  decode(ids) {
    const s = ids.map((i) => TOY_PIECES[i]).join("").replace(/▁/g, " ");
    return s.startsWith(" ") ? s.slice(1) : s;
  },
};

export function toyManifest() {
  const m = clone(JSON.parse(readFileSync(new URL("../manifests/smollm2-135m-instruct.en.json", import.meta.url), "utf8")));
  m.encoding.template = m.encoding.template.replace("You are a helpful AI assistant named SmolLM, trained by Hugging Face", "");
  m.generation.turn = m.generation.turn.replace("You are a helpful AI assistant named SmolLM, trained by Hugging Face", "");
  m.markers.decide.tokens = [{ id: TOY_ID.get(">"), text: ">" }];
  m.markers.option.tokens = [{ id: TOY_ID.get(">"), text: ">" }];
  m.eos = [{ id: TOY_ID.get("<|im_end|>"), text: "<|im_end|>" }];
  m.warmup = { id: TOY_ID.get("<s>"), text: "<s>" };
  m.logits.predicate = { yes: { id: TOY_ID.get("▁Yes"), text: "Yes" }, no: { id: TOY_ID.get("▁No"), text: "No" } };
  m.logits.choice = { map: "tokens", tokens: [{ label: " Yes", ...m.logits.predicate.yes }, { label: " No", ...m.logits.predicate.no }] };
  return m;
}
