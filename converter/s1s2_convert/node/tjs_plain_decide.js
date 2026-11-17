import { AutoModelForCausalLM, AutoTokenizer } from "@huggingface/transformers";
const repo = "__REPO__";
const get = (f) => fetch(`https://huggingface.co/${repo}/resolve/main/${f}`);
const m = await (await get("__MANIFEST__")).json(); // prompt template, markers, head file, option words (lab)
const tok = await AutoTokenizer.from_pretrained(repo);
const model = await AutoModelForCausalLM.from_pretrained(repo, { dtype: "q4" });
const [head] = Object.values(Object.values(m.builds)[0].heads);
const hb = await (await get(head.file.path)).arrayBuffer();
//--decide--
const state = "__STATE__", instructions = "__INSTRUCTIONS__", options = __OPTIONS__;

// 1. Prompt in the manifest's template (s1s2 also escapes the tags <state>, <option>, ... if your text contains them).
const fill = (t, v) => t.replace(/\{(\w+)\}/g, (s, k) => (k in v ? v[k] : s));
const opts = options.map((label, i) => fill(m.encoding.option, { key: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i], label, index: i + 1 }));
const question = fill(m.encoding.question.choice, { instructions, options: opts.join(m.encoding.optionSeparator ?? "") });
const prompt = fill(m.encoding.template, { state, question });

// 2. Read positions: the last token of " <decide>" (query) and of each " </option>" (keys), by prefix tokenization.
const enc = (s) => tok.encode(s, { add_special_tokens: false });
const ends = (mark) => { const r = []; for (let i = prompt.indexOf(mark); i >= 0; i = prompt.indexOf(mark, i + 1)) r.push(i + mark.length); return r; };
const read = [...ends(m.markers.decide.text), ...ends(m.markers.option.text)].map((end) => enc(prompt.slice(0, end)).length - 1);

// 3. One forward pass; `hidden_states` = final hidden state (after the model's last norm), [1, tokens, hidden].
const inputs = tok(prompt, { add_special_tokens: false });
const { hidden_states: H } = await model.forward(inputs);
const D = H.dims.at(-1), h = (p) => Array.from(H.data.subarray(p * D, (p + 1) * D), Number);

// 4. head.bin: uint32 LE JSON length, JSON {headDim, hiddenSize, temperature, score_offset?, scale?}, then fp16 q.weight
//    [headDim x hidden], q.bias, k.weight, k.bias. score_i = scale * (Wq h_decide + bq) . (Wk h_option_i + bk) / T + offset.
const dv = new DataView(hb), n = dv.getUint32(0, true), meta = JSON.parse(new TextDecoder().decode(new Uint8Array(hb, 4, n)));
const f16 = (x) => { const e = (x >> 10) & 31, f = x & 1023, s = x >> 15 ? -1 : 1;
  return e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024); };
let off = 4 + n;
const take = (k) => { const a = new Float32Array(k); for (let i = 0; i < k; i++) a[i] = f16(dv.getUint16(off + 2 * i, true)); off += 2 * k; return a; };
const Dh = meta.headDim, Wq = take(Dh * D), bq = take(Dh), Wk = take(Dh * D), bk = take(Dh);
const scale = meta.scale ?? (meta.scale_folded_into_q !== undefined ? 1 : head.scale ?? 1 / Math.sqrt(Dh));
const proj = (W, b, x) => Array.from(b, (acc, d) => { for (let i = 0; i < D; i++) acc += W[d * D + i] * x[i]; return acc; });
const q = proj(Wq, bq, h(read[0]));
const scores = read.slice(1).map((p) => { const k = proj(Wk, bk, h(p)); let s = 0; for (let i = 0; i < Dh; i++) s += q[i] * k[i];
  return (s * scale) / meta.temperature + (meta.score_offset ?? 0); });

// 5. Softmax over the options -> decision and probability.
const top = Math.max(...scores), ex = scores.map((s) => Math.exp(s - top)), Z = ex.reduce((a, b) => a + b, 0);
const probs = ex.map((x) => x / Z), best = probs.indexOf(Math.max(...probs));
console.log(options[best], probs[best]);
