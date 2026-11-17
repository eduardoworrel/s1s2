// Joint schema head (spec 0.2 section 6.4), the head of Cloudflare's Clef checkpoints: it reads the final hidden state
// at EVERY prompt token and scores every option of every question of one prompt together (span means of each question
// and option, evidence routing over the prompt, attention between questions, a lexical prior from the LM-head rows of
// each option's tokens). Port of sglang@da614ba python/sglang/srt/layers/joint_schema_head.py
// (JointSchemaHead._forward_group, one request) in plain fp64 JavaScript: no batching, attention over one request
// (the reference's flash-attention call is non-causal within a request), nn.LayerNorm eps 1e-5, exact (erf) GELU.
// Also the prompt layout (encode_joint_schema in systemone/joint_schema.py): pieces tokenized one by one.

import { parseSafetensors, type Tensor } from "./safetensors.js";

export const JOINT_QUESTION_TYPES = ["noul", "choice", "score"] as const;

export interface JointConfig { hidden_size: number; width: number; routing_layers: number; layers: number; heads: number; feedforward: number }
export interface JointHead { kind: "joint"; cfg: JointConfig; t: Record<string, Tensor> }
export interface JointQuestion { type: number; span: [number, number]; options: [number, number][] }

export function loadJointHead(buf: ArrayBuffer, cfg: JointConfig): JointHead {
  const st = parseSafetensors(buf);
  const need = ["hidden_norm.weight", "memory_projection.weight", "question_projection.weight", "residual_scorer.3.weight", "joint_logit_scale"];
  for (const n of need) if (!st.tensors[n]) throw new Error(`joint head: missing tensor ${n}`);
  const H = st.tensors["memory_projection.weight"].shape[1], W = st.tensors["memory_projection.weight"].shape[0];
  if (H !== cfg.hidden_size || W !== cfg.width) throw new Error(`joint head: memory_projection [${W}, ${H}] != config width ${cfg.width} x hidden ${cfg.hidden_size}`);
  if (cfg.width % cfg.heads) throw new Error("joint head: width must be a multiple of heads");
  return { kind: "joint", cfg, t: st.tensors };
}

type Mat = Float64Array[]; // rows

const erf = (x: number): number => {
  // erf to ~1e-15: Taylor series for |x| < 2.5, continued fraction for erfc beyond.
  const ax = Math.abs(x);
  if (ax < 2.5) {
    let term = x, sum = x;
    for (let n = 1; n < 200; n++) { term *= -x * x / n; const add = term / (2 * n + 1); sum += add; if (Math.abs(add) < 1e-17 * Math.abs(sum)) break; }
    return (2 / Math.sqrt(Math.PI)) * sum;
  }
  let f = 0;
  for (let n = 60; n >= 1; n--) f = (n / 2) / (ax + f);
  const erfc = Math.exp(-ax * ax) / Math.sqrt(Math.PI) / (ax + f);
  return x > 0 ? 1 - erfc : erfc - 1;
};
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));

function T(h: JointHead, name: string): Tensor {
  const t = h.t[name];
  if (!t) throw new Error(`joint head: missing tensor ${name}`);
  return t;
}
/** y = W x (+ b), W [out, in] row-major. */
function linear(W: Tensor, x: Float64Array, b?: Tensor, rowOffset = 0, rows = W.shape[0]): Float64Array {
  const n = W.shape[1], y = new Float64Array(rows);
  for (let o = 0; o < rows; o++) {
    let a = b ? b.data[rowOffset + o] : 0;
    const r = (rowOffset + o) * n;
    for (let i = 0; i < n; i++) a += W.data[r + i] * x[i];
    y[o] = a;
  }
  return y;
}
function layerNorm(x: Float64Array, w: Tensor, b: Tensor, eps = 1e-5): Float64Array {
  let m = 0; for (const v of x) m += v; m /= x.length;
  let s = 0; for (const v of x) s += (v - m) * (v - m); s /= x.length;
  const inv = 1 / Math.sqrt(s + eps), y = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = (x[i] - m) * inv * w.data[i] + b.data[i];
  return y;
}
const ln = (h: JointHead, p: string, x: Float64Array) => layerNorm(x, T(h, p + ".weight"), T(h, p + ".bias"));
const add = (a: Float64Array, b: Float64Array) => { const y = new Float64Array(a.length); for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i]; return y; };
const dot = (a: Float64Array, b: Float64Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const l2 = (a: Float64Array, eps = 1e-12) => { const n = Math.max(Math.sqrt(dot(a, a)), eps); return a.map((v) => v / n); };

/** nn.MultiheadAttention (in_proj packed q|k|v, out_proj), keys/values from `keys` (self-attention when null). */
function mha(h: JointHead, p: string, queries: Mat, keys: Mat | null): Mat {
  const W = h.cfg.width, heads = h.cfg.heads, d = W / heads;
  const ip = T(h, p + ".in_proj_weight"), ib = T(h, p + ".in_proj_bias");
  const src = keys ?? queries;
  const Q = queries.map((x) => linear(ip, x, ib, 0, W));
  const K = src.map((x) => linear(ip, x, ib, W, W));
  const V = src.map((x) => linear(ip, x, ib, 2 * W, W));
  const out: Mat = Q.map(() => new Float64Array(W));
  const sc = 1 / Math.sqrt(d);
  for (let hh = 0; hh < heads; hh++) {
    const o0 = hh * d;
    for (let qi = 0; qi < Q.length; qi++) {
      const s = K.map((k) => { let a = 0; for (let j = 0; j < d; j++) a += Q[qi][o0 + j] * k[o0 + j]; return a * sc; });
      const mx = Math.max(...s); let z = 0; const e = s.map((v) => { const x = Math.exp(v - mx); z += x; return x; });
      for (let ki = 0; ki < K.length; ki++) { const w = e[ki] / z; for (let j = 0; j < d; j++) out[qi][o0 + j] += w * V[ki][o0 + j]; }
    }
  }
  return out.map((x) => linear(T(h, p + ".out_proj.weight"), x, T(h, p + ".out_proj.bias")));
}

/**
 * Option logits of one prompt, questions and options in layout order.
 * hidden: final hidden state at every prompt token [T][hidden]; lexicalRows: LM-head (output embedding) row of each
 * prompt token, only needed at option-span tokens (return any row elsewhere, it is not read).
 */
export function jointScores(h: JointHead, hidden: Float32Array[], lexicalRows: (pos: number) => Float32Array, questions: JointQuestion[]): number[][] {
  if (!questions.length) throw new Error("joint head: no questions");
  const n = hidden.length, H = h.cfg.hidden_size, W = h.cfg.width;
  for (const q of questions) {
    if (!(q.type >= 0 && q.type < 3) || !q.options.length) throw new Error("joint head: bad question");
    for (const [a, b] of [q.span, ...q.options]) if (!(0 <= a && a < b && b <= n)) throw new Error(`joint head: span [${a}, ${b}) outside the prompt (${n})`);
  }
  const normalized = hidden.map((x) => { if (x.length !== H) throw new Error(`joint head: hidden ${x.length} != ${H}`); return ln(h, "hidden_norm", Float64Array.from(x)); });
  const mean = (rows: (i: number) => ArrayLike<number>, [a, b]: [number, number], dim: number) => {
    const s = new Float64Array(dim);
    for (let i = a; i < b; i++) { const r = rows(i); for (let j = 0; j < dim; j++) s[j] += r[j]; }
    return s.map((v) => v / (b - a));
  };
  const memory = normalized.map((x) => linear(T(h, "memory_projection.weight"), x));
  const global = normalized[n - 1];
  const qv = questions.map((q) => mean((i) => normalized[i], q.span, H));
  const owner: number[] = [], spans: [number, number][] = [];
  questions.forEach((q, qi) => q.options.forEach((s) => { owner.push(qi); spans.push(s); }));
  const oc = spans.map((s) => mean((i) => normalized[i], s, H));
  const lexDim = lexicalRows(spans[0][0]).length;
  const lex = spans.map((s) => mean(lexicalRows, s, lexDim));
  const oqp = qv.map((x) => linear(T(h, "option_question_projection.weight"), x));
  let routed: Mat = spans.map((_, o) => add(add(linear(T(h, "option_context_projection.weight"), oc[o]), linear(T(h, "option_lexical_projection.weight"), lex[o])), oqp[owner[o]]));
  for (let l = 0; l < h.cfg.routing_layers; l++) {
    const p = `evidence_layers.${l}`;
    const att = mha(h, p + ".attention", routed.map((x) => ln(h, p + ".query_norm", x)), memory.map((x) => ln(h, p + ".memory_norm", x)));
    routed = routed.map((x, i) => add(x, att[i]));
    routed = routed.map((x) => {
      const a = linear(T(h, p + ".feedforward.0.weight"), ln(h, p + ".feedforward_norm", x), T(h, p + ".feedforward.0.bias")).map(gelu);
      return add(x, linear(T(h, p + ".feedforward.3.weight"), a, T(h, p + ".feedforward.3.bias")));
    });
  }
  const base = qv.map((x) => linear(T(h, "question_projection.weight"), x));
  const scores = routed.map((x, o) => dot(x, base[owner[o]]) / Math.sqrt(W));
  const summaries: Mat = questions.map(() => new Float64Array(W));
  questions.forEach((_, qi) => {
    const idx = owner.map((ow, o) => (ow === qi ? o : -1)).filter((o) => o >= 0);
    const mx = Math.max(...idx.map((o) => scores[o])); let z = 0;
    const e = idx.map((o) => { const v = Math.exp(scores[o] - mx); z += v; return v; });
    idx.forEach((o, k) => { for (let j = 0; j < W; j++) summaries[qi][j] += (e[k] / z) * routed[o][j]; });
  });
  const gp = linear(T(h, "global_projection.weight"), global), te = T(h, "type_embedding.weight");
  let fields: Mat = questions.map((q, qi) => {
    const s = ln(h, "option_summary_norm", summaries[qi]), y = new Float64Array(W);
    for (let j = 0; j < W; j++) y[j] = base[qi][j] + s[j] + gp[j] + te.data[q.type * W + j];
    return y;
  });
  for (let l = 0; l < h.cfg.layers; l++) {
    const p = `layers.${l}`;
    const sa = mha(h, p + ".self_attn", fields.map((x) => ln(h, p + ".norm1", x)), null);
    fields = fields.map((x, i) => add(x, sa[i]));
    const ca = mha(h, p + ".multihead_attn", fields.map((x) => ln(h, p + ".norm2", x)), memory);
    fields = fields.map((x, i) => add(x, ca[i]));
    fields = fields.map((x) => add(x, linear(T(h, p + ".linear2.weight"), linear(T(h, p + ".linear1.weight"), ln(h, p + ".norm3", x), T(h, p + ".linear1.bias")).map(gelu), T(h, p + ".linear2.bias"))));
  }
  fields = fields.map((x) => ln(h, "field_norm", x));
  const cap = Math.log(100);
  const priorScale = Math.exp(Math.min(T(h, "prior_logit_scale").data[0], cap));
  const jointScale = Math.exp(Math.min(T(h, "joint_logit_scale").data[0], cap));
  const gate = 1 / (1 + Math.exp(-T(h, "residual_gate").data[0]));
  const anchor = qv.map((x) => l2(add(x, global)));
  const out: number[][] = questions.map(() => []);
  spans.forEach((_, o) => {
    const prior = priorScale * dot(l2(lex[o]), anchor[owner[o]]);
    const opt = ln(h, "option_norm", routed[o]), rep = fields[owner[o]];
    const cos = dot(rep, opt) / Math.max(Math.sqrt(dot(rep, rep)) * Math.sqrt(dot(opt, opt)), 1e-8);
    const feat = new Float64Array(4 * W);
    for (let j = 0; j < W; j++) { feat[j] = rep[j]; feat[W + j] = opt[j]; feat[2 * W + j] = rep[j] * opt[j]; feat[3 * W + j] = Math.abs(rep[j] - opt[j]); }
    const hid = linear(T(h, "residual_scorer.0.weight"), feat, T(h, "residual_scorer.0.bias")).map(gelu);
    const residual = linear(T(h, "residual_scorer.3.weight"), hid, T(h, "residual_scorer.3.bias"))[0];
    out[owner[o]].push(prior + gate * (jointScale * cos + residual));
  });
  return out;
}

export interface JointPromptSpec {
  /** Pieces of encode_joint_schema; defaults are the Clef strings of sglang@da614ba systemone/joint_schema.py. */
  prefix: string; suffix: string; schemaHeader: string; noulCriteria: { true: string | null; false: string | null };
}
export const CLEF_PROMPT: JointPromptSpec = {
  prefix: "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n",
  suffix: "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
  schemaHeader: "\n\nSCHEMA FIELDS:\n",
  noulCriteria: { true: "The proposition is true or the answer is yes.", false: "The proposition is false or the answer is no." },
};

export interface JointRequestQuestion { id: string; type: "noul" | "choice" | "score"; instructions?: unknown; criteria?: any }

/** JSON as Python json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True). */
export function pyJson(v: any): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "string") return JSON.stringify(v);
  if (typeof v === "number" || typeof v === "boolean") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(pyJson).join(",") + "]";
  return "{" + Object.keys(v).sort().map((k) => JSON.stringify(k) + ":" + pyJson(v[k])).join(",") + "}";
}
const render = (v: unknown) => (typeof v === "string" ? v : pyJson(v));

/** Options of a question in the order the head scores them (joint_schema_options). */
export function jointOptions(q: JointRequestQuestion, spec: JointPromptSpec = CLEF_PROMPT): [string, unknown][] {
  if (q.type === "noul") {
    const c: Record<string, unknown> = { ...spec.noulCriteria, ...(q.criteria ?? {}) };
    return [["true", c.true], ["false", c.false]];
  }
  if (q.type === "choice") return Object.entries<unknown>(q.criteria).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return (q.criteria as unknown[]).map((d, i) => [String(i), d]);
}

/** Prompt ids + layout (encode_joint_schema, text only, no truncation): each piece encoded on its own. */
export function encodeJointSchema(encode: (s: string) => number[], state: unknown, questions: JointRequestQuestion[], spec: JointPromptSpec = CLEF_PROMPT): { ids: number[]; questions: JointQuestion[] } {
  let schema = encode(spec.schemaHeader);
  const qs: JointQuestion[] = [];
  questions.forEach((q, index) => {
    schema = schema.concat(encode(`\nFIELD ${index + 1}\nID: ${q.id}\nTYPE: ${q.type}\nINSTRUCTION: `));
    const instr = q.instructions === undefined || q.instructions === null || q.instructions === "" ? q.id : q.instructions;
    const qs0 = schema.length; schema = schema.concat(encode(render(instr)));
    const span: [number, number] = [qs0, schema.length];
    schema = schema.concat(encode("\nALLOWED OPTIONS:\n"));
    const options: [number, number][] = [];
    jointOptions(q, spec).forEach(([name, desc], oi) => {
      schema = schema.concat(encode(`OPTION ${oi + 1}: `));
      const o0 = schema.length;
      const sem: Record<string, unknown> = { option_id: name };
      if (desc !== null && desc !== undefined) sem.description = desc;
      schema = schema.concat(encode(render(sem)));
      options.push([o0, schema.length]);
      schema = schema.concat(encode("\n"));
    });
    schema = schema.concat(encode("END FIELD\n"));
    qs.push({ type: JOINT_QUESTION_TYPES.indexOf(q.type), span, options });
  });
  const prefix = encode(spec.prefix), st = encode(render(state)), suffix = encode(spec.suffix);
  const off = prefix.length + st.length;
  return { ids: [...prefix, ...st, ...schema, ...suffix], questions: qs.map((q) => ({ type: q.type, span: [q.span[0] + off, q.span[1] + off], options: q.options.map(([a, b]) => [a + off, b + off] as [number, number]) })) };
}
