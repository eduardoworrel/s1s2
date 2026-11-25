// Decoder questions of the audio package (language, term): text utilities against fixtures made by the Python originals,
// float16 rows, the engine (greedy, beam, language scores) on a small deterministic model, manifest rules.
// Real-model parity (Node CPU, Chrome) is recipes/marimba-audio/decoder-parity.mjs and test/browser/media-chrome.mjs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateMediaManifest, FIX } from "./media-helpers.mjs";
import { clone } from "./helpers.mjs";
import { schemaProblems } from "../dist/package.js";
import { normalizeTerm, termFraction, detectAndFixRepetitions, parseAsrText, decodeIds, f32ToF16Bits, DecoderEngine, AudioModel } from "../dist/media.js";
import path from "node:path";

const FX = JSON.parse(readFileSync(path.join(FIX, "decoder-text.json"), "utf8"));
const MANIFEST = JSON.parse(readFileSync(new URL("../manifests/marimba-audio.json", import.meta.url), "utf8"));

test("normalize equals the Python original on transcripts, hypotheses and special spellings", () => {
  const bad = FX.normalize.filter(([a, b]) => normalizeTerm(a) !== b);
  assert.deepEqual(bad, []);
});

test("normalize equals the Python original for every code point the fixture lists (letters, digits, marks, Unicode 14)", () => {
  const bad = [];
  for (const [h, v] of Object.entries(FX.codepoints)) { const got = normalizeTerm(String.fromCodePoint(parseInt(h, 16))); if (got !== v) bad.push([h, got, v]); }
  assert.deepEqual(bad.slice(0, 10), []);
  assert.ok(Object.keys(FX.codepoints).length > 20000);
});

test("repetition collapse and parse of the model output equal the Python originals", () => {
  assert.deepEqual(FX.repetitions.filter(([a, b]) => detectAndFixRepetitions(a) !== b), []);
  assert.deepEqual(FX.parse.filter(([a, b]) => parseAsrText(a) !== b), []);
  assert.ok(FX.repetitions.length > 100 && FX.parse.length > 30);
});

test("byte-level decoding of ids equals the tokenizer (specials skipped, cut multi-byte sequences replaced)", () => {
  const tokens = Array.from({ length: 151936 }, (_, i) => FX.pieces[i] ?? "");
  const t = { tokens, added: FX.added, special: FX.special };
  assert.deepEqual(FX.decode.filter(([ids, s]) => decodeIds(ids, t) !== s), []);
});

test("term fraction equals exact_fraction (whole words after normalisation)", () => {
  for (const [term, hyps, f] of FX.fractions) {
    if (f === null) assert.throws(() => termFraction(term, hyps));
    else assert.equal(termFraction(term, hyps), f);
  }
  assert.equal(termFraction("ola", ["Olá, mundo", "bolas", "ola"]), 2 / 3);
  assert.equal(termFraction("sao joao", ["São João chegou"]), 1);
  assert.equal(termFraction("jo", ["São João chegou"]), 0);
});

test("float16 rounding of the audio rows is round-to-nearest-even like the hardware conversion", (t) => {
  const F16 = globalThis.Float16Array;
  if (!F16) return t.skip("Float16Array needs Node 24+ (the rounding is also checked against Math.f16round / the exhaustive half table in media-audio.test.mjs)");
  const f = new F16(1), u = new Uint16Array(f.buffer);
  let seed = 12345; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < 200000; i++) { const x = Math.fround((rnd() * 2 - 1) * 2 ** Math.floor(rnd() * 40 - 30)); f[0] = x; assert.equal(f32ToF16Bits(x), u[0], String(x)); }
  for (const x of [0, -0, 1, 65504, 65519, 65520, 6e-8, 3e-8, 5.96e-8, 6.1e-5, 6.0e-5, 1e-9]) { f[0] = x; assert.equal(f32ToF16Bits(x), u[0], String(x)); }
});

// ---- the engine on a small deterministic language model ---------------------------------------------------------------------------------

const H = 4, V = 12, EOS = [10, 11];
/** next-token log-probabilities as a function of the last token only (a Markov chain): fixed, with a spread of probabilities. */
function chain(seed) {
  let s = seed; const r = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296);
  return Array.from({ length: V }, (_, last) => { const w = Array.from({ length: V }, (_, t) => Math.exp(3 * r()) * (t >= 10 ? 0.9 + last / 8 : 1)); const z = w.reduce((a, b) => a + b); return w.map((x) => Math.log(x / z)); });
}
function fakeBackend(T) {
  const logits = (last) => Float32Array.from(T[last]);
  return {
    hidden: H, vocab: V,
    empty: () => ({ tensors: {}, past: 0, last: 0 }),
    async feed(rows, kv, wantLogits) {
      const S = rows.length / H, last = rows[(S - 1) * H];       // the first value of a row carries the token id
      return { kv: { tensors: {}, past: kv.past + S, last }, logits: wantLogits ? logits(last) : undefined };
    },
    dispose() {}, async close() {},
  };
}
const table = { hidden: H, vocab: V, async rows(ids) { const o = new Uint16Array(ids.length * H); ids.forEach((id, i) => { o[i * H] = id; }); return o; } };
const spec = (extra = {}) => ({
  runtime: "onnx-genai", model: { layers: 1, kvHeads: 1, headDim: H, hidden: H, vocab: V },
  prompt: { before: [1, 2], after: [3, 4], audioFrames: 3, languageToken: 5 }, eos: EOS,
  languages: { aa: { name: "Aa", ids: [6] }, bb: { name: "Bb", ids: [7] }, cc: { name: "Cc", ids: [8, 9] } },
  term: { beams: 4, maxNewTokens: 12, lengthPenalty: 1, repetitionThreshold: 20 }, devices: ["cpu"], ...extra,
});
const tokens = { tokens: Array.from({ length: V }, (_, i) => (i < 10 ? "abcdefghij"[i] : "")), added: [], special: EOS };
const engine = (T, extra) => new DecoderEngine(fakeBackend(T), table, spec(extra), tokens, { minBlock: 0, prefixRows: 3 });
const audio = () => { const a = new Float32Array(3 * H); a[0] = 2; a[H] = 2; a[2 * H] = 2; return a; };   // first value 2 = token id carried by the audio rows

test("language: one-token names read the row after 'language'; a two-token name adds the next row (full log-softmax each)", async () => {
  const T = chain(7), e = engine(T), w = await e.open(audio());
  const { codes, scores } = await w.language();
  assert.deepEqual(codes, ["aa", "bb", "cc"]);
  assert.ok(Math.abs(scores[0] - T[5][6]) < 1e-6 && Math.abs(scores[1] - T[5][7]) < 1e-6);
  assert.ok(Math.abs(scores[2] - (T[5][8] + T[8][9])) < 1e-6);
  const sub = await w.language(["bb", "cc"]);
  assert.deepEqual(sub.codes, ["bb", "cc"]); assert.ok(Math.abs(sub.scores[0] - scores[1]) < 1e-9);
  await assert.rejects(() => w.language(["zz"]), /unsupported code/);
});

test("greedy follows the arg max until an EOS and stops at the limit", async () => {
  const T = chain(11), e = engine(T), w = await e.open(audio());
  const g = await w.greedy();
  let last = 4, want = [];     // the prompt ends with token 4
  for (;;) { const t = T[last].indexOf(Math.max(...T[last])); want.push(t); last = t; if (EOS.includes(t) || want.length >= 12) break; }
  assert.deepEqual(g, want);
});

/** All finished sequences of length <= max by exhaustive search, scored sum log-prob / length (length penalty 1). */
function exhaustive(T, max) {
  const out = [];
  (function go(seq, lp, last) {
    for (let t = 0; t < V; t++) {
      const s = [...seq, t], l = lp + T[last][t];
      if (EOS.includes(t) || s.length >= max) out.push({ seq: s, score: l / s.length }); else go(s, l, t);
    }
  })([], 0, 4);
  return out.sort((a, b) => b.score - a.score);
}

test("beam: returns the 4 best finished sequences, best first, and the best one is the exhaustive optimum", async () => {
  for (const seed of [1, 2, 3, 5, 8, 13]) {
    const MAX = 5, T = chain(seed), e = engine(T, { term: { beams: 4, maxNewTokens: MAX, lengthPenalty: 1, repetitionThreshold: 20 } }), w = await e.open(audio());
    const beams = await w.beam();
    assert.equal(beams.length, 4);
    const score = (seq) => seq.reduce((a, t, i) => a + T[i === 0 ? 4 : seq[i - 1]][t], 0) / seq.length;
    for (let i = 1; i < 4; i++) assert.ok(score(beams[i - 1]) >= score(beams[i]) - 1e-12, `sorted (seed ${seed})`);
    for (const b of beams) assert.ok(EOS.includes(b.at(-1)) || b.length >= MAX, "finished");
    assert.deepEqual(beams[0], exhaustive(T, MAX)[0].seq, `optimum (seed ${seed})`);
  }
});

test("beam with a length penalty other than 1: still returns four finished sequences, best first by the penalised score", async () => {
  for (const lengthPenalty of [0.6, 2]) for (const seed of [4, 9]) {
    const MAX = 6, T = chain(seed), e = engine(T, { term: { beams: 4, maxNewTokens: MAX, lengthPenalty, repetitionThreshold: 20 } }), w = await e.open(audio());
    const beams = await w.beam();
    assert.equal(beams.length, 4);
    const score = (seq) => seq.reduce((a, t, i) => a + T[i === 0 ? 4 : seq[i - 1]][t], 0) / seq.length ** lengthPenalty;
    for (let i = 1; i < 4; i++) assert.ok(score(beams[i - 1]) >= score(beams[i]) - 1e-12, `sorted (lp ${lengthPenalty}, seed ${seed})`);
    for (const b of beams) assert.ok(EOS.includes(b.at(-1)) || b.length >= MAX, "finished");
  }
});

test("the repetition threshold of the manifest is the one the transcript is cleaned with; a BOM inside a piece is kept", async () => {
  const toks = { tokens: ["a", "b", "x", "y", "", "", "", "", "", "", "", ""], added: [], special: EOS };
  const ids = Array(25).fill(0);
  const e20 = engine(chain(1), {}), e5 = engine(chain(1), { term: { beams: 4, maxNewTokens: 12, lengthPenalty: 1, repetitionThreshold: 5 } });
  const mk = (e) => new DecoderEngine(e.be, table, e.spec, toks, { minBlock: 0, prefixRows: 3 });
  assert.equal(mk(e20).decode(ids), "a".repeat(25).replace(/^a+$/, "a"));       // 25 > 20: collapsed to one
  assert.equal(mk(e5).decode(Array(4).fill(0)), "aaaa");                          // 4 <= 5: kept
  assert.equal(mk(e5).decode(Array(6).fill(0)), "a");                             // 6 > 5: collapsed (with the default 20 it would stay)
  assert.equal(detectAndFixRepetitions("a".repeat(6)), "aaaaaa");
  // the byte-level pieces of EF BB BF (three characters of the byte table) decode to U+FEFF, which Python keeps
  assert.equal(decodeIds([0, 1], { tokens: ["\u00ef\u00bb\u00bf", "a"], added: [], special: [] }), "\ufeffa");
});

test("hypotheses: greedy first, then four beams, decoded and parsed", async () => {
  const T = chain(3), e = engine(T), w = await e.open(audio());
  const h = await w.hypotheses();
  assert.equal(h.length, 5);
  assert.equal(h[0], (await w.greedy()).filter((t) => t < 10).map((t) => "abcdefghij"[t]).join(""));
});

// ---- manifest ---------------------------------------------------------------------------------------------------------------------------

const problems = (m) => { try { validateMediaManifest(m); return []; } catch (e) { assert.equal(e.name, "ManifestError"); return e.problems; } };

test("the manifest lists the decoder, validates, and its rules catch broken blocks", async () => {
  assert.ok(MANIFEST.audio.decoder, "decoder block");
  assert.deepEqual(problems(MANIFEST), []);
  assert.deepEqual(await schemaProblems(MANIFEST), []);
  assert.equal(MANIFEST.audio.decoder.embeddings.size, MANIFEST.audio.decoder.model.vocab * MANIFEST.audio.decoder.model.hidden * 2);
  const m1 = clone(MANIFEST); delete m1.audio.decoder;
  assert.ok(problems(m1).some((x) => /needs audio\.decoder/.test(x)));
  const m2 = clone(MANIFEST); m2.audio.readers.term.tap = "h9";
  assert.ok(problems(m2).some((x) => /must be 104 x 1024/.test(x)));
  const m3 = clone(MANIFEST); m3.audio.decoder.languages.pt.ids = [1, 2, 3];
  assert.ok(problems(m3).some((x) => /languages\.pt/.test(x)));
  const m4 = clone(MANIFEST); delete m4.audio.decoder.cpuBuild;
  assert.ok(problems(m4).some((x) => /need cpuBuild/.test(x)));
  const m6 = clone(MANIFEST); m6.audio.readers.term.tap = "h18";
  assert.ok(problems(m6).some((x) => /must be 104 x 1024|same tap/.test(x)));
  const m5 = clone(MANIFEST); m5.audio.decoder.embeddings.size = 1000;
  assert.ok(problems(m5).some((x) => /vocab \* hidden \* 2/.test(x)));
});

test("a request without a term, or with one that has no letters, is refused before anything runs", async () => {
  const fake = { tower: async () => ({}), speech: async () => 0, dispose: async () => {}, backend: "fake", device: "cpu" };
  const m = clone(MANIFEST);
  const T = chain(2), e = new DecoderEngine(fakeBackend(T), table, spec(), tokens, { minBlock: 0, prefixRows: 3 });
  m.audio.decoder.languages = spec().languages;
  const weights = {}; // pooled readers need weights; drop them for this check
  for (const k of ["turn_end", "same_speaker", "speaker_change", "speaker_count", "voice_activity"]) delete m.audio.readers[k];
  const model = new AudioModel({ spec: m.audio, weights, questions: {}, runtime: fake, decoder: e });
  await assert.rejects(() => model.decodeFeatures(audio(), [{ task: "term" }]), /needs \{ task, term \}/);
  await assert.rejects(() => model.decodeFeatures(audio(), [{ task: "term", term: "!!!" }]), /needs \{ task, term \}/);
  await assert.rejects(() => model.decodeFeatures(audio(), [{ task: "language", options: ["pt"] }]), /at least two distinct/);
  await assert.rejects(() => model.decodeFeatures(audio(), [{ task: "language", options: ["aa", "constructor"] }]), /not codes of this package/);
  const a = await model.decodeFeatures(audio(), [{ task: "language", options: ["aa", "bb"] }, { task: "term", term: "a" }, { task: "term", term: "zz" }]);
  assert.equal(a[0].modality, "audio"); assert.equal(a[0].type, "choice"); assert.deepEqual(a[0].probabilities.map((x) => x.value), ["aa", "bb"]);
  assert.ok(Math.abs(a[0].probabilities.reduce((s, x) => s + x.probability, 0) - 1) < 1e-12);
  assert.equal(a[1].type, "predicate"); assert.equal(a[1].hypotheses.length, 5); assert.equal(a[1].threshold, 0.5);
  assert.equal(a[2].probability, 0); assert.equal(a[2].decision, "no");
});
