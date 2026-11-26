// AudioModel.transcribe: free transcription over the decoder, on a scripted decoder that replays token ids of real transcripts
// (the "decode" fixtures made by the Python tokenizer: id lists and the text they decode to), and the parse of "language X<asr_text>..." against the
// "parse" fixtures made by parse_asr_output. The int4 text of the real package is checked by recipes/marimba-audio/transcribe-parity.mjs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { FIX } from "./media-helpers.mjs";
import { clone } from "./helpers.mjs";
import { DecoderEngine, AudioModel, parseAsr, parseAsrText } from "../dist/media.js";

const FX = JSON.parse(readFileSync(path.join(FIX, "decoder-text.json"), "utf8"));
const MANIFEST = JSON.parse(readFileSync(new URL("../manifests/marimba-audio.json", import.meta.url), "utf8"));
const D = MANIFEST.audio.decoder, H = 4, V = D.model.vocab, PROMPT = D.prompt.before.length + D.prompt.audioFrames + D.prompt.after.length, EOS = D.eos;
const TOKENS = { tokens: Array.from({ length: V }, (_, i) => FX.pieces[i] ?? ""), added: FX.added, special: FX.special };
const table = { hidden: H, vocab: V, async rows(ids) { return new Uint16Array(ids.length * H); } };

/** A decoder that says `seq[k]` at step k whatever it was fed (the position, not the content, picks the token); EOS after the end. */
function scripted(seq, { onFeed, delay = 0 } = {}) {
  const st = { feeds: 0, closed: false, closedWhileActive: false, active: 0, disposed: 0 };
  const logits = (step) => { const l = new Float32Array(V); l[step < seq.length ? seq[step] : EOS[1]] = 40; return l; };
  return Object.assign(st, {
    hidden: H, vocab: V,
    empty: () => ({ tensors: {}, past: 0 }),
    async feed(rows, kv, want) {
      st.feeds++; st.active++;
      try {
        onFeed?.(st);
        if (delay) await new Promise((r) => setTimeout(r, delay));
        const past = kv.past + rows.length / H;
        return { kv: { tensors: {}, past }, logits: want ? logits(past - PROMPT) : undefined };
      } finally { st.active--; }
    },
    dispose() { st.disposed++; },
    async close() { if (st.active) st.closedWhileActive = true; st.closed = true; },
  });
}

/** An AudioModel with only the decoder readers, over a tower that returns constant vectors (calls counted). */
function model(seq, o = {}, specEdit) {
  const be = scripted(seq, o), tower = { calls: 0 };
  const m = clone(MANIFEST);
  for (const k of ["voice_activity", "turn_end", "same_speaker", "speaker_change", "speaker_count"]) delete m.audio.readers[k];
  specEdit?.(m.audio);
  const engine = new DecoderEngine(be, table, m.audio.decoder, TOKENS, { minBlock: 0 });
  const runtime = { backend: "fake", device: "cpu", async tower() { tower.calls++; return { audio_features: new Float32Array(D.prompt.audioFrames * H).fill(0.1) }; }, speech: async () => 0, dispose: async () => {} };
  return { be, tower, model: new AudioModel({ spec: m.audio, weights: {}, questions: {}, runtime, decoder: engine }) };
}
const audio = (seconds) => ({ samples: Float32Array.from({ length: Math.round(seconds * 16000) }, (_, i) => 0.2 * Math.sin(i / 7)), sampleRate: 16000 });
const PT = FX.decode.filter(([ids, s]) => s.startsWith("language Portuguese<asr_text>") && ids.at(-1) === 151645);

test("fixtures: parseAsr gives the text of parse_asr_output on every fixture, and parseAsrText is its text", () => {
  assert.ok(FX.parse.length > 30);
  assert.deepEqual(FX.parse.filter(([raw, text]) => parseAsr(raw).text !== text || parseAsrText(raw) !== text), []);
});

test("parseAsr: the language name before the tag, '' when there is none, 'None', or a line that does not start with 'language '", () => {
  assert.equal(parseAsr("language Portuguese<asr_text>Olá").language, "Portuguese");
  assert.equal(parseAsr("  language English\n<asr_text>two\nlines").language, "English");
  assert.equal(parseAsr("language None<asr_text>algo").language, "");
  assert.equal(parseAsr("sem etiqueta").language, "");
  assert.equal(parseAsr("<asr_text>só texto").language, "");
  assert.equal(parseAsr("lang x<asr_text>y").language, "");
  assert.equal(parseAsr("").language, "");
});

test("greedy: the text without the 'language' prefix, the language as a code, the token count, 'stop' (every Portuguese fixture)", async () => {
  assert.ok(PT.length > 20);
  for (const [ids, raw] of PT) {
    const { model: m } = model(ids);
    const r = await m.transcribe(audio(8));
    assert.equal(r.text, parseAsrText(raw), raw);
    assert.equal(r.language, "pt");
    assert.equal(r.method, "greedy"); assert.equal(r.hypotheses, undefined);
    assert.equal(r.tokens, ids.length - 1); assert.equal(r.finishReason, "stop");
    assert.ok(!/language|asr_text/.test(r.text));
    await m.dispose();
  }
});

test("beam: the best beam is the text, hypotheses are the four beams best first, and the same call twice gives the same", async () => {
  const [ids, raw] = PT[0];
  const { model: m } = model(ids);
  const r = await m.transcribe(audio(8), { beam: true });
  assert.equal(r.method, "beam"); assert.equal(r.hypotheses.length, 4);
  assert.equal(r.text, parseAsrText(raw)); assert.equal(r.hypotheses[0], r.text);
  assert.equal(r.language, "pt"); assert.equal(r.finishReason, "stop");
  const again = await m.transcribe(audio(8), { beam: true });
  assert.deepEqual({ ...again, cost: 0 }, { ...r, cost: 0 });
  await m.dispose();
});

test("text without a prefix: language is null and the text is what was said", async () => {
  const [ids, raw] = PT[0], bare = ids.slice(ids.indexOf(151704) + 1);      // the ids after the <asr_text> token
  const { model: m } = model(bare);
  const r = await m.transcribe(audio(8));
  assert.equal(r.language, null); assert.equal(r.text, parseAsrText(raw)); assert.equal(r.tokens, bare.length - 1);
  const { model: empty } = model([EOS[0]]);
  const e = await empty.transcribe(audio(8));
  assert.deepEqual([e.text, e.language, e.tokens, e.finishReason], ["", null, 0, "stop"]);
  await m.dispose(); await empty.dispose();
});

test("an unknown language name gives null, a known one gives its code, whatever the case of the name", async () => {
  const { model: m } = model(PT[0][0], {}, (a) => { a.decoder.languages.pt.name = "Lusitano"; });
  assert.equal((await m.transcribe(audio(8))).language, null);
  const { model: m2 } = model(PT[0][0], {}, (a) => { a.decoder.languages = { zz: { name: "PORTUGUESE", ids: [1, 2] } }; });
  assert.equal((await m2.transcribe(audio(8))).language, "zz");
});

test("maxNewTokens: cuts the text, says 'length', and never generates more (greedy and beam); the manifest's limit is the default", async () => {
  const [ids] = PT[0];
  for (const beam of [false, true]) {
    const { model: m } = model(ids);
    const r = await m.transcribe(audio(8), { beam, maxNewTokens: 5 });
    assert.equal(r.finishReason, "length"); assert.equal(r.tokens, 5);
    assert.ok(r.text.length > 0 && r.text.length < parseAsrText(PT[0][1]).length);
    await m.dispose();
  }
  const long = Array(40).fill(PT[0][0][4]).concat(EOS[0]);
  const { model: m, be } = model(long, {}, (a) => { a.decoder.term.maxNewTokens = 7; });
  const d = await m.transcribe(audio(8));
  assert.equal(d.tokens, 7); assert.equal(d.finishReason, "length");
  await assert.rejects(() => m.transcribe(audio(8), { maxNewTokens: 0 }), /maxNewTokens/);
  await assert.rejects(() => m.transcribe(audio(8), { maxNewTokens: 2.5 }), /maxNewTokens/);
  void be;
});

test("window: a full window is one tower pass; a short one runs the left-aligned pass once; an encoding is reused; both agree on the text", async () => {
  const [ids, raw] = PT[1];
  const full = model(ids), short = model(ids);
  const a = await full.model.transcribe(audio(8)), b = await short.model.transcribe(audio(3));
  assert.equal(full.tower.calls, 1); assert.equal(short.tower.calls, 2);
  assert.equal(a.cost.passes, 1); assert.equal(b.cost.passes, 2);
  assert.ok(b.cost.parts.leftTower >= 0 && b.cost.parts.mel >= 0);
  assert.equal(a.text, parseAsrText(raw)); assert.equal(b.text, a.text);
  const enc = await short.model.encode(audio(3));
  const c = await short.model.transcribe(enc), d = await short.model.transcribe(enc, { beam: true });
  assert.equal(short.tower.calls, 4);       // encode: 1, left-aligned pass: 1 more, shared by both transcriptions
  assert.equal(c.text, a.text); assert.equal(d.text, a.text);
  assert.equal(c.cost.parts.mel, undefined); assert.equal(c.cost.passes, 1);
  assert.equal(c.cost.backend, "fake"); assert.ok(c.cost.ms >= 0);
  await full.model.dispose(); await short.model.dispose();
});

test("an AudioEncoding from read() is also what transcribe takes, and the questions of read still work on it", async () => {
  const [ids, raw] = PT[2];
  const { model: m } = model(ids);
  const enc = await m.encode(audio(8));
  const t = await m.transcribe(enc);
  const [lang] = await m.read(enc, ["language"]);
  assert.equal(t.text, parseAsrText(raw)); assert.equal(lang.modality, "audio");
  await m.dispose();
});

test("signal: an aborted signal rejects with an AbortError before and during generation, and the decoder state is released", async () => {
  const [ids] = PT[0];
  const pre = model(ids), ac = new AbortController(); ac.abort();
  await assert.rejects(() => pre.model.transcribe(audio(8), { signal: ac.signal }), (e) => e.name === "AbortError");
  for (const beam of [false, true]) {
    const ac2 = new AbortController(), { model: m, be } = model(ids, { onFeed: (s) => { if (s.feeds === 6) ac2.abort(); } });
    await assert.rejects(() => m.transcribe(audio(8), { beam, signal: ac2.signal }), (e) => e.name === "AbortError");
    assert.ok(be.feeds >= 6 && be.feeds < 14, "stopped early: " + be.feeds);
    const ok = await m.transcribe(audio(8));        // the model still works
    assert.equal(ok.text, parseAsrText(PT[0][1]));
    await m.dispose();
  }
});

test("dispose() during a transcription waits for it: the decoder is closed after the last feed, and the answer arrives", async () => {
  const [ids, raw] = PT[0];
  const { model: m, be } = model(ids, { delay: 2 });
  const q = m.transcribe(audio(8), { beam: true }).then((r) => r.text, (e) => "failed: " + e.message);
  for (let i = 0; i < 2000 && be.feeds < 3; i++) await new Promise((r) => setTimeout(r, 1));
  assert.ok(be.feeds >= 3 && !be.closed, "running");
  await m.dispose();
  assert.equal(be.closedWhileActive, false); assert.equal(be.closed, true);
  assert.equal(await q, parseAsrText(raw));
  await assert.rejects(() => m.transcribe(audio(8)), /disposed/);
});

test("a package without a decoder block says so; the language and term questions are unchanged by transcribe's existence", async () => {
  const m = clone(MANIFEST);
  for (const k of ["voice_activity", "turn_end", "same_speaker", "speaker_change", "speaker_count", "language", "term"]) delete m.audio.readers[k];
  delete m.audio.decoder;
  const runtime = { backend: "fake", device: "cpu", async tower() { return { audio_features: new Float32Array(104 * H) }; }, speech: async () => 0, dispose: async () => {} };
  const bare = new AudioModel({ spec: m.audio, weights: {}, questions: {}, runtime });
  await assert.rejects(() => bare.transcribe(audio(8)), /no decoder/);
  const { model: ok } = model(PT[0][0]);
  const d = await ok.decide({ audio: audio(8), questions: ["language", { task: "term", term: "bajuns" }] });
  assert.equal(d.answers[0].task, "language"); assert.equal(d.answers[1].hypotheses.length, 5);
  await ok.dispose();
});
