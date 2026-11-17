// exclude_embeds builds (Jurema): the graph takes inputs_embeds and the lib looks the token rows up in a pinned fp16
// table (file role "embeddings"). Fixtures (test/tools/make_stub_genai.py embeds --hidden 16 --vocab 128 --seed 7):
// two equivalent decoders over one fp16 table, one with the Gather in the graph (input_ids), one without
// (inputs_embeds), with a KV-like state so chunked prefill is exercised. The outputs must match bit for bit.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import { genaiFromSession, embeddingTableFromBytes, embeddingTable, fileTable, httpRangeTable, validateManifest, S1S2, nodeFloat16Workaround } from "../dist/index.js";
import { toy, toyManifest } from "./helpers.mjs";

// fp16 inputs_embeds tensors on onnxruntime-node: same workaround S1S2.load applies, before any ort tensor exists.
nodeFloat16Workaround(ort);

const DIR = fileURLToPath(new URL("./fixtures/embeds/", import.meta.url));
const H = 16;
const table = readFileSync(DIR + "embed_tokens.fp16.bin");
const session = (d) => ort.InferenceSession.create(DIR + d + "/model.onnx", { executionProviders: ["cpu"], logSeverityLevel: 3 });
const eos = new Set([2]);
const ids = [5, 17, 3, 99, 17, 64, 0, 127, 42, 42, 8];

test("embeddings table: fp16 rows by id, cache, bounds and size checks", async () => {
  const t = embeddingTableFromBytes(new Uint8Array(table), H);
  assert.equal(t.vocab, 128);
  const r = await t.rows([3, 0, 3]);
  const u16 = new Uint16Array(table.buffer, table.byteOffset, table.byteLength / 2);
  assert.deepEqual(Array.from(r.subarray(0, H)), Array.from(u16.subarray(3 * H, 4 * H)));
  assert.deepEqual(Array.from(r.subarray(2 * H)), Array.from(u16.subarray(3 * H, 4 * H)));
  await assert.rejects(() => t.rows([128]), /outside the table/);
  assert.throws(() => embeddingTableFromBytes(new Uint8Array(table.subarray(0, 100)), H), /not a multiple/);
  let reads = 0;
  const c = embeddingTable(async (o, n) => { reads++; return new Uint8Array(table.subarray(o, o + n)); }, table.byteLength, H);
  await c.rows([1, 2, 1]); await c.rows([2, 1]);
  assert.equal(reads, 2, "each distinct row read once");
  const f = await fileTable(DIR + "embed_tokens.fp16.bin", H);
  assert.deepEqual(Array.from(await f.rows([7, 9])), Array.from(await t.rows([7, 9])));
  await f.close();
});

test("browser tables above 300 MiB: HTTP Range rows (206); a server without Range (200) is refused", async () => {
  const seen = [];
  const f206 = async (url, o) => {
    const [, a, b] = /bytes=(\d+)-(\d+)/.exec(o.headers.Range); seen.push(o.headers.Range);
    return new Response(table.subarray(Number(a), Number(b) + 1), { status: 206 });
  };
  const t = httpRangeTable("http://x/embed_tokens.fp16.bin", table.byteLength, H, f206);
  assert.deepEqual(Array.from(await t.rows([9, 2, 9])), Array.from(await embeddingTableFromBytes(new Uint8Array(table), H).rows([9, 2, 9])));
  assert.deepEqual(seen, [`bytes=${9 * H * 2}-${10 * H * 2 - 1}`, `bytes=${2 * H * 2}-${3 * H * 2 - 1}`]);
  const t200 = httpRangeTable("http://x/e.bin", table.byteLength, H, async () => new Response(table, { status: 200 }));
  await assert.rejects(() => t200.rows([1]), /must support Range \(206\)/);
});

test("inputs_embeds graph + table == input_ids graph (logits, hidden states, chunked state, generation)", async () => {
  const A = genaiFromSession(await session("ids"), ort, { eos, hiddenOutput: "hidden_states", chunk: 4 });
  const B = genaiFromSession(await session("embeds"), ort, { eos, hiddenOutput: "hidden_states", chunk: 4, embeddings: embeddingTableFromBytes(new Uint8Array(table), H) });
  const pos = [0, 4, 7, ids.length - 1];
  const a = await A.prefill(ids, { positions: pos }), b = await B.prefill(ids, { positions: pos });
  assert.deepEqual(Array.from(b.lastLogits), Array.from(a.lastLogits));
  b.hiddens.forEach((h, i) => assert.deepEqual(Array.from(h), Array.from(a.hiddens[i]), `hidden at ${pos[i]}`));
  // the state carries across chunks (same result as one chunk, up to float summation order)
  const A1 = genaiFromSession(await session("ids"), ort, { eos, hiddenOutput: "hidden_states", chunk: 128 });
  const one = (await A1.prefill(ids, { positions: [] })).lastLogits;
  assert.ok(one.every((x, i) => Math.abs(x - a.lastLogits[i]) < 1e-5), "chunk 4 vs one chunk");
  const g = async (be) => { const o = []; for await (const t of be.generate(ids, { maxNewTokens: 6, temperature: 0 })) o.push(t); return o; };
  assert.deepEqual(await g(B), await g(A));
  await A.dispose(); await B.dispose(); await A1.dispose();
});

test("inputs_embeds graph without a table, or a table with an input_ids graph, is refused", async () => {
  assert.throws(() => genaiFromSession({ inputNames: ["inputs_embeds", "attention_mask"], outputNames: ["logits"], inputMetadata: [] }, ort, { eos, hiddenOutput: null }),
    /inputs_embeds.*embeddings/);
  const S = await session("ids");
  assert.throws(() => genaiFromSession(S, ort, { eos, hiddenOutput: null, embeddings: embeddingTableFromBytes(new Uint8Array(table), H) }), /takes input_ids/);
  await S.release();
  const S2 = await session("embeds");
  assert.throws(() => genaiFromSession(S2, ort, { eos, hiddenOutput: null, embeddings: embeddingTableFromBytes(new Uint8Array(table.subarray(0, 8 * 32)), 8) }), /hidden 16 != embeddings table hidden 8/);
  await S2.release();
});

const pin = (rel) => ({ path: rel, sha256: createHash("sha256").update(readFileSync(DIR + rel)).digest("hex"), size: statSync(DIR + rel).size });

function manifest() {
  const m = toyManifest();
  m.hiddenSize = H;
  m.tokenizer = { repo: "local/toy-embeds", revision: "0".repeat(40), files: { tokenizer: pin("tokenizer.json"), config: pin("tokenizer_config.json") } };
  const b = (sub, extra) => ({ runtime: "genai", repo: "local/toy-embeds", revision: "0".repeat(40), hiddenOutput: "hidden_states", devices: ["cpu"], dtype: "fp32", fidelity: null,
    files: { model: pin(`${sub}/model.onnx`), data: pin(`${sub}/model.onnx.data`), ...extra } });
  m.builds = { ids: b("ids", {}), embeds: b("embeds", { embeddings: pin("embed_tokens.fp16.bin") }) };
  return m;
}
// transformers.js stand-in: the toy tokenizer (the pinned tokenizer files are only hashed).
const tjs = { PreTrainedTokenizer: class { encode(t) { return toy.encode(t); } decode(i) { return toy.decode(i); } } };

test("manifest role embeddings: S1S2.load (genai, Node CPU) gives the same decisions and text as the input_ids build", async () => {
  const load = (build) => S1S2.load({ manifest: manifest(), build, device: "cpu", localDirs: { "local/toy-embeds": DIR }, modules: { ort, transformers: tjs }, cacheDir: join(tmpdir(), "s1s2-test-embeds") });
  const [la, lb] = [await load("ids"), await load("embeds")];
  const req = { input: "We share data with partners.", questions: [
    { type: "choice", name: "c", instructions: "Is it so?", choices: [" Yes", " No"], method: "logits" },
    { type: "predicate", name: "p", instructions: "Is it so?", method: "logits" }] };
  const [sa, sb] = [await la.decide(req), await lb.decide(req)];
  assert.deepEqual(sb.answers, sa.answers);
  const [ga, gb] = [await sa.generate("Why?", { maxNewTokens: 5, temperature: 0 }), await sb.generate("Why?", { maxNewTokens: 5, temperature: 0 })];
  assert.deepEqual(gb.tokenIds, ga.tokenIds);
  await sa.dispose(); await sb.dispose(); await la.dispose(); await lb.dispose();
});

test("manifest validation: embeddings only for genai, size pinned and a multiple of hiddenSize * 2", () => {
  const m = manifest();
  m.builds.embeds.files.embeddings.size = 100;
  assert.throws(() => validateManifest(m), /files.embeddings.size/);
  const m2 = manifest(); m2.builds.embeds.runtime = "decode-only";
  assert.throws(() => validateManifest(m2), /only for runtime genai/);
});
