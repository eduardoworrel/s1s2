import test from "node:test";
import assert from "node:assert/strict";
import { validateManifest, ManifestError, fileUrl, headFor } from "../dist/index.js";
import { CT0, clone } from "./helpers.mjs";

const bad = (mut, re) => { const m = clone(CT0); mut(m); assert.throws(() => validateManifest(m), (e) => e instanceof ManifestError && re.test(e.message), re.toString()); };

test("CT0 manifest validates (7 builds, pinned revisions, heads per type)", () => {
  const m = validateManifest(clone(CT0));
  assert.equal(Object.keys(m.builds).length, 7);
  assert.equal(m.hiddenSize, 2560);
  assert.equal(headFor(m.builds["genai-webgpu-int4"], "choice").name, "predicateChoice");
  assert.equal(headFor(m.builds["genai-webgpu-int4"], "score").spec.file.path, "heads/head_score.bin");
  for (const b of Object.values(m.builds)) assert.equal(b.fidelity, null, "fidelity slot present, unfilled");
});

test("revisions must be pinned commit shas", () => {
  bad((m) => { m.builds["decode-only-q4"].revision = "main"; }, /builds.decode-only-q4.revision: pinned 40-hex/);
  bad((m) => { m.base.revision = "61c0e71"; }, /base.revision/);
  bad((m) => { m.tokenizer.revision = m.tokenizer.files.tokenizer.sha256; }, /tokenizer.revision/);
});

test("file sha256 must be 64 lowercase hex; paths relative or https", () => {
  bad((m) => { m.builds["genai-webgpu-int4"].files.data.sha256 = "ddb6c7e0ae9d37d90362c22c2e7222634969c5f4"; }, /files.data.sha256/);
  bad((m) => { m.tokenizer.files.config.sha256 = CT0.tokenizer.files.config.sha256.toUpperCase(); }, /tokenizer.files.config.sha256/);
  bad((m) => { m.builds["decode-only-q4"].files.model.path = "../x.onnx"; }, /files.model.path/);
  bad((m) => { m.builds["decode-only-q4"].files.model.path = "http://x/y.onnx"; }, /files.model.path/);
});

test("required file roles per runtime", () => {
  bad((m) => { delete m.builds["genai-webgpu-int4"].files.data; }, /genai-webgpu-int4.files.data: required for runtime genai/);
  bad((m) => { delete m.builds["litert-int4"].files.embed; }, /files.embed: required for runtime litert/);
  bad((m) => { m.builds["decode-only-q4"].runtime = "llama.cpp"; }, /runtime: one of/);
});

test("heads: two heads cannot claim the same question type", () => {
  bad((m) => { m.builds["decode-only-q4"].heads.score.types = ["score", "choice"]; }, /type choice read by both/);
});

test("kev markers must be the delimiters; options readout needs kev", () => {
  bad((m) => { m.readout.predicate = "options"; }, /requires the kev encoding/);
  bad((m) => {
    m.encoding = { kind: "kev", delimiters: { state: { id: 1, text: "a" }, question: { id: 2, text: "b" }, optionStart: { id: 3, text: "c" }, optionEnd: { id: 4, text: "d" }, decide: { id: 5, text: "e" } } };
    m.markers = { decide: { text: "e", at: "token", tokens: [{ id: 5, text: "e" }] }, option: { text: "x", at: "token", tokens: [{ id: 9, text: "x" }] } };
  }, /kev, the decide\/option marker tokens/);
});

test("logits / defaults / generation consistency", () => {
  bad((m) => { m.logits.predicate.no.id = m.logits.predicate.yes.id; }, /different tokens/);
  bad((m) => { m.logits.score.tokens[1].id = 16; }, /ids must be distinct/);
  bad((m) => { m.defaults.levels = ["1", "2", "3"]; }, /count must equal logits.score.tokens/);
  bad((m) => { m.generation.continue = "{prompt}"; }, /generation.continue/);
  bad((m) => { m.encoding.template = "{question}{state}"; }, /\{state\} must come before/);
  bad((m) => { m.hiddenSize = 0; }, /hiddenSize/);
  bad((m) => { m.layer.norm = "rmsnorm-js"; }, /layer/);
  bad((m) => { m.schema = "other/1"; }, /schema/);
  bad((m) => { m.builds["decode-only-q4"].fidelity = undefined; }, /fidelity: object or null/);
});

test("all problems are listed at once", () => {
  const m = clone(CT0); m.hiddenSize = -1; m.base.revision = "main";
  try { validateManifest(m); assert.fail(); } catch (e) { assert.equal(e.problems.length, 2); }
});

test("fileUrl pins the revision", () => {
  assert.equal(fileUrl("o/r", "abc", "heads/head.bin"), "https://huggingface.co/o/r/resolve/abc/heads/head.bin");
  assert.equal(fileUrl("o/r", "abc", "x.bin", "https://h.example/m"), "https://h.example/m/x.bin");
  assert.equal(fileUrl("o/r", "abc", "https://cdn.example/lib.wasm"), "https://cdn.example/lib.wasm");
});
