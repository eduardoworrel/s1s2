// Browser CPU path (onnxruntime-web WASM) through the public S1S2.load: a genai build whose manifest lists device
// "wasm" loads on the CPU when WebGPU is absent (navigator.gpu missing or no adapter), with the session options of
// the measured browser-CPU route; builds without "wasm" are refused with the reason; cpuFallback: false opts out.
// The browser is simulated (window/document defined); the "onnxruntime-web" module is a stand-in that records the
// session options and runs the toy graph on onnxruntime-node CPU, so the decisions can be compared with Node CPU.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import { S1S2, planBuilds, nodeFloat16Workaround, validateManifest } from "../dist/index.js";
import { toy, toyManifest } from "./helpers.mjs";

nodeFloat16Workaround(ort);

const DIR = fileURLToPath(new URL("./fixtures/embeds/", import.meta.url));
const REPO = "local/toy-embeds", BASE = "http://local.test/toy/";
const pin = (rel) => ({ path: rel, sha256: createHash("sha256").update(readFileSync(DIR + rel)).digest("hex"), size: statSync(DIR + rel).size });

function manifest(devices) {
  const m = toyManifest();
  m.hiddenSize = 16;
  m.tokenizer = { repo: REPO, revision: "0".repeat(40), files: { tokenizer: pin("tokenizer.json"), config: pin("tokenizer_config.json") } };
  m.builds = { "genai-int4-webgpu": { runtime: "genai", repo: REPO, revision: "0".repeat(40), hiddenOutput: "hidden_states", devices, dtype: "fp32", fidelity: null,
    files: { model: pin("ids/model.onnx"), data: pin("ids/model.onnx.data") } } };
  return validateManifest(m);
}
const tjs = { PreTrainedTokenizer: class { encode(t) { return toy.encode(t); } decode(i) { return toy.decode(i); } } };
const fetchLocal = async (u) => {
  if (!String(u).startsWith(BASE)) throw new Error(`unexpected fetch ${u}`);
  return new Response(readFileSync(DIR + String(u).slice(BASE.length)));
};
/** onnxruntime-web stand-in: records InferenceSession.create(model, options); runs the toy graph on onnxruntime-node CPU. */
function webOrt(log, fail) {
  return { Tensor: ort.Tensor, env: { versions: { web: "1.31.1" } }, InferenceSession: { async create(model, o) {
    log.push({ model, o });
    if (fail) throw new Error(fail);
    return ort.InferenceSession.create(DIR + "ids/model.onnx", { executionProviders: ["cpu"], logSeverityLevel: 3 });
  } } };
}
/** Run fn as if in a browser whose navigator is `nav`. */
async function inBrowser(nav, fn) {
  const saved = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  globalThis.window = {}; globalThis.document = {};
  Object.defineProperty(globalThis, "navigator", { value: nav, configurable: true, writable: true });
  try { return await fn(); }
  finally {
    delete globalThis.window; delete globalThis.document;
    if (saved) Object.defineProperty(globalThis, "navigator", saved); else delete globalThis.navigator;
  }
}
const browserLoad = (m, log, extra = {}, fail) => S1S2.load({ manifest: m, fetch: fetchLocal, baseUrls: { [REPO]: BASE }, modules: { ort: webOrt(log, fail), transformers: tjs }, ...extra });
const req = { input: "We share data with partners.", questions: [
  { type: "choice", name: "c", instructions: "Is it so?", choices: [" Yes", " No"], method: "logits" },
  { type: "predicate", name: "p", instructions: "Is it so?", method: "logits" }] };

test("no navigator.gpu: a build listing \"wasm\" loads on the browser CPU with the measured browser-CPU session options; same decisions as Node CPU", async () => {
  const log = [];
  const lib = await inBrowser({}, () => browserLoad(manifest(["webgpu", "wasm"]), log));
  assert.equal(lib.device, "wasm");
  assert.equal(lib.runtime, "genai");
  assert.match(lib.fallbackReason, /navigator\.gpu is missing.*browser CPU \(WASM\)/);
  assert.equal(log.length, 1, "one session, no EP-placement probe");
  assert.ok(log[0].model instanceof Uint8Array, "session from the sha256-verified graph bytes");
  assert.deepEqual(log[0].o, { executionProviders: ["wasm"], externalData: [{ path: "model.onnx.data", data: BASE + "ids/model.onnx.data" }] });
  const cpu = await S1S2.load({ manifest: manifest(["cpu"]), build: "genai-int4-webgpu", device: "cpu", localDirs: { [REPO]: DIR }, modules: { ort, transformers: tjs }, cacheDir: join(tmpdir(), "s1s2-test-wasm") });
  const [a, b] = [await lib.decide(req), await cpu.decide(req)];
  assert.deepEqual(a.answers, b.answers);
  await a.dispose(); await b.dispose(); await lib.dispose(); await cpu.dispose();
});

test("adapter null: same fallback; device \"wasm\" asked explicitly has no fallbackReason", async () => {
  const nav = { gpu: { requestAdapter: async () => null } };
  const log = [];
  const lib = await inBrowser(nav, () => browserLoad(manifest(["webgpu", "wasm"]), log));
  assert.equal(lib.device, "wasm");
  assert.match(lib.fallbackReason, /returned no adapter/);
  await lib.dispose();
  const asked = await inBrowser(nav, () => browserLoad(manifest(["webgpu", "wasm"]), log, { device: "wasm" }));
  assert.equal(asked.device, "wasm");
  assert.equal(asked.fallbackReason, undefined);
  await asked.dispose();
});

test("cpuFallback: false refuses the automatic move to the CPU; device \"wasm\" still works", async () => {
  const log = [];
  await assert.rejects(inBrowser({}, () => browserLoad(manifest(["webgpu", "wasm"]), log, { cpuFallback: false })), /cpuFallback is false/);
  assert.equal(log.length, 0);
  const lib = await inBrowser({}, () => browserLoad(manifest(["webgpu", "wasm"]), log, { cpuFallback: false, device: "wasm" }));
  assert.equal(lib.device, "wasm");
  await lib.dispose();
});

test("a build without \"wasm\" (S5-nano) is never tried on the CPU: the error says why", async () => {
  const log = [];
  await assert.rejects(inBrowser({}, () => browserLoad(manifest(["webgpu"]), log)), (e) => /navigator\.gpu is missing/.test(e.message) && /not "wasm"|no build lists device "wasm"/.test(e.message) && /GatedDeltaNet/.test(e.message));
  await assert.rejects(inBrowser({}, () => browserLoad(manifest(["webgpu"]), log, { build: "genai-int4-webgpu" })), /cannot run here.*not "wasm".*GatedDeltaNet/s);
  await assert.rejects(inBrowser({}, () => browserLoad(manifest(["webgpu"]), log, { device: "wasm" })), /no build lists device "wasm"/);
  assert.equal(log.length, 0, "no session created");
});

test("a kernel missing in onnxruntime-web WASM is reported as such", async () => {
  const log = [];
  const msg = "Can't create a session. ERROR_CODE: 9, ERROR_MESSAGE: Failed to find kernel for com.microsoft.GatedDeltaNet(1)";
  await assert.rejects(inBrowser({}, () => browserLoad(manifest(["webgpu", "wasm"]), log, {}, msg)), /no WASM kernel.*needs WebGPU/s);
});

test("planning: WebGPU present keeps WebGPU; auto without WebGPU puts genai/wasm first; Node is unchanged", () => {
  const m = manifest(["webgpu", "wasm"]);
  const web = { platform: "browser", webgpu: true, cuda: false }, noGpu = { ...web, webgpu: false, webgpuReason: "navigator.gpu is missing" };
  assert.deepEqual(planBuilds(m, {}, web), [{ build: "genai-int4-webgpu", runtime: "genai", device: "webgpu" }]);
  assert.deepEqual(planBuilds(m, {}, noGpu), [{ build: "genai-int4-webgpu", runtime: "genai", device: "wasm" }]);
  assert.deepEqual(planBuilds(m, { build: "genai-int4-webgpu", device: "cpu" }, web), [{ build: "genai-int4-webgpu", runtime: "genai", device: "wasm" }]);
  assert.deepEqual(planBuilds(m, { runtime: "genai" }, noGpu), [{ build: "genai-int4-webgpu", runtime: "genai", device: "wasm" }]);
  assert.throws(() => planBuilds(m, { build: "genai-int4-webgpu", device: "webgpu" }, noGpu), /WebGPU is not available/);
  assert.throws(() => planBuilds(m, { device: "wasm" }, { platform: "node", webgpu: false, cuda: false }), /browser-only/);
});
