import test from "node:test";
import assert from "node:assert/strict";
import { planBuilds, deviceFor } from "../dist/index.js";
import { CT0, clone } from "./helpers.mjs";

const node = { platform: "node", webgpu: false, cuda: false }, nodeCuda = { ...node, cuda: true };
const web = { platform: "browser", webgpu: true, cuda: false }, webNoGpu = { ...web, webgpu: false };

test("auto picks builds from the manifest in ct0-edge order", () => {
  assert.deepEqual(planBuilds(CT0, {}, node).map((p) => p.build), ["transformers-q4", "decode-only-q4"]);
  assert.deepEqual(planBuilds(CT0, {}, nodeCuda).map((p) => p.build), ["genai-cuda-int4", "transformers-q4", "transformers-q4", "decode-only-q4"]);
  assert.deepEqual(planBuilds(CT0, {}, web).map((p) => `${p.build}/${p.device}`), ["genai-webgpu-int4/webgpu", "decode-only-q4/webgpu", "transformers-q4/webgpu"]);
  assert.deepEqual(planBuilds(CT0, {}, webNoGpu).map((p) => `${p.build}/${p.device}`), ["decode-only-q4/wasm", "transformers-q4/wasm"]);
});

test("explicit build / runtime", () => {
  assert.deepEqual(planBuilds(CT0, { build: "decode-only-q4" }, node), [{ build: "decode-only-q4", runtime: "decode-only", device: "cpu" }]);
  assert.deepEqual(planBuilds(CT0, { runtime: "litert" }, web), [{ build: "litert-int4", runtime: "litert", device: "webgpu" }]);
  assert.throws(() => planBuilds(CT0, { build: "nope" }, node), /unknown build/);
  assert.throws(() => planBuilds(CT0, { build: "genai-webgpu-int4" }, node), /does not run on onnxruntime-node CPU/);
  assert.throws(() => planBuilds(CT0, { runtime: "webllm" }, node), /only in the browser/);
  assert.throws(() => planBuilds(CT0, { build: "decode-only-q4", runtime: "genai" }, node), /is runtime decode-only/);
  assert.throws(() => deviceFor("genai", "wasm", web), /lists device "wasm"/);
  assert.equal(deviceFor("transformers.js", "cpu", web), "wasm");
});

test("genai on Node CPU only when the build lists device cpu (opt-in)", () => {
  const m = clone(CT0);
  m.builds["genai-webgpu-int4"].devices = ["webgpu", "cpu"];
  assert.deepEqual(planBuilds(m, { build: "genai-webgpu-int4" }, node), [{ build: "genai-webgpu-int4", runtime: "genai", device: "cpu" }]);
  assert.deepEqual(planBuilds(m, { build: "genai-webgpu-int4", device: "cpu" }, nodeCuda), [{ build: "genai-webgpu-int4", runtime: "genai", device: "cpu" }]);
  assert.deepEqual(planBuilds(m, { build: "genai-webgpu-int4" }, web), [{ build: "genai-webgpu-int4", runtime: "genai", device: "webgpu" }]);
  assert.throws(() => planBuilds(CT0, { build: "genai-webgpu-int4", device: "cpu" }, node), /does not run on onnxruntime-node CPU/);
});
