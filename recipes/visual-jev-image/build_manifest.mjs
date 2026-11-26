// node build_manifest.mjs BUILD_DIR out.json : the manifest of a BUILD made by README.md steps 1 to 3 (hashes and sizes read from the files)
import fs from "node:fs"; import path from "node:path"; import crypto from "node:crypto";
const dir = process.argv[2], out = process.argv[3];
const pin = (f) => { const p = path.join(dir, f), h = crypto.createHash("sha256"); const fd = fs.openSync(p, "r"), b = Buffer.alloc(1 << 24); let n; while ((n = fs.readSync(fd, b, 0, b.length, null)) > 0) h.update(b.subarray(0, n)); fs.closeSync(fd); return { path: f, sha256: h.digest("hex"), size: fs.statSync(p).size }; };
const fx = JSON.parse(fs.readFileSync(new URL("../../test/fixtures/image/image.json", import.meta.url), "utf8"));
const ds = ["deepstack_0", "deepstack_1", "deepstack_2"];
const m = { schema: "s1s2.manifest/1", id: "visual-jev-4b-image-local", modalities: ["image"], repo: "local/visual-jev-4b", revision: "local", license: "Apache-2.0", image: {
  preprocess: { kind: "qwen-vl-patches", patchSize: 16, mergeSize: 2, temporalPatchSize: 2, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], resample: "pillow-bicubic", minPixels: 65536, maxPixels: Number(process.env.MAX_PIXELS ?? 200704), referenceMaxPixels: 16777216 },
  vision: { runtime: "onnx", graph: pin("vision.onnx"), data: [pin("vision.onnx.data")], inputs: { pixelValues: "pixel_values", grid: "image_grid_thw" }, outputs: { features: "image_features", deepstack: ds }, devices: ["cpu", "webgpu", "wasm"] },
  text: { runtime: "onnx", graph: pin("text.onnx"), data: [pin("text.onnx.data"), pin("text.onnx.data_1")], inputs: { embeds: "inputs_embeds", attentionMask: "attention_mask", positionIds: "position_ids", deepstack: ds },
    cache: { pastKey: "past_key_values.%d.key", pastValue: "past_key_values.%d.value", presentKey: "present.%d.key", presentValue: "present.%d.value" }, output: { hidden: "hidden" }, model: { layers: 36, kvHeads: 8, headDim: 128, hidden: 2560, vocab: 151936 }, ioDtype: process.env.IO ?? "float16", devices: ["cpu", "webgpu", "wasm"] },
  embeddings: { file: pin("embedding.fp16"), format: "f16" }, tokenizer: { tokenizer: pin("tokenizer.json"), config: pin("tokenizer_config.json") },
  prompt: { imagePad: 151655, visionStart: 151652, visionEnd: 151653, imStart: 151644, imEnd: 151645 },
  readout: { kind: "option-letters", tokenIds: fx.optionTokenIds.map((x) => x[0]), rows: pin("option_rows.f16"), minOptions: 2, maxOptions: 16, temperature: 1 } } };
fs.writeFileSync(out, JSON.stringify(m, null, 1)); console.log("wrote", out);
