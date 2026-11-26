// Image reader, pure parts (spec/MULTIMODAL.md section 10): pre-processing, prompt, M-RoPE positions, option readout, against the reference implementation
// (fixtures from test/fixtures/image/make_fixtures.py: transformers 5.19 Pillow backend, Visual Jev prompts.py, Qwen3VLModel.get_rope_index).
// Tolerances, fixed before the first run of these tests:
//   smart_resize: identical integers (30 cases, three pixel budgets).   Pillow bicubic resize: identical bytes (5 images, up- and down-scaling).
//   pixel_values: max |difference| <= 1e-6 per value (float32 normalisation); patch layout identical.   Prompt token ids: identical (3 prompts).
//   M-RoPE positions: identical integers (4 grids).   Option readout: softmax in float64, |difference| <= 1e-12 against a direct computation.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { smartResize, roundHalfEven, resizeBicubicRGB, patchify, prepareImage, prefixText, suffixText, mropePositions, nextPosition, optionScores, rgbaToRgb } from "../dist/media.js";

const fx = JSON.parse(readFileSync(new URL("./fixtures/image/image.json", import.meta.url), "utf8"));
const SPEC = { patchSize: 16, mergeSize: 2, temporalPatchSize: 2, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] };

test("roundHalfEven is Python's round", () => {
  assert.deepEqual([0.5, 1.5, 2.5, -0.5, 2.4, 2.6, 7.5].map(roundHalfEven), [0, 2, 2, 0, 2, 3, 8]);
});

test("smartResize equals the reference on 30 cases", () => {
  assert.ok(fx.smart_resize.length >= 30);
  for (const c of fx.smart_resize) assert.deepEqual(smartResize(c.h, c.w, { factor: c.factor, minPixels: c.min, maxPixels: c.max }), c.out, JSON.stringify(c));
  assert.throws(() => smartResize(1, 1000, { factor: 32, minPixels: 1024, maxPixels: 4096 }), /aspect ratio/);
});

test("Pillow-compatible bicubic resize is byte-identical to PIL", () => {
  for (const im of fx.images) {
    const out = resizeBicubicRGB(Uint8Array.from(im.rgb), im.h, im.w, im.resized[0], im.resized[1]);
    assert.equal(out.length, im.resizedRgb.length);
    let bad = 0, worst = 0;
    for (let i = 0; i < out.length; i++) if (out[i] !== im.resizedRgb[i]) { bad++; worst = Math.max(worst, Math.abs(out[i] - im.resizedRgb[i])); }
    assert.equal(bad, 0, `${im.w}x${im.h} -> ${im.resized}: ${bad} bytes differ, worst ${worst}`);
  }
});

test("prepareImage: grid, patch layout and values equal the processor's", () => {
  for (const im of fx.images) {
    const r = prepareImage(Uint8Array.from(im.rgb), im.h, im.w, { ...SPEC, minPixels: im.min, maxPixels: im.max });
    assert.deepEqual(r.grid, im.grid);
    assert.deepEqual([r.rows, r.dim], im.shape);
    assert.equal(r.mergedTokens, im.grid[1] * im.grid[2] / 4);
    let worst = 0;
    for (let i = 0; i < r.pixelValues.length; i++) worst = Math.max(worst, Math.abs(r.pixelValues[i] - im.pixelValues[i]));
    assert.ok(worst <= 1e-6, `${im.w}x${im.h}: max |diff| ${worst}`);
  }
});

test("patchify refuses sizes that are not a multiple of 32; rgbaToRgb drops alpha", () => {
  assert.throws(() => patchify(new Uint8Array(30 * 32 * 3), 30, 32, SPEC), /multiple/);
  assert.deepEqual([...rgbaToRgb(Uint8Array.from([1, 2, 3, 255, 4, 5, 6, 0]))], [1, 2, 3, 4, 5, 6]);
});

test("prompt text is the Visual Jev format and its token ids equal the reference tokenizer's", async (t) => {
  const tokDir = "/Users/pc/cai/image-jev/base";
  let tok = null;
  if (existsSync(`${tokDir}/tokenizer.json`)) {
    const tjs = await import("@huggingface/transformers");
    const { tokenizerFromJson } = await import("../dist/index.js");
    tok = tokenizerFromJson(tjs, JSON.parse(readFileSync(`${tokDir}/tokenizer.json`, "utf8")), JSON.parse(readFileSync(`${tokDir}/tokenizer_config.json`, "utf8")));
  }
  for (const p of fx.prompts) {
    assert.equal(prefixText(p.context, p.nPad), p.prefixText);
    assert.equal(suffixText(p.instruction, p.options), p.suffixText);
    if (tok) {
      assert.deepEqual(tok.encode(p.prefixText), p.prefixIds);
      assert.deepEqual(tok.encode(p.suffixText), p.suffixIds);   // separately, as the published inference does
    } else t.diagnostic("tokenizer.json not downloaded: token ids not compared");
  }
  assert.equal(fx.prompts[0].jointEqual, false);   // reference finding: a context ending in "." merges with "\n\n" when tokenised jointly (training), not when separately (inference)
  assert.ok(fx.optionTokenIds.slice(0, 16).every((x) => x.length === 1));   // " A" .. " P" are single tokens
  assert.throws(() => suffixText("q", ["only one"]), /2 to 16/);
  assert.throws(() => suffixText("q", Array.from({ length: 17 }, (_, i) => `o${i}`)), /2 to 16/);
});

test("M-RoPE positions equal get_rope_index on 4 grids; suffix positions continue the prefix", () => {
  for (const c of fx.rope) {
    const isImg = c.ids.map((x) => x === c.imageTokenId);
    const pos = mropePositions(isImg, c.grid);
    for (let a = 0; a < 3; a++) assert.deepEqual([...pos[a]], c.positions[a], `axis ${a} grid ${c.grid}`);
    const pre = pos.map((p) => p.subarray(0, c.prefixLen));
    assert.equal(nextPosition(pre), c.positions[0][c.prefixLen]);   // the first suffix token
  }
});

test("optionScores: softmax over the supplied options of hidden . rows", () => {
  const d = 8, k = 3, h = Array.from({ length: d }, (_, i) => Math.sin(i + 1)), rows = Array.from({ length: k * d }, (_, i) => Math.cos(i * 0.7));
  const r = optionScores(h, rows, k);
  const logits = [0, 1, 2].map((o) => h.reduce((s, v, i) => s + v * rows[o * d + i], 0));
  const z = logits.reduce((s, v) => s + Math.exp(v), 0);
  r.probabilities.forEach((p, i) => assert.ok(Math.abs(p - Math.exp(logits[i]) / z) < 1e-12));
  assert.ok(Math.abs(r.probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.equal(r.choice, logits.indexOf(Math.max(...logits)));
  assert.ok(Math.abs(r.rawMargin - (Math.max(...logits) - [...logits].sort((a, b) => b - a)[1])) < 1e-12);
  assert.throws(() => optionScores(h, rows, 1), /two/);
});

// ---- manifest block `image` (schema + cross-field rules) -------------------------------------------------------------------------------------------
import { validateMediaManifest } from "../dist/media.js";
import { schemaProblems } from "../dist/package.js";
import { clone } from "./helpers.mjs";

const sha = (n) => n.toString(16).padStart(64, "0");
const pin = (path, n, size) => ({ path, sha256: sha(n), ...(size ? { size } : {}) });
export const IMAGE_MANIFEST = {   // synthetic: sizes are the real ones of the 4B build, hashes are placeholders for this test only
  schema: "s1s2.manifest/1", id: "test-image-reader", modalities: ["image"], repo: "example/image-reader", revision: "local",
  image: {
    preprocess: { kind: "qwen-vl-patches", patchSize: 16, mergeSize: 2, temporalPatchSize: 2, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], resample: "pillow-bicubic", minPixels: 65536, maxPixels: 786432, referenceMaxPixels: 16777216 },
    vision: { runtime: "onnx", graph: pin("vision.onnx", 1), data: [pin("vision.onnx.data", 2)], inputs: { pixelValues: "pixel_values", grid: "image_grid_thw" }, outputs: { features: "image_features", deepstack: ["deepstack_0", "deepstack_1", "deepstack_2"] }, devices: ["webgpu", "wasm"] },
    text: { runtime: "onnx", graph: pin("text.onnx", 3), data: [pin("text.onnx.data", 4, 1900000000), pin("text.onnx.data_1", 5, 600000000)], inputs: { embeds: "inputs_embeds", attentionMask: "attention_mask", positionIds: "position_ids", deepstack: ["deepstack_0", "deepstack_1", "deepstack_2"] },
      cache: { pastKey: "past_key_values.%d.key", pastValue: "past_key_values.%d.value", presentKey: "present.%d.key", presentValue: "present.%d.value" }, output: { hidden: "hidden" }, model: { layers: 36, kvHeads: 8, headDim: 128, hidden: 2560, vocab: 151936 }, ioDtype: "float16", devices: ["webgpu"] },
    embeddings: { file: pin("embed_tokens.f16", 6, 151936 * 2560 * 2), format: "f16" },
    tokenizer: { tokenizer: pin("tokenizer.json", 7), config: pin("tokenizer_config.json", 8) },
    prompt: { imagePad: 151655, visionStart: 151652, visionEnd: 151653, imStart: 151644, imEnd: 151645 },
    readout: { kind: "option-letters", tokenIds: [362, 425, 356, 422, 468, 434, 479, 472, 358, 619, 730, 444, 386, 451, 506, 393], rows: pin("option_rows.f16", 9, 16 * 2560 * 2), minOptions: 2, maxOptions: 16, temperature: 1 },
  },
};
const problems = (m) => { try { validateMediaManifest(m); return []; } catch (e) { assert.equal(e.name, "ManifestError"); return e.problems; } };

test("an image manifest validates by schema and by rules", async () => {
  assert.deepEqual(problems(IMAGE_MANIFEST), []);
  assert.deepEqual(await schemaProblems(IMAGE_MANIFEST), []);
  assert.deepEqual(IMAGE_MANIFEST.image.readout.tokenIds, fx.optionTokenIds.map((x) => x[0]));   // the manifest ids are the reference tokenizer's
  assert.deepEqual([IMAGE_MANIFEST.image.prompt.imagePad], fx.specialIds["<|image_pad|>"]);
});

test("image manifest rules: what each mutation says", async () => {
  const bad = (mut) => { const m = clone(IMAGE_MANIFEST); mut(m); return problems(m).join("\n"); };
  assert.match(bad((m) => { m.image.vision.outputs.deepstack = []; }), /DeepStack outputs are required/);
  assert.match(bad((m) => { m.image.text.inputs.deepstack = ["a", "b"]; }), /2 inputs for 3 vision DeepStack outputs/);
  assert.match(bad((m) => { m.image.text.data[0].size = 2.5e9; }), /at most 2 GB/);
  assert.match(bad((m) => { m.image.embeddings.file.size = 5; }), /vocab \* hidden \* 2/);
  assert.match(bad((m) => { m.image.readout.rows.size = 5; }), /maxOptions \* hidden \* 2/);
  assert.match(bad((m) => { m.image.readout.tokenIds[1] = m.image.readout.tokenIds[0]; }), /distinct token ids/);
  assert.match(bad((m) => { m.image.readout.tokenIds[0] = 999999; }), /inside the vocabulary/);
  assert.match(bad((m) => { m.image.embeddings.format = "q4-block128"; }), /scales/);
  assert.match(bad((m) => { m.image.preprocess.resample = "bilinear"; }), /pillow-bicubic/);
  assert.match(bad((m) => { m.image.preprocess.minPixels = 10 ** 9; }), /minPixels <= maxPixels/);
  assert.match(bad((m) => { m.image.text.cache.pastKey = "past"; }), /%d/);
  assert.match(bad((m) => { m.modalities = ["audio"]; }), /image: present, but modalities does not list "image"/);
  assert.match(bad((m) => { delete m.image; }), /image: required by modalities "image"/);
  const s = clone(IMAGE_MANIFEST); s.image.text.data[0].size = 2.5e9; s.image.readout.tokenIds.pop();
  assert.ok((await schemaProblems({ ...clone(IMAGE_MANIFEST), image: { ...clone(IMAGE_MANIFEST).image, prompt: { imagePad: -1 } } })).length > 0);
  assert.ok(problems(s).length > 0);
});

test("a text manifest that lists image has the image block checked too", () => {
  const m = clone(IMAGE_MANIFEST); m.modalities = ["image", "audio"];
  assert.match(problems(m).join("\n"), /audio: required by modalities "audio"/);
});

test("loadMedia on an image manifest needs the files it pins", async () => {
  const { loadMedia } = await import("../dist/media.js");
  await assert.rejects(() => loadMedia({ manifest: IMAGE_MANIFEST, modality: "image", localDirs: { "example/image-reader": "/nonexistent" }, device: "cpu", cacheDir: "/nonexistent-cache", fetch: async () => ({ ok: false, status: 404 }), modules: { ort: {}, transformers: {} }, tokenizer: { encode: () => [], decode: () => "" } }), /./);
});

test("positionEmbeddings equals fast_pos_embed_interpolate (4 grids), tolerance 1e-5", async () => {
  const { positionEmbeddings } = await import("../dist/media.js");
  for (const c of fx.posembed) {
    const o = positionEmbeddings(c.table, c.side, c.hidden, c.grid);
    assert.equal(o.length, c.out.length);
    let worst = 0;
    for (let i = 0; i < o.length; i++) worst = Math.max(worst, Math.abs(o[i] - c.out[i]));
    assert.ok(worst <= 1e-5, `grid ${c.grid}: ${worst}`);
  }
  assert.throws(() => positionEmbeddings(new Float32Array(3), 6, 8, [1, 2, 2]), /table/);
});
