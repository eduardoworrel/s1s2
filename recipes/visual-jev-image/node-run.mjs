// node node-run.mjs MANIFEST BUILD_DIR Q_JSONL OUT.jsonl [n]: the ImageModel on Node CPU (onnxruntime-node), same questions as ort_reader.py
import fs from "node:fs"; import { createRequire } from "node:module";
import { loadMedia } from "../../dist/media.js"; import { nodeFloat16Workaround } from "../../dist/index.js";
import * as tjs from "@huggingface/transformers"; import * as ort from "onnxruntime-node"; import sharp from "sharp";
const [manifest, build, qf, out, n = "5"] = process.argv.slice(2);
nodeFloat16Workaround(ort);
const m = JSON.parse(fs.readFileSync(manifest, "utf8"));
const model = await loadMedia({ manifest: m, modality: "image", localDirs: { [m.repo]: build }, device: "cpu", modules: { ort, transformers: tjs }, onStatus: console.log });
const qs = fs.readFileSync(qf, "utf8").trim().split("\n").slice(0, +n).map((l) => JSON.parse(l)), rows = [];
for (const x of qs) {
  const { data, info } = await sharp(x.image).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const r = await model.decide({ image: { rgb: data, width: info.width, height: info.height, channels: 3 }, context: x.context, questions: [{ instruction: x.instruction, options: x.options }] });
  rows.push({ image: x.image, instruction: x.instruction, options: x.options, label: x.label, grid: r.cost.grid, probabilities: r.answers[0].probabilities, logits: r.answers[0].logits });
  console.log(rows.length, r.cost.ms, JSON.stringify(r.cost.parts));
}
fs.writeFileSync(out, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
