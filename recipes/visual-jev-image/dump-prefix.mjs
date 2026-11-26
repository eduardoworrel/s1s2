// node dump-prefix.mjs MANIFEST BUILD Q_JSONL INDEX OUTDIR : writes the prefix inputs of one real question (raw float32 / int64) for the provider comparison pages
import fs from "node:fs"; import { loadMedia } from "../../dist/media.js"; import * as tjs from "@huggingface/transformers"; import * as ort from "onnxruntime-node"; import sharp from "sharp";
import { nodeFloat16Workaround } from "../../dist/index.js";
const [manifest, build, qf, idx, outdir] = process.argv.slice(2); nodeFloat16Workaround(ort); fs.mkdirSync(outdir, { recursive: true });
const m = JSON.parse(fs.readFileSync(manifest, "utf8")), x = fs.readFileSync(qf, "utf8").trim().split("\n").map((l) => JSON.parse(l))[+idx];
const model = await loadMedia({ manifest: m, modality: "image", localDirs: { [m.repo]: build }, device: "cpu", modules: { ort, transformers: tjs }, onPrefix: (p) => {
  fs.writeFileSync(outdir + "/embeds.f32", Buffer.from(p.embeds.buffer, p.embeds.byteOffset, p.embeds.byteLength)); fs.writeFileSync(outdir + "/pos.i64", Buffer.from(p.positions.buffer));
  p.deepstack.forEach((d, i) => fs.writeFileSync(`${outdir}/ds${i}.f32`, Buffer.from(d.buffer, d.byteOffset, d.byteLength))); fs.writeFileSync(outdir + "/meta.json", JSON.stringify({ S: p.length })); } });
const { data, info } = await sharp(x.image).removeAlpha().raw().toBuffer({ resolveWithObject: true });
await model.decide({ image: { rgb: data, width: info.width, height: info.height }, context: x.context, questions: [{ instruction: x.instruction, options: x.options }] });
