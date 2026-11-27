// node scripts/pin-image-manifest.mjs <40-hex commit of eduardoworrel/Visual-Jev-4B-ONNX-S1S2>: pins manifests/visual-jev-image.json to the Hub commit of the weights.
import fs from "node:fs";
const sha = process.argv[2], f = new URL("../manifests/visual-jev-image.json", import.meta.url);
if (!/^[0-9a-f]{40}$/.test(sha ?? "")) { console.error("usage: node scripts/pin-image-manifest.mjs <40-hex commit>"); process.exit(1); }
const m = JSON.parse(fs.readFileSync(f, "utf8")); m.revision = sha; fs.writeFileSync(f, JSON.stringify(m, null, 1)); console.log("pinned", sha);
