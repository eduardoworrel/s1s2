// Shared helpers of the media tests: fixtures (stored gzip'd float32 arrays), safetensors writer, optional weights.
import { readFileSync, existsSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import path from "node:path";

export { validateManifest, ManifestError } from "../dist/index.js";
export { validateMediaManifest, mediaProblems } from "../dist/media.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.join(HERE, "..");
export const FIX = path.join(HERE, "fixtures/media");
export const have = (p) => existsSync(p);

/** Float32Array stored as <name>.gz in the audio fixture dir. */
export function fixtureF32(name) {
  const b = gunzipSync(readFileSync(path.join(FIX, "audio", name + ".gz")));
  const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  return new Float32Array(ab);
}
export const audioMeta = () => { const f = path.join(FIX, "audio/meta.json"); return existsSync(f) ? JSON.parse(readFileSync(f, "utf8")) : null; };

/** Weights of the audio package, when present: $S1S2_MM_WEIGHTS/marimba-audio or <repo>/.weights/marimba-audio. */
export function audioWeightsDir() {
  const d = path.join(process.env.S1S2_MM_WEIGHTS ?? path.join(ROOT, ".weights"), "marimba-audio");
  return existsSync(path.join(d, "turn_end/readout.safetensors")) ? d : null;
}

/** safetensors bytes from {name: {dtype, shape, data}} (F32 / F64). */
export function safetensors(tensors, metadata) {
  const header = {}, parts = []; let off = 0;
  if (metadata) header.__metadata__ = metadata;
  for (const [n, t] of Object.entries(tensors)) {
    const w = t.dtype === "F64" ? 8 : 4, b = Buffer.alloc(t.data.length * w);
    t.data.forEach((x, i) => (t.dtype === "F64" ? b.writeDoubleLE(x, i * w) : b.writeFloatLE(x, i * w)));
    header[n] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + b.length] }; off += b.length; parts.push(b);
  }
  const h = Buffer.from(JSON.stringify(header)), len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(h.length));
  const all = Buffer.concat([len, h, ...parts]);
  return all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength);
}
