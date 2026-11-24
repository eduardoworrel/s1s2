// Smoke test of the forecast reader on a real graph (Node CPU): load the fp32 and the int4 build of a package made by
// converter/examples/forecast_smoke.py, run decide({forecast:true}) on the reference prompts, and compare the head output
//   (a) fp32 build against the PyTorch reference (proves the exported layer and the pooling are the ones the head was trained on),
//   (b) every other build (int8, int4) against the fp32 build,
// with the tolerance of forecastParity (max |dp| 0.05, flips <= 5%), fixed in src/fidelity.ts before this was run.
// A mechanism check, not a measurement of the reader. Usage: node test/smoke/forecast_parity.mjs <pkgDir> <reference.jsonl> [out.json]
import { S1S2 } from "../../dist/index.js";
import { forecastParity } from "../../dist/fidelity.js";
import fs from "node:fs";
import path from "node:path";
import { homedir } from "node:os";

const [pkg, refFile, outFile] = process.argv.slice(2);
if (!pkg || !refFile) { console.error("usage: forecast_parity.mjs <pkgDir> <reference.jsonl> [out.json]"); process.exit(2); }
const manifest = JSON.parse(fs.readFileSync(path.join(pkg, "manifest.json"), "utf8"));
const ref = fs.readFileSync(refFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const question = { name: "q", type: "predicate", instructions: "Does the request ask for a list?" };
const snap = process.env.SMOLLM2_DIR ?? path.join(homedir(), ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots", manifest.tokenizer.revision);
const localDirs = { [manifest.tokenizer.repo]: snap, "local/smollm2-forecast-smoke": path.resolve(pkg) };

async function run(build) {
  const t0 = Date.now();
  const model = await S1S2.load({ manifest, build, localDirs, device: "cpu", cacheDir: path.join(pkg, ".cache") });
  const out = [];
  try {
    for (const r of ref) {
      const { forecast, session } = await model.forecast({ input: r.prompt, questions: [question] });
      out.push({ id: r.id, behaviors: Object.fromEntries(Object.entries(forecast.behaviors).map(([k, v]) => [k, { p: v.p }])), tokens: forecast.cost.prefillTokens, extra: forecast.cost.extraPrefillTokens });
      await session.dispose();
    }
  } finally { await model.dispose(); }
  console.log(`${build}: ${out.length} prompts in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  return out;
}
const fp32 = await run("genai-fp32-cpu");
const a = forecastParity(ref, fp32);
const others = {};
for (const b of Object.keys(manifest.builds).filter((n) => n !== "genai-fp32-cpu")) others[b] = forecastParity(fp32, await run(b));
const fmt = (x) => ({ n: x.n, maxAbsDp: +x.maxAbsDp.toFixed(5), meanAbsDp: +x.meanAbsDp.toFixed(5), flips: x.flips, pairs: x.pairs, pass: x.pass, tolerance: x.tolerance });
const res = { layer: manifest.forecast.layer, extraPrefillTokens: [...new Set(fp32.map((r) => r.extra))], "fp32 vs PyTorch reference": fmt(a), ...Object.fromEntries(Object.entries(others).map(([k, v]) => [`${k} vs fp32`, fmt(v)])) };
console.log(JSON.stringify(res, null, 1));
if (outFile) fs.writeFileSync(outFile, JSON.stringify(res, null, 1) + "\n");
// Exit status: the fp32 build must match the PyTorch reference (the layer, pooling and head are the trained ones). Lower precisions are
// reported against the written tolerance, not required: whether a quantised mid-layer state keeps the head output is measured per model.
process.exit(a.pass ? 0 : 1);
