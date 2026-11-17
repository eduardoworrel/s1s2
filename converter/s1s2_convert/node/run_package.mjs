// Runtime side of the converter's parity check: load a package with s1s2 (S12.load, sha256-checked files, load-time
// token checks, heads) on onnxruntime-node and answer each row with method "both" (+ evidence when the row has it).
// usage: node run_package.mjs <lib dir> <package dir> <build> <rows.jsonl> <out.jsonl>
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [lib, pkg, build, rowsPath, outPath] = process.argv.slice(2);
const { S12 } = await import(pathToFileURL(path.join(lib, "dist/index.js")).href);
const manifestPath = path.join(pkg, "manifest.json");
const man = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const t0 = Date.now();
const s12 = await S12.load({ manifest: manifestPath, build, localDirs: { [man.tokenizer.repo]: pkg, [man.builds[build].repo]: pkg },
  cacheDir: process.env.S12_CACHE, onStatus: (m) => console.error("[s1s2]", m) });
console.error(`[s1s2] loaded ${s12.build} (${s12.runtime}, ${s12.device}) heads=${s12.heads.join(",")} in ${Date.now() - t0} ms`);
const rows = fs.readFileSync(rowsPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const out = [];
for (const r of rows) {
  const s = await s12.decide({ input: r.input, questions: [{ ...r.question, method: "both" }], ...(r.evidence ? { evidence: r.evidence } : {}) }, { keepState: false });
  const a = s.answers[0];
  out.push(JSON.stringify({ id: r.id, head: { probabilities: a.probabilities.map((p) => p.probability), choice: a.choice, margin: a.margin, rawMargin: a.rawMargin },
    logits: { probabilities: a.logits.probabilities.map((p) => p.probability), choice: a.logits.choice, margin: a.logits.margin, rawMargin: a.logits.rawMargin },
    ...(s.evidence ? { evidence: { index: s.evidence.index, probabilities: s.evidence.probabilities, margin: s.evidence.margin } } : {}) }));
  await s.dispose();
}
fs.writeFileSync(outPath, out.join("\n") + "\n");
await s12.dispose();
console.error(`[s1s2] ${rows.length} rows answered in ${Date.now() - t0} ms`);
