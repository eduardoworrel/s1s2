// One side of the pack_tjs parity check: load either the source ONNX Runtime GenAI build or the repackaged
// transformers.js build with s1s2 (sha256-checked files, load-time token checks, heads), on Node CPU, and for each row
// write the prompt ids, the read positions, the hidden state at those positions and the head's answer.
// The GenAI side runs one prefill per prompt (chunk larger than any prompt), like the transformers.js side.
// usage: node tjs_parity.mjs <genai|tjs> <lib dir> <source dir> <transformers.js dir> <rows.jsonl> <out.json>
// rows: {id, manifest, input, question}; `manifest` is the manifest file name at the root of both folders.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [side, lib, srcDir, tjsDir, rowsPath, outPath] = process.argv.slice(2);
if (!["genai", "tjs"].includes(side)) throw new Error("side: genai | tjs");
const S = await import(pathToFileURL(path.join(lib, "dist/index.js")).href);
const ver = (p) => { try { return JSON.parse(fs.readFileSync(path.join(lib, "node_modules", p, "package.json"), "utf8")).version; } catch { return null; } };
const rows = fs.readFileSync(rowsPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const groups = new Map();
for (const r of rows) { if (!groups.has(r.manifest)) groups.set(r.manifest, []); groups.get(r.manifest).push(r); }

const out = [];
for (const [mname, rs] of groups) {
  const dir = side === "genai" ? srcDir : tjsDir;
  const man = JSON.parse(fs.readFileSync(path.join(dir, mname), "utf8"));
  const [build] = Object.keys(man.builds);
  const b = man.builds[build];
  if (side === "genai" && !b.devices.includes("cpu")) b.devices = [...b.devices, "cpu"]; // opt-in for this check only
  const t0 = Date.now();
  const model = await S.S1S2.load({ manifest: man, build, device: "cpu", localDirs: { [man.tokenizer.repo]: dir, [b.repo]: dir },
    genai: { chunk: 1 << 16 }, cacheDir: process.env.S1S2_CACHE, onStatus: (m) => console.error(`[${side}]`, m) });
  console.error(`[${side}] ${mname}: ${model.build} (${model.runtime}, ${model.device}) heads=${model.heads.join(",")} in ${Date.now() - t0} ms`);
  const e = model.engine;
  for (const r of rs) {
    const nq = S.normalizeQuestions([{ ...r.question, method: "head" }], e.manifest)[0];
    const plan = S.planQuestion(e, nq, S.inputToText(r.input));
    const res = await e.backend.prefill(plan.enc.ids, { positions: plan.positions }, { keep: false });
    const a = plan.read(res);
    out.push({ id: r.id, manifest: mname, ids: plan.enc.ids, positions: plan.positions,
      hiddens: res.hiddens.map((h) => Array.from(h)), choice: a.choice, probabilities: a.probabilities.map((p) => p.probability) });
  }
  await model.dispose();
}
fs.writeFileSync(outPath, JSON.stringify({ side, versions: { s1s2: ver("@s1s2/web") ?? JSON.parse(fs.readFileSync(path.join(lib, "package.json"), "utf8")).version,
  "onnxruntime-node": ver("onnxruntime-node"), "@huggingface/transformers": ver("@huggingface/transformers"), node: process.version }, rows: out }));
console.error(`[${side}] ${out.length} rows`);
