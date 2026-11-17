// s1s2 reference on Node CPU: head decision + s1s2 label-logits decision ("both"), read positions, head metas.
// usage: node measure.mjs <lib> <model dir> <items.jsonl> <out.json>
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
const [lib, dir, itemsPath, outPath] = process.argv.slice(2);
const S = await import(pathToFileURL(path.join(lib, "dist/index.js")).href);
const rows = fs.readFileSync(itemsPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const groups = new Map();
for (const r of rows) { if (!groups.has(r.manifest)) groups.set(r.manifest, []); groups.get(r.manifest).push(r); }
const out = [], metas = {};
for (const [mname, rs] of groups) {
  const man = JSON.parse(fs.readFileSync(path.join(dir, mname), "utf8"));
  const [build] = Object.keys(man.builds);
  const b = man.builds[build];
  for (const [k, h] of Object.entries(b.heads)) {
    const buf = fs.readFileSync(path.join(dir, h.file.path)); const n = buf.readUInt32LE(0);
    metas[k] = { ...JSON.parse(buf.subarray(4, 4 + n).toString()), manifestScale: h.scale ?? null };
  }
  const model = await S.S1S2.load({ manifest: man, build, device: "cpu", localDirs: { [man.tokenizer.repo]: dir, [b.repo]: dir } });
  const e = model.engine;
  for (const r of rs) {
    const nq = S.normalizeQuestions([{ ...r.question, method: "head" }], e.manifest)[0];
    const plan = S.planQuestion(e, nq, S.inputToText(r.input));
    const res = await e.backend.prefill(plan.enc.ids, { positions: plan.positions }, { keep: false });
    const a = plan.read(res);
    out.push({ id: r.id, manifest: mname, ids: plan.enc.ids, positions: plan.positions, head: a.choice,
      headProbs: a.probabilities.map((p) => p.probability) });
    console.error(r.id, a.choice);
  }
  await model.dispose();
}
fs.writeFileSync(outPath, JSON.stringify({ metas, rows: out }));
