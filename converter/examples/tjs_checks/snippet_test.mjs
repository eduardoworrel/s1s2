// Runs the card snippet (snippet.js, verbatim) against a local copy of the repo and compares with measure.mjs output.
// usage: node snippet_test.mjs <tjs dist module> <models root> <repo> <items.jsonl> <ref.json> [tokonly]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
const SC = path.dirname(fileURLToPath(import.meta.url));
const [tjsMod, root, repo, itemsPath, refPath, tokonly] = process.argv.slice(2);
const T = await import(pathToFileURL(tjsMod).href);
T.env.allowRemoteModels = false; T.env.localModelPath = root + "/";
globalThis.__T = T;
const realFetch = globalThis.fetch;
globalThis.fetch = async (u) => {
  const pre = `https://huggingface.co/${repo}/resolve/main/`;
  if (String(u).startsWith(pre)) return new Response(fs.readFileSync(path.join(root, repo, String(u).slice(pre.length))));
  return realFetch(u);
};
const src = fs.readFileSync(path.join(SC, "../../s1s2_convert/node/tjs_plain_decide.js"), "utf8");
let [setup, decide] = src.split("//--decide--\n");
const rows = fs.readFileSync(itemsPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const ref = JSON.parse(fs.readFileSync(refPath, "utf8")).rows;
const byMan = new Map();
for (const r of rows) { if (!byMan.has(r.manifest)) byMan.set(r.manifest, []); byMan.get(r.manifest).push(r); }
let same = 0, posSame = 0, n = 0, maxdp = 0;
let model = null;
for (const [mname, rs] of byMan) {
  let s = setup.replace('import { AutoModelForCausalLM, AutoTokenizer } from "@huggingface/transformers";', "const { AutoModelForCausalLM, AutoTokenizer } = globalThis.__T;")
    .replace("__REPO__", repo).replace("__MANIFEST__", mname);
  if (tokonly) s = s.replace(/const model = await AutoModelForCausalLM[^\n]*\n/, "");
  else if (model) s = s.replace(/const model = await AutoModelForCausalLM[^\n]*\n/, "const model = globalThis.__model;\n");
  let d = decide.replace(/^const state = [^\n]*\n/, "");
  if (tokonly) d = d.split("// 3.")[0] + "return { read };\n";
  else d = d.replace("console.log(options[best], probs[best]);", "return { read, choice: options[best], p: probs[best], probs };");
  const code = `${s}\nexport const getModel = () => ${tokonly ? "null" : "model"};\nexport async function run(state, instructions, options) {\n${d}\n}\n`;
  const f = path.join(SC, `.snip-${process.pid}.mjs`);
  fs.writeFileSync(f, code);
  const mod = await import(pathToFileURL(f).href + "?" + mname);
  fs.unlinkSync(f);
  if (!tokonly) { model = mod.getModel(); globalThis.__model = model; }
  for (const r of rs) {
    const a = await mod.run(r.input, r.question.instructions, r.question.choices);
    const g = ref.find((x) => x.id === r.id);
    n++;
    const pOk = JSON.stringify(a.read) === JSON.stringify(g.positions);
    posSame += pOk;
    if (!tokonly) { same += a.choice === g.head; maxdp = Math.max(maxdp, ...a.probs.map((p, i) => Math.abs(p - g.headProbs[i]))); }
    console.error(r.id, pOk ? "pos ok" : `pos ${a.read} vs ${g.positions}`, tokonly ? "" : `${a.choice} vs ${g.head}`);
  }
}
console.log(JSON.stringify({ repo, items: n, positionsEqual: posSame, ...(tokonly ? {} : { decisionsEqual: same, probMaxAbsDiff: maxdp }) }));
