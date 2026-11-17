// transformers.js PR #1943 pipeline("decision") on the same items; agreement with the s1s2 head decision.
// usage: node pr_decision.mjs <PR dist transformers.node.mjs> <models root> <repo> <items.jsonl> <ref.json> <out.json>
import fs from "node:fs";
import { pathToFileURL } from "node:url";
const [distMod, root, repo, itemsPath, refPath, outPath] = process.argv.slice(2);
const T = await import(pathToFileURL(distMod).href);
T.env.allowRemoteModels = false; T.env.localModelPath = root + "/";
const rows = fs.readFileSync(itemsPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const ref = JSON.parse(fs.readFileSync(refPath, "utf8")).rows;
const decide = await T.pipeline("decision", repo, { dtype: "q4", device: "cpu" });
// AutoTokenizer does not read chat_template.jinja and the root tokenizer_config.json has no chat_template: set it (card does the same).
const hadTemplate = !!decide.tokenizer.chat_template;
decide.tokenizer.chat_template ??= fs.readFileSync(`${root}/${repo}/chat_template.jinja`, "utf8");
let agree = 0, agreeS1S2Logits = 0;
const out = [];
for (const r of rows) {
  const opts = r.question.choices;
  const criteria = Object.fromEntries(opts.map((o) => [o.trim(), null]));
  const a = (await decide({ state: r.input, questions: { q: { type: "choice", instructions: r.question.instructions, criteria } } })).q;
  const choice = opts[opts.map((o) => o.trim()).indexOf(a.choice)];
  const g = ref.find((x) => x.id === r.id);
  agree += choice === g.head;
  out.push({ id: r.id, pipeline: choice, confidence: a.confidence, head: g.head });
  console.error(r.id, choice, "| head", g.head);
}
await decide.dispose();
const summary = { repo, hadTemplate, items: rows.length, pipelineAgreesWithHead: agree, rows: out };
fs.writeFileSync(outPath, JSON.stringify(summary, null, 1));
console.log(JSON.stringify({ repo, items: rows.length, pipelineAgreesWithHead: agree }));
