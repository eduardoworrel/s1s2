// Parity on real weights: s1s2 decide() vs ct0-edge 1.0.0 askMany() on the same decode-only q4 session.
// ct0-edge is imported read-only from its built dist (CT0_EDGE_DIR, a built ct0-edge checkout). Heavy: about 4 GB of RAM.
import * as ort from "onnxruntime-node";
import * as tjs from "@huggingface/transformers";
import { homedir } from "node:os";
import { join } from "node:path";
import { S1S2, Files, createDecodeOnly as libDecodeOnly, loadHead } from "../../dist/index.js";
import { CT0 } from "../helpers.mjs";

const edge = process.env.CT0_EDGE_DIR ?? "../ct0-edge";
const { createDecodeOnly } = await import(join(edge, "dist/decodeOnly.js"));
const { askMany } = await import(join(edge, "dist/ask.js"));
const { loadHead: edgeLoadHead } = await import(join(edge, "dist/systemone.js"));

const repo = "eduardoworrel/alias3-micro-1026-ONNX-decode-only";
const dir = join(homedir(), ".cache/ct0-edge/huggingface.co", repo, "resolve/main");
const b = CT0.builds["decode-only-q4"];
const files = new Files({ platform: "node", localDirs: { [repo]: dir } });
const t = CT0.tokenizer;
const json = (p) => files.json(t.repo, t.revision, p);
const rawTok = new tjs[(await json(t.files.config)).tokenizer_class](await json(t.files.tokenizer), await json(t.files.config));
const ab = async (p) => { const u = await files.bytes(b.repo, b.revision, p); return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength); };
const session = await ort.InferenceSession.create(await files.local(b.repo, b.revision, b.files.model), { executionProviders: ["cpu"], logSeverityLevel: 3 });

const edgeBe = await createDecodeOnly(session, rawTok, { ort, skipVersionCheck: true });
const edgeHeads = { noulChoice: edgeLoadHead(await ab(b.heads.predicateChoice.file)), score: edgeLoadHead(await ab(b.heads.score.file)) };
const { tokenizerFromJson } = await import("../../dist/index.js");
const tok = tokenizerFromJson(tjs, await json(t.files.tokenizer), await json(t.files.config));
const libBe = await libDecodeOnly(session, { ort, eos: new Set(CT0.eos.map((x) => x.id)), hiddenOutput: b.hiddenOutput, skipVersionCheck: true });
const heads = { predicate: loadHead(await ab(b.heads.predicateChoice.file)), score: loadHead(await ab(b.heads.score.file)) };
heads.choice = heads.predicate;
const lib = S1S2.fromBackend(CT0, "decode-only-q4", libBe, tok, heads);

const state = '<form action="http://x.example/login" method="get"><input name="password" type="password"></form>';
const q = "Does this excerpt itself contain a form submitting sensitive data insecurely?";
const options = ["insecure-forms", "dom-xss", "sensitive-path", "exfil-egress"];
const t0 = Date.now();
const [en, ec, es] = await askMany(edgeBe, rawTok, edgeHeads, state, [{ kind: "noul", q }, { kind: "choice", options }, { kind: "score" }], { method: "both" });
console.log(`ct0-edge askMany: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
const t1 = Date.now();
const s = await lib.decide({ input: state, questions: [
  { name: "n", type: "noul", instructions: q }, { name: "c", type: "choice", choices: options }, { name: "s", type: "score" },
] }, { method: "both", keepState: false });
console.log(`s1s2 decide:   ${((Date.now() - t1) / 1000).toFixed(1)}s`);
const [n, c, sc] = s.answers;
const rows = [
  ["predicate head p", en.p, n.probability], ["predicate logits p", en.logits.p, n.logits.probability],
  ...options.map((o, i) => [`choice head ${o}`, ec.choice[o], c.probabilities[i].probability]),
  ...options.map((o, i) => [`choice logits ${o}`, ec.logits.choice[o], c.logits.probabilities[i].probability]),
  ["score head raw", es.raw, sc.raw], ["score head level (1-based)", es.level, Number(sc.level)],
  ["score logits expected (1-based)", es.logits.expected, sc.logits.score + 1],
];
let maxd = 0;
for (const [k, a, bb] of rows) { const d = Math.abs(a - bb); maxd = Math.max(maxd, d); console.log(`${k.padEnd(34)} ct0-edge ${(+a).toFixed(6)}  s1s2 ${(+bb).toFixed(6)}  |d| ${d.toExponential(2)}`); }
console.log(maxd < 1e-5 ? `PARITY OK (max |d| ${maxd.toExponential(2)})` : `PARITY FAIL (max |d| ${maxd})`);
await session.release();
process.exit(maxd < 1e-5 ? 0 : 1);
