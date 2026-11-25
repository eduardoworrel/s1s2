// Integration smoke test (real weights, Node CPU): CT0 decode-only q4 through S1S2.load.
// Uses the files ct0-edge already cached (read-only) via localDirs. Heavy: about 4 GB of RAM.
import { S1S2 } from "../../dist/index.js";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const t0 = Date.now(), dt = () => ((Date.now() - t0) / 1000).toFixed(1) + "s";
const repo = "eduardoworrel/alias3-micro-1026-ONNX-decode-only";
const localDir = process.env.CT0_DECODE_ONLY_DIR ?? join(homedir(), ".cache/ct0-edge/huggingface.co", repo, "resolve/main");
const lib = await S1S2.load({
  manifest: fileURLToPath(new URL("../../manifests/ct0-alias3-micro-1026.json", import.meta.url)),
  build: "decode-only-q4", localDirs: { [repo]: localDir },
  onStatus: (m) => console.log(`[${dt()}] ${m}`),
});
console.log(`[${dt()}] loaded ${lib.build} (${lib.runtime}/${lib.device}) hidden=${lib.hiddenOutput} heads=${lib.heads} sharedPrefill=${lib.sharedPrefill}`);

const state = '<input name="password" type="password" autocomplete="on">';
const s = await lib.decide({
  input: state,
  questions: [
    { name: "insecure_form", type: "noul", instructions: "Does this excerpt itself contain a form submitting sensitive data insecurely?", method: "both" },
    { name: "severity", type: "score" },
  ],
});
console.log(`[${dt()}] answers`, JSON.stringify(s.answers, (k, v) => (typeof v === "number" ? +v.toFixed(4) : v), 1));
const fail = [];
const [p, sc] = s.answers;
if (!(p.type === "predicate" && p.probability >= 0 && p.probability <= 1 && Number.isFinite(p.raw) && p.logits && Number.isFinite(p.logits.probability))) fail.push("predicate shape");
if (!(sc.type === "score" && sc.method === "head" && Number.isFinite(sc.score) && sc.score >= 0 && sc.score <= 4 && typeof sc.level === "string")) fail.push("score shape");
if (s.sharedPrefill !== true) fail.push("session.sharedPrefill should be true on decode-only");
const g = await s.generate("In one sentence, why?", { maxNewTokens: 24, temperature: 0, onToken: (t) => process.stdout.write(t) });
console.log(`\n[${dt()}] generate: sharedPrefill=${g.sharedPrefill} reusedTokens=${g.reusedTokens} tokens=${g.tokenIds.length} finish=${g.finishReason}`);
if (!(g.sharedPrefill && g.reusedTokens > 50 && g.text.trim().length > 0)) fail.push("session.generate shared prefill / text");
await s.dispose();
await lib.dispose();
console.log(fail.length ? `SMOKE FAIL: ${fail.join("; ")}` : `SMOKE OK (${dt()})`);
process.exit(fail.length ? 1 : 0);
