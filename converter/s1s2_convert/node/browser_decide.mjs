// Load a packed WebLLM (MLC) or LiteRT.js S1S2 folder in Chrome (WebGPU) with s1s2 and decide a few prompts.
// The folder is served locally (baseUrls), so it need not be on the Hub yet; its manifests must be pinned to any 40-hex
// revision (`pack_mlc.py pin --revision <sha>`). One model load per manifest, then every row of that manifest.
//
// usage (the s1s2 checkout needs `npm ci && npm run build`): node browser_decide.mjs --lib <s1s2 checkout> --folder <packed folder> --rows rows.jsonl --out out.json
//          --modules <node_modules dir with @mlc-ai/web-llm and/or @litertjs/core> [--playwright <dir of the playwright package>]
//          [--methods head,logits] [--profile <chrome profile dir>]
// rows: {id, manifest, input, question}  (same file as pack_tjs parity; see examples/pack_tjs_items.py)
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import os from "node:os";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const a = {};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].slice(2)] = process.argv[i + 1];
for (const k of ["lib", "folder", "rows", "out", "modules"]) if (!a[k]) throw new Error(`--${k} is required`);
const methods = (a.methods ?? "head,logits").split(",");
const rows = fs.readFileSync(a.rows, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const pwDir = a.playwright ?? path.dirname(createRequire(path.join(process.cwd(), "x.js")).resolve("playwright/package.json"));
const { chromium } = await import(pathToFileURL(path.join(pwDir, "index.mjs")).href);

const nm = path.join(a.lib, "node_modules");
let ortc = path.join(nm, "onnxruntime-web/node_modules/onnxruntime-common/dist/esm");
if (!fs.existsSync(ortc)) ortc = path.join(nm, "onnxruntime-common/dist/esm");
const routes = { "/tjs/": path.join(nm, "@huggingface/transformers/dist"), "/ort/": path.join(nm, "onnxruntime-web/dist"), "/ortc/": ortc, "/d/": path.join(a.lib, "dist"), "/pack/": a.folder, "/pack/resolve/main/": a.folder,  // WebLLM appends resolve/main/ to the model URL
  "/webllm/": path.join(a.modules, "@mlc-ai/web-llm"),
  "/litert/": path.join(a.modules, "@litertjs/core"), "/litertu/": path.join(a.modules, "@litertjs/wasm-utils") };
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const srv = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const pre = Object.keys(routes).sort((x, y) => y.length - x.length).find((p) => url.startsWith(p));
  if (url === "/index.html") { res.writeHead(200, { "Content-Type": "text/html" }); return res.end(PAGE); }
  if (!pre) { console.error("404", url); res.writeHead(404); return res.end(); }
  const f = path.join(routes[pre], url.slice(pre.length));
  const st = fs.statSync(f, { throwIfNoEntry: false });
  if (!st?.isFile()) { console.error("404", url); res.writeHead(404); return res.end(); }
  const hdr = { "Content-Type": TYPES[path.extname(f)] ?? "application/octet-stream", "Accept-Ranges": "bytes", "Access-Control-Allow-Origin": "*" };
  const m = /bytes=(\d+)-(\d+)?/.exec(req.headers.range ?? "");
  if (m) {
    const s0 = +m[1], e0 = m[2] ? Math.min(+m[2], st.size - 1) : st.size - 1;
    res.writeHead(206, { ...hdr, "Content-Range": `bytes ${s0}-${e0}/${st.size}`, "Content-Length": e0 - s0 + 1 });
    return fs.createReadStream(f, { start: s0, end: e0 }).pipe(res);
  }
  res.writeHead(200, { ...hdr, "Content-Length": st.size });
  fs.createReadStream(f).pipe(res);
});
const PAGE = `<!doctype html><meta charset="utf-8"><script type="importmap">
{ "imports": { "onnxruntime-web": "/ort/ort.webgpu.min.mjs", "onnxruntime-web/webgpu": "/ort/ort.webgpu.min.mjs", "onnxruntime-common": "/ortc/index.js",
  "@huggingface/transformers": "/tjs/transformers.web.min.js", "@mlc-ai/web-llm": "/webllm/lib/index.js", "@litertjs/core": "/litert/dist/index.js", "@litertjs/wasm-utils": "/litertu/dist/index.js" } }
</script><script type="module">
import { S1S2, normalizeQuestions } from "/d/index.js";
import * as ort from "onnxruntime-web/webgpu"; import * as transformers from "@huggingface/transformers";
ort.env.wasm.wasmPaths = "/ort/";
let webllm = null, litert = null;
try { webllm = await import("@mlc-ai/web-llm"); } catch {}
try { litert = await import("@litertjs/core"); } catch {}
let model = null;
window.load = async (man, baseUrls, build) => { const t0 = performance.now(); try { if (model) { await model.dispose?.(); model = null; }
  model = await S1S2.load({ manifest: man, build, device: "webgpu", baseUrls, modules: { ort, transformers, webllm, litert }, litert: { wasmUrl: "/litert/wasm/", quiet: true } });
  return { ok: true, ms: Math.round(performance.now() - t0), build: model.build, runtime: model.runtime, hidden: model.hiddenOutput, heads: model.heads }; }
  catch (e) { return { ok: false, err: String(e?.stack ?? e).slice(0, 1500) }; } };
window.ask = async (input, question, method) => { const t0 = performance.now(); try {
  const s = await model.decide({ input, questions: [{ ...question, method }] }, { keepState: false }); const x = s.answers[0]; await s.dispose?.();
  return { ok: true, ms: Math.round(performance.now() - t0), choice: x.choice, p: x.probabilities.map((y) => y.probability), raw: x.raw };
} catch (e) { return { ok: false, err: String(e?.stack ?? e).slice(0, 1500) }; } };
window.ready = true;
</script>`;
await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
const base = `http://127.0.0.1:${srv.address().port}`;
const profile = a.profile ?? fs.mkdtempSync(path.join(os.tmpdir(), "s1s2-chrome-"));
const ctx = await chromium.launchPersistentContext(profile, { channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--ignore-gpu-blocklist", "--js-flags=--max-old-space-size=4096"] });
const page = ctx.pages()[0] ?? await ctx.newPage();
const logs = [];
page.on("console", (m) => { if (logs.length < 200) logs.push(m.text().slice(0, 300)); });
page.on("pageerror", (e) => logs.push("PAGEERR " + e.message));
page.setDefaultTimeout(0);
await page.goto(base + "/index.html");
await page.waitForFunction(() => window.ready === true);
const out = { loads: [], rows: [] };
const save = () => fs.writeFileSync(a.out, JSON.stringify({ ...out, logs }, null, 1));
for (const mname of [...new Set(rows.map((r) => r.manifest))]) {
  const man = JSON.parse(fs.readFileSync(path.join(a.folder, mname), "utf8"));
  const baseUrls = { [man.tokenizer.repo]: base + "/pack/" };
  for (const b of Object.values(man.builds)) baseUrls[b.repo] = base + "/pack/";
  const L = await page.evaluate(([m, u, b]) => window.load(m, u, b), [man, baseUrls, Object.keys(man.builds)[0]]);
  out.loads.push({ manifest: mname, ...L });
  console.log("load", mname, JSON.stringify(L).slice(0, 300));
  save();
  if (!L.ok) continue;
  for (const r of rows.filter((x) => x.manifest === mname)) for (const me of methods) {
    if (me === "head" && !L.heads?.length) continue;
    const x = await page.evaluate(([i, q, m]) => window.ask(i, q, m), [r.input, r.question, me]);
    out.rows.push({ id: r.id, manifest: mname, method: me, ...x });
    console.log(r.id, me, x.ok ? `${x.choice} ${x.p.map((v) => v.toFixed(3)).join("/")} ${x.ms} ms` : "ERR " + x.err.slice(0, 200));
  }
  save();
}
save();
await ctx.close();
srv.close();
