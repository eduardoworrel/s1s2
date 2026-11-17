#!/usr/bin/env node
// Smoke: the public S1S2.load on the browser CPU path. Headless Chrome with WebGPU hidden (navigator.gpu undefined, no
// GPU), cross-origin isolated page (ORT's default threads, as in lab-s12 study 011 arm C), the built dist/index.js with
// its default imports resolved by an import map ("onnxruntime-web/wasm" -> ort.wasm.min.mjs, the bundle the study
// measured). Loads a published manifest from the Hub (or a local JSON with --manifest-file), asserts device "wasm",
// that the WASM binary used is ort-wasm-simd-threaded.wasm (not the WebGPU/JSEP one), and runs one decision.
//   PLAYWRIGHT_DIR=<.../node_modules/playwright> node test/smoke/browser_wasm.mjs [--manifest URL | --manifest-file F]
//     [--question "text"] [--input "text"]
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const MANIFEST = arg("--manifest", "https://huggingface.co/eduardoworrel/Qwen3-0.6B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2/resolve/main/manifest.clinc.json");
const MANIFEST_FILE = arg("--manifest-file");
const INPUT = arg("--input", "Can you book me a table for two at an Italian place tonight?");
const QUESTION = arg("--question", "Is this request in scope for a banking assistant?");
const PW = process.env.PLAYWRIGHT_DIR ?? "playwright";

const nm = (p) => path.join(ROOT, "node_modules", p);
const routes = {
  "/lib/": path.join(ROOT, "dist"),
  "/vendor/ort/": nm("onnxruntime-web/dist"),
  "/vendor/ortc/": fs.existsSync(nm("onnxruntime-web/node_modules/onnxruntime-common/dist/esm")) ? nm("onnxruntime-web/node_modules/onnxruntime-common/dist/esm") : nm("onnxruntime-common/dist/esm"),
  "/vendor/tjs/": nm("@huggingface/transformers/dist"),
};
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const page = `<!doctype html><html><head><meta charset="utf-8"><script type="importmap">
{ "imports": { "onnxruntime-web/wasm": "/vendor/ort/ort.wasm.min.mjs", "onnxruntime-web/webgpu": "/vendor/ort/ort.webgpu.min.mjs",
  "onnxruntime-web": "/vendor/ort/ort.min.mjs", "onnxruntime-common": "/vendor/ortc/index.js", "@huggingface/transformers": "/vendor/tjs/transformers.web.js" } }
</script></head><body><script type="module">
import { S1S2 } from "/lib/index.js";
window.run = async (manifest, input, question) => {
  const t0 = performance.now();
  const lib = await S1S2.load({ manifest });
  const msLoad = performance.now() - t0;
  const q = { name: "q", type: "choice", choices: lib.manifest.lab?.options ?? [" Yes", " No"], instructions: question };
  const t1 = performance.now();
  const s = await lib.decide({ input, questions: [q] });
  const msDecide = performance.now() - t1;
  const a = s.answers[0];
  await s.dispose(); await lib.dispose();
  const wasm = performance.getEntriesByType("resource").map((e) => e.name).filter((n) => n.endsWith(".wasm")).map((n) => n.split("/").pop());
  return { device: lib.device, runtime: lib.runtime, build: lib.build, fallbackReason: lib.fallbackReason, msLoad: Math.round(msLoad), msDecide: Math.round(msDecide),
    answer: { choice: a.choice, confidence: a.confidence, margin: a.margin, path: a.path }, wasm,
    env: { gpu: typeof navigator.gpu, crossOriginIsolated: self.crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency } };
};
window.ready = true;
</script></body></html>`;

const srv = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const hdr = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "same-origin" };
  if (url === "/") { res.writeHead(200, { ...hdr, "Content-Type": "text/html" }); return res.end(page); }
  const pre = Object.keys(routes).find((p) => url.startsWith(p));
  const file = pre && path.resolve(routes[pre], url.slice(pre.length));
  if (!file || !file.startsWith(path.resolve(routes[pre])) || !fs.statSync(file, { throwIfNoEntry: false })?.isFile()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { ...hdr, "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
});
await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
const pw = PW === "playwright" ? await import("playwright") : await import(pathToFileURL(path.join(PW, "index.mjs")).href);
const browser = await pw.chromium.launch({ channel: "chrome", headless: true, args: ["--disable-features=WebGPU", "--disable-gpu"] });
try {
  const ctx = await browser.newContext();
  // Chrome may keep navigator.gpu despite the flags: hide it, as a browser without WebGPU looks to JS.
  await ctx.addInitScript(() => { Object.defineProperty(Navigator.prototype, "gpu", { get: () => undefined, configurable: true }); });
  const p = await ctx.newPage();
  p.on("pageerror", (e) => console.error("pageerror:", e.message));
  await p.goto(`http://127.0.0.1:${srv.address().port}/`);
  await p.waitForFunction(() => window.ready === true, null, { timeout: 60000 });
  const manifest = MANIFEST_FILE ? JSON.parse(fs.readFileSync(MANIFEST_FILE, "utf8")) : MANIFEST;
  const r = await p.evaluate(([m, i, q]) => window.run(m, i, q), [manifest, INPUT, QUESTION]);
  console.log(JSON.stringify({ chrome: browser.version(), manifest: MANIFEST_FILE ?? MANIFEST, ...r }, null, 1));
  const ok = r.device === "wasm" && r.wasm.includes("ort-wasm-simd-threaded.wasm") && !r.wasm.some((n) => n.includes("jsep")) && Number.isFinite(r.answer.confidence);
  if (!ok) { console.error("FAIL: expected device wasm on ort-wasm-simd-threaded.wasm and a finite answer"); process.exitCode = 1; }
  else console.log("OK: S1S2.load on the browser CPU (WASM)");
} finally { await browser.close(); srv.close(); }
