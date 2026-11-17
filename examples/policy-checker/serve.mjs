#!/usr/bin/env node
// Local static server for the Privacy Policy Checker (127.0.0.1 only; COOP/COEP for WebGPU/wasm threads).
// Serves the page, s1s2/dist as ./lib, onnxruntime-web and transformers.js from node_modules as ./vendor,
// s1s2/manifests as ./manifests, and local model mirrors as ./models/<repo>/ (listed in ./models.json so the
// page passes them to S12.load as baseUrls; every file is still sha256-checked against the manifest).
// Usage: node serve.mjs [--port 8787] [--mirror <repo>=<dir>]...
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(HERE, "../..");
const NM = path.join(LIB, "node_modules/@huggingface/transformers");
const argv = process.argv.slice(2);
let port = 8787;
const mirrors = {};
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--port") port = Number(argv[++i]);
  else if (argv[i] === "--mirror") { const [repo, dir] = argv[++i].split("="); mirrors[repo] = path.resolve(dir); }
}
const routes = {
  "/lib/": path.join(LIB, "dist"),
  "/vendor/ort/": path.join(NM, "node_modules/onnxruntime-web/dist"),
  "/vendor/ortc/": path.join(NM, "node_modules/onnxruntime-common/dist/esm"),
  "/vendor/tjs/": path.join(NM, "dist"),
  "/manifests/": path.join(LIB, "manifests"),
  "/": HERE,
};
const modelsJson = {};
for (const [repo, dir] of Object.entries(mirrors)) {
  const r = `/models/${repo.replace(/\//g, "__")}/`;
  routes[r] = dir;
  modelsJson[repo] = "." + r;
}
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".png": "image/png" };
http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const head = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "same-origin" };
  if (url === "/models.json") { res.writeHead(200, { ...head, "Content-Type": "application/json" }); return res.end(JSON.stringify(modelsJson)); }
  const pre = Object.keys(routes).sort((a, b) => b.length - a.length).find((p) => url.startsWith(p));
  const root = routes[pre];
  let file = path.resolve(root, "." + url.slice(pre.length - 1));
  if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
  if (fs.statSync(file, { throwIfNoEntry: false })?.isDirectory()) file = path.join(file, "index.html");
  const st = fs.statSync(file, { throwIfNoEntry: false });
  if (!st?.isFile()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { ...head, "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream", "Content-Length": st.size, "Cache-Control": "no-cache" });
  fs.createReadStream(file).pipe(res);
}).listen(port, "127.0.0.1", () => console.log(`policy checker on http://127.0.0.1:${port}/`));
