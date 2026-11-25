#!/usr/bin/env node
// Local static server for the single-page demos (127.0.0.1 only; COOP/COEP for WebGPU/wasm threads).
// /<demo>/ is examples/<demo>/, /lib/ is dist/, /vendor/ holds onnxruntime-web and transformers.js from node_modules,
// /models.json lists local model mirrors (repo -> ./models/<repo>/), which the pages pass to S1S2.load as baseUrls;
// every file is still sha256-checked against the manifest.
// Usage: node examples/serve.mjs [--port 8787] [--mirror <repo>=<dir>]...
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(HERE, "..");
const NM = path.join(LIB, "node_modules");
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".png": "image/png" };

export function createServer(mirrors = {}) {
  const routes = {
    "/lib/": path.join(LIB, "dist"),
    "/vendor/ort/": path.join(NM, "onnxruntime-web/dist"),
    "/vendor/ortc/": path.join(NM, "onnxruntime-web/node_modules/onnxruntime-common/dist/esm"),
    "/vendor/tjs/": path.join(NM, "@huggingface/transformers/dist"),
    "/": HERE,
  };
  const modelsJson = {};
  for (const [repo, dir] of Object.entries(mirrors)) {
    const r = `/models/${repo.replace(/\//g, "__")}/`;
    routes[r] = path.resolve(dir);
    modelsJson[repo] = r;
  }
  return http.createServer((req, res) => {
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
    const h = { ...head, "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream", "Cache-Control": "no-cache", "Accept-Ranges": "bytes" };
    const m = /bytes=(\d+)-(\d+)?/.exec(req.headers.range ?? "");
    if (m) {
      const a = Number(m[1]), b = m[2] ? Math.min(Number(m[2]), st.size - 1) : st.size - 1;
      res.writeHead(206, { ...h, "Content-Range": `bytes ${a}-${b}/${st.size}`, "Content-Length": b - a + 1 });
      return fs.createReadStream(file, { start: a, end: b }).pipe(res);
    }
    res.writeHead(200, { ...h, "Content-Length": st.size });
    fs.createReadStream(file).pipe(res);
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2);
  let port = 8787;
  const mirrors = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--port") port = Number(argv[++i]);
    else if (argv[i] === "--mirror") { const [repo, dir] = argv[++i].split("="); mirrors[repo] = dir; }
  }
  createServer(mirrors).listen(port, "127.0.0.1", () =>
    console.log(`demos on http://127.0.0.1:${port}/  (roteador/, contrato/, sentinela/)`));
}
