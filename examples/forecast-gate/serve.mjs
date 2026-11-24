#!/usr/bin/env node
// Static server for this page (127.0.0.1): the page, s1s2/dist as ./lib, onnxruntime-web and transformers.js from node_modules as ./vendor,
// and your package directory as ./model (node serve.mjs --model <package dir> [--port 8788]). Start it yourself; nothing here runs it.
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url)), LIB = path.resolve(HERE, "../..");
const NM = path.join(LIB, "node_modules/@huggingface/transformers");
const argv = process.argv.slice(2), opt = (k, d) => (argv.includes(`--${k}`) ? argv[argv.indexOf(`--${k}`) + 1] : d);
const port = Number(opt("port", 8788)), model = opt("model") && path.resolve(opt("model"));
const routes = { "/lib/": path.join(LIB, "dist"), "/vendor/ort/": path.join(NM, "node_modules/onnxruntime-web/dist"), "/vendor/ortc/": path.join(NM, "node_modules/onnxruntime-common/dist/esm"),
  "/vendor/tjs/": path.join(NM, "dist"), ...(model ? { "/model/": model } : {}), "/": HERE };
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream" };
http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname === "/" ? "/index.html" : new URL(req.url, "http://x").pathname);
  const [prefix, root] = Object.entries(routes).sort((a, b) => b[0].length - a[0].length).find(([p]) => u.startsWith(p)) ?? [];
  const f = root && path.join(root, u.slice(prefix.length));
  if (!f || !f.startsWith(root) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404).end("not found"); return; }
  res.writeHead(200, { "content-type": types[path.extname(f)] ?? "application/octet-stream", "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp" });
  fs.createReadStream(f).pipe(res);
}).listen(port, "127.0.0.1", () => console.log(`http://127.0.0.1:${port}/ (model: ${model ?? "none"})`));
