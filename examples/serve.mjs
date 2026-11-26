#!/usr/bin/env node
// Static server for the text demos (roteador/, contrato/, sentinela/, policy-checker/), 127.0.0.1 only.
// It serves the page files in this folder and nothing else: the library, onnxruntime-web, transformers.js and the
// models all load from the internet at pinned versions (see each page's import map). No COOP/COEP headers: the demos
// run onnxruntime-web with one thread (no SharedArrayBuffer), and COEP would block the CDN and Hub requests.
// Usage: node examples/serve.mjs [--port 8787]
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".md": "text/plain; charset=utf-8" };
const HIDDEN = new Set(["serve.mjs", "serve-media.mjs", "audio-turn"]); // the audio demo needs local weights: use serve-media.mjs

/** A path inside `root` (never the root's prefix-sharing siblings). */
const inside = (root, file) => { const rel = path.relative(root, file); return rel !== ".." && !rel.startsWith(".." + path.sep) && !path.isAbsolute(rel); };

export function createServer() {
  return http.createServer((req, res) => {
    let url;
    try { url = decodeURIComponent(new URL(req.url, "http://x").pathname); } catch { res.writeHead(400); return res.end(); }
    let file = path.resolve(HERE, "." + url);
    if (!inside(HERE, file) || HIDDEN.has(path.relative(HERE, file).split(path.sep)[0])) { res.writeHead(403); return res.end(); }
    if (fs.statSync(file, { throwIfNoEntry: false })?.isDirectory()) file = path.join(file, "index.html");
    const st = fs.statSync(file, { throwIfNoEntry: false });
    if (!st?.isFile()) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream", "Content-Length": st.size, "Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff" });
    fs.createReadStream(file).pipe(res);
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const i = process.argv.indexOf("--port");
  const port = i > 0 ? Number(process.argv[i + 1]) : 8787;
  createServer().listen(port, "127.0.0.1", () =>
    console.log(`demos on http://127.0.0.1:${port}/  (roteador/, contrato/, sentinela/, policy-checker/)`));
}
