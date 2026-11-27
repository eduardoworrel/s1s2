#!/usr/bin/env node
// Local static server for the audio demo (127.0.0.1 only; COOP/COEP for wasm threads; HTTP Range for the big graphs).
//   /                examples/
//   /lib/            dist/                         (npm run build first)
//   /manifests/      manifests/
//   /vendor/ort/     onnxruntime-web (the version the audio parity was measured with)
//   /weights/<pkg>/  the package directory <weights>/<pkg>/   (marimba-audio)
// The weights are NOT part of the repository and are not sent anywhere: point --weights at the directory that holds
// marimba-audio/ (see recipes/marimba-audio/README.md for how it is produced). Before it starts, the server checks the
// sha256 and size of every pinned file it will serve against the manifests (the page cannot hash 2 GB); --no-verify skips that.
// Usage: node examples/serve-media.mjs --weights <dir> [--port 8788] [--no-verify]
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(HERE, "..");
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".png": "image/png", ".jpg": "image/jpeg", ".wav": "audio/wav", ".onnx": "application/octet-stream", ".md": "text/plain; charset=utf-8" };

export function createMediaServer({ weights, extra = {}, canonical = false }) {
  const routes = {
    "/lib/": path.join(LIB, "dist"),
    "/manifests/": path.join(LIB, "manifests"),
    "/vendor/ort/": path.join(LIB, "node_modules/onnxruntime-web/dist"),
    "/": HERE,
  };
  for (const [r, d] of Object.entries(extra)) routes[r] = path.resolve(d);          // tests only: reference data served next to the pages
  if (weights) for (const pkg of ["marimba-audio"]) if (fs.existsSync(path.join(weights, pkg))) routes[`/weights/${pkg}/`] = path.resolve(weights, pkg);
  return http.createServer((req, res) => {
    try { handle(routes, req, res, canonical); }
    catch { if (!res.headersSent) res.writeHead(500); res.end(); }
  });
}

const HEAD = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "same-origin" };
const fail = (res, code, extra = {}) => { res.writeHead(code, { ...HEAD, ...extra }); res.end(); };

/** A path inside `root` (never the root's prefix-sharing siblings, e.g. dist-x next to dist). */
const inside = (root, file) => { const rel = path.relative(root, file); return rel !== ".." && !rel.startsWith(".." + path.sep) && !path.isAbsolute(rel); };

/** "bytes=a-b", "bytes=a-", "bytes=-n" -> [start, end] clamped to the file, or null when it cannot be satisfied. */
function parseRange(header, size) {
  const m = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!m || (m[1] === "" && m[2] === "")) return null;
  let a, b;
  if (m[1] === "") { const n = Number(m[2]); if (!(n > 0) || size === 0) return null; a = Math.max(0, size - n); b = size - 1; }
  else { a = Number(m[1]); b = m[2] === "" ? size - 1 : Math.min(Number(m[2]), size - 1); }
  return Number.isFinite(a) && Number.isFinite(b) && a <= b && a < size ? [a, b] : null;
}

function handle(routes, req, res, canonical) {
  let url;
  try { url = decodeURIComponent(new URL(req.url, "http://x").pathname); } catch { return fail(res, 400); }
  if (url.includes("\0")) return fail(res, 400);
  const pre = Object.keys(routes).sort((a, b) => b.length - a.length).find((p) => url.startsWith(p));
  if (pre === undefined) return fail(res, 404);
  const root = routes[pre];
  let file = path.resolve(root, "." + url.slice(pre.length - 1));
  if (!inside(root, file)) return fail(res, 403);
  if (fs.statSync(file, { throwIfNoEntry: false })?.isDirectory()) file = path.join(file, "index.html");
  const st = fs.statSync(file, { throwIfNoEntry: false });
  if (!st?.isFile()) return fail(res, 404);
  if (canonical) {
    file = fs.realpathSync(file);
    if (!inside(fs.realpathSync(root), file)) return fail(res, 403);
  }
  const h = { ...HEAD, "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream", "Cache-Control": "no-cache", "Accept-Ranges": "bytes" };
  let opts, status = 200, length = st.size;
  if (req.headers.range !== undefined) {
    const r = parseRange(req.headers.range, st.size);
    if (!r) return fail(res, 416, { "Content-Range": `bytes */${st.size}` });
    const [a, b] = r;
    opts = { start: a, end: b }; status = 206; length = b - a + 1; h["Content-Range"] = `bytes ${a}-${b}/${st.size}`;
  }
  const stream = fs.createReadStream(file, opts);
  stream.on("error", () => res.destroy());
  res.on("close", () => stream.destroy());
  res.writeHead(status, { ...h, "Content-Length": length });
  stream.pipe(res);
}

/** sha256 + size check of every pinned file of the media manifests against the weights directory. Returns {checked, problems}. */
export async function verifyWeights(weights) {
  const { validateManifestFile } = await import(pathToFileURL(path.join(LIB, "dist/package.js")).href);
  const out = { checked: 0, problems: [] };
  for (const [manifest, pkg] of [["marimba-audio.json", "marimba-audio"]]) {
    const dir = path.join(weights, pkg);
    if (!fs.existsSync(dir)) continue;
    const m = JSON.parse(fs.readFileSync(path.join(LIB, "manifests", manifest), "utf8"));
    const r = await validateManifestFile(path.join(LIB, "manifests", manifest), { strict: true, roots: { [m.repo]: dir } });
    out.checked += r.checked.length; out.problems.push(...r.errors.map((e) => `${manifest}: ${e}`));
  }
  return out;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2);
  const opt = (k) => { const i = argv.indexOf(k); return i < 0 ? null : argv[i + 1]; };
  const weights = opt("--weights") ? path.resolve(opt("--weights")) : null, port = Number(opt("--port") ?? 8788);
  if (!weights) console.log("no --weights: the pages load, but the models cannot (see examples/README.md)");
  else if (!argv.includes("--no-verify")) {
    console.log("checking the weights against the manifests (sha256, once)...");
    const v = await verifyWeights(weights);
    if (v.problems.length) { console.error("weights do not match the manifests:\n  " + v.problems.join("\n  ")); process.exit(1); }
    console.log(`${v.checked} files match`);
  }
  createMediaServer({ weights }).listen(port, "127.0.0.1", () => console.log(`http://127.0.0.1:${port}/audio-turn/`));
}
