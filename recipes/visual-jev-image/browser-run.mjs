// node browser-run.mjs BUILD_DIR MANIFEST Q_JSONL IMG_DIR OUT.json [webgpu|wasm] [n]   (no puppeteer: the page posts its result to this server; Chrome is only launched)
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { spawn } from "node:child_process";
const [build, manifest, qfile, imgdir, out, ep = "webgpu", n = "3"] = process.argv.slice(2), LIB = new URL("../../", import.meta.url).pathname;
const roots = { "/build/": build + "/", "/ort/": LIB + "node_modules/onnxruntime-web/dist/", "/tjs/": LIB + "node_modules/@huggingface/transformers/dist/", "/dist/": LIB + "dist/", "/img/": imgdir + "/" };
const mime = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".jpg": "image/jpeg" };
let resolveResult; const done = new Promise((r) => (resolveResult = r));
const hdr = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "same-origin" };
const srv = http.createServer((req, res) => {
  const u = decodeURIComponent(req.url.split("?")[0]);
  if (req.method === "POST") { let b = ""; req.on("data", (c) => (b += c)); req.on("end", () => { const o = JSON.parse(b); if (u === "/log") console.log("[page]", o.s); else resolveResult(o); res.writeHead(200, hdr); res.end("ok"); }); return; }
  let p = u === "/" ? new URL(process.env.PAGE ?? "./browser-page.html", import.meta.url).pathname : u === "/manifest.json" ? manifest : u === "/q.jsonl" ? qfile : null;
  for (const [k, v] of Object.entries(roots)) if (u.startsWith(k)) p = v + u.slice(k.length);
  if (!p || !fs.existsSync(p) || !fs.statSync(p).isFile()) { res.writeHead(404, hdr); res.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d*)-(\d*)/.exec(req.headers.range ?? "");
  const h = { "Content-Type": mime[path.extname(p)] ?? "application/octet-stream", "Accept-Ranges": "bytes", ...hdr };
  if (m) { const a = m[1] === "" ? size - +m[2] : +m[1], b = m[2] === "" || m[1] === "" ? size - 1 : Math.min(+m[2], size - 1); res.writeHead(206, { ...h, "Content-Range": `bytes ${a}-${b}/${size}`, "Content-Length": b - a + 1 }); fs.createReadStream(p, { start: a, end: b }).pipe(res); }
  else { res.writeHead(200, { ...h, "Content-Length": size }); fs.createReadStream(p).pipe(res); }
});
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const chrome = spawn(process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", ["--headless=new", "--enable-unsafe-webgpu", "--enable-features=SharedArrayBuffer", "--ignore-gpu-blocklist", "--no-first-run", "--user-data-dir=/tmp/vj-chrome-profile", `http://127.0.0.1:${srv.address().port}/?ep=${ep}&n=${n}&g=${process.env.G ?? ""}&opt=${process.env.OPT ?? "all"}&vd=${process.env.VD ?? ""}&loc=${process.env.LOC ?? "cpu"}&nc=${process.env.NC ?? "0"}&loc=${process.env.LOC ?? "cpu"}&nc=${process.env.NC ?? "0"}`], { stdio: "ignore" });
const r = await done; fs.writeFileSync(out, JSON.stringify(r, null, 1)); console.log(r.error ? "ERROR " + r.error : `rows ${r.rows?.length ?? "-"}`);
chrome.kill(); srv.close(); process.exit(0);
