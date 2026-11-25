// AUDIO_REF_DIR: directory with web/ and port/ (default: current directory). Usage: node browser-run.mjs <webgpu|wasm> [n] [out.json] [extra querystring]
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import puppeteer from "puppeteer-core";
const ROOT = process.env.AUDIO_REF_DIR ?? process.cwd(), [ ep = "wasm", n = "56", out = `result_${ep}.json`, extra = ""] = process.argv.slice(2);
const mime = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream" };
const srv = http.createServer((req, res) => { const p = path.join(ROOT, decodeURIComponent(req.url.split("?")[0]).replace(/^\/w\//, "/web/")); if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "Content-Type": mime[path.extname(p)] || "application/octet-stream", "Content-Length": fs.statSync(p).size, "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" }); fs.createReadStream(p).pipe(res); });
await new Promise(r => srv.listen(0, "127.0.0.1", r)); const port = srv.address().port;
const browser = await puppeteer.launch({ executablePath: process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: "new", protocolTimeout: 3600000,
  args: ["--enable-unsafe-webgpu", "--enable-features=SharedArrayBuffer", "--ignore-gpu-blocklist", "--no-first-run", "--js-flags=--max-old-space-size=8192"] });
const page = await browser.newPage(); page.on("console", m => console.log("[page]", m.text())); page.on("pageerror", e => console.log("[pageerror]", e.message));
console.log("Chrome", await browser.version());
await page.goto(`http://127.0.0.1:${port}/web/browser-index.html?ep=${ep}&n=${n}${extra}`);
await page.waitForFunction("window.__result", { timeout: 3600000, polling: 2000 });
const r = await page.evaluate("window.__result"); fs.writeFileSync(path.join(ROOT, out), JSON.stringify(r, null, 1)); console.log(JSON.stringify({ ...r, rowsA: undefined, rowsB: undefined }, null, 1));
await browser.close(); srv.close();
