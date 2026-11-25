#!/usr/bin/env node
// Chrome, s1s2 + transformers.js, one packed folder served locally as if it were the Hub (no upload needed):
//   PLAYWRIGHT_DIR=<.../node_modules/playwright> node wasm_bench.mjs --mode wasm|webgpu --dir <pack folder> --items items.jsonl --ref ref.json --out out.json
// wasm: WebGPU hidden (navigator.gpu undefined), cross-origin isolated (ORT default threads), ORT from transformers.js's
// own onnxruntime-web. Numbers behind the cards' browser CPU line (tjs_card_checks.json, wasm). Memory: peak RSS of Chrome's processes (sampled every 0.5 s) and measureUserAgentSpecificMemory().
import fs from "node:fs"; import http from "node:http"; import path from "node:path"; import { execFile, execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
const ROOT = process.env.S1S2_DIR ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const MODE = arg("--mode", "wasm"), DIR = path.resolve(arg("--dir"));
const nm = (p) => path.join(ROOT, "node_modules", p);
const ORT = nm("@huggingface/transformers/node_modules/onnxruntime-web/dist");
const routes = { "/lib/": path.join(ROOT, "dist"), "/vendor/ort/": ORT, "/vendor/ortc/": nm("@huggingface/transformers/node_modules/onnxruntime-common/dist/esm"),
  "/vendor/tjs/": nm("@huggingface/transformers/dist") };
if (!fs.existsSync(routes["/vendor/ortc/"])) routes["/vendor/ortc/"] = nm("onnxruntime-common/dist/esm");
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const page = `<!doctype html><html><head><meta charset="utf-8"><script type="importmap">
{ "imports": { "onnxruntime-web/wasm": "/vendor/ort/ort.wasm.min.mjs", "onnxruntime-web/webgpu": "/vendor/ort/ort.webgpu.min.mjs",
  "onnxruntime-web": "/vendor/ort/ort.min.mjs", "onnxruntime-common": "/vendor/ortc/index.js", "@huggingface/transformers": "/vendor/tjs/transformers.web.js" } }
</script></head><body><script type="module">
import { S1S2 } from "/lib/index.js";
import { env } from "@huggingface/transformers";
env.remoteHost = location.origin + "/hub/"; env.useBrowserCache = false;
const mem = async () => { try { const r = await performance.measureUserAgentSpecificMemory(); return r.bytes; } catch (e) { return null; } };
window.run = async (manifest, items, device) => {
  const repo = Object.values(manifest.builds)[0].repo, rev = Object.values(manifest.builds)[0].revision;
  const t0 = performance.now();
  const lib = await S1S2.load({ manifest, device, baseUrls: { [repo]: location.origin + "/hub/" + repo + "/resolve/" + rev + "/" }, allowUnverified: true });
  const msLoad = performance.now() - t0;
  const memLoad = await mem();
  const rows = [], ms = [];
  for (const it of items) {
    const t = performance.now();
    const s = await lib.decide({ input: it.input, questions: [{ ...it.question, method: "head" }] });
    ms.push(performance.now() - t);
    const a = s.answers[0];
    rows.push({ id: it.id, choice: a.choice, probabilities: a.probabilities.map((p) => p.probability) });
    await s.dispose();
  }
  const memRun = await mem();
  const out = { device: lib.device, runtime: lib.runtime, build: lib.build, msLoad: Math.round(msLoad), msPerDecision: ms.map(Math.round), memLoad, memRun, rows,
    wasm: performance.getEntriesByType("resource").map((e) => e.name).filter((n) => n.endsWith(".wasm")).map((n) => n.split("/").pop()),
    env: { gpu: typeof navigator.gpu, crossOriginIsolated: self.crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency,
      adapter: navigator.gpu ? await navigator.gpu.requestAdapter().then((a) => a && (a.info ? a.info.vendor + " " + a.info.architecture + " " + a.info.description : "?")) : null } };
  await lib.dispose();
  return out;
};
window.ready = true;
</script></body></html>`;
const hdr = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "cross-origin", "Access-Control-Allow-Origin": "*" };
const srv = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (url === "/") { res.writeHead(200, { ...hdr, "Content-Type": "text/html" }); return res.end(page); }
  let file;
  const hub = url.match(/^\/hub\/[^/]+\/[^/]+\/resolve\/[^/]+\/(.+)$/);
  if (hub) file = path.resolve(DIR, hub[1]);
  else { const pre = Object.keys(routes).find((p) => url.startsWith(p)); file = pre && path.resolve(routes[pre], url.slice(pre.length)); }
  const st = file && fs.statSync(file, { throwIfNoEntry: false });
  if (!st?.isFile()) { res.writeHead(404, hdr); return res.end(); }
  res.writeHead(200, { ...hdr, "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream", "Content-Length": st.size });
  if (req.method === "HEAD") return res.end();
  fs.createReadStream(file).pipe(res);
});
await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
// Peak RSS of this process's Chrome descendants. Rows: pid, ppid, rss in KiB, name.
const WIN = process.platform === "win32";
const peak = { renderer: 0, total: 0 };
const tally = (text) => {
  const ps = text.trim().split(/\r?\n/).map((l) => l.trim().split(/\s+/));
  const kids = new Set([process.pid]); let grew = true;
  while (grew) { grew = false; for (const [p, pp] of ps) if (kids.has(+pp) && !kids.has(+p)) { kids.add(+p); grew = true; } }
  let tot = 0, ren = 0;
  for (const [p, , rss, ...c] of ps) if (kids.has(+p) && +p !== process.pid) { tot += +rss; if (/Renderer/.test(c.join(" "))) ren = Math.max(ren, +rss); }
  peak.total = Math.max(peak.total, tot * 1024); peak.renderer = Math.max(peak.renderer, ren * 1024);
};
// Windows has no ps: the same rows come from CIM (working set). That call takes about a second, so it runs in the
// background and less often, and never blocks the local server.
const WIN_PS = "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | ForEach-Object { '{0} {1} {2} {3}' -f $_.ProcessId,$_.ParentProcessId,[int64]($_.WorkingSetSize/1024),$(if ($_.CommandLine -match '--type=renderer') { 'Renderer' } else { 'chrome' }) }";
let busy = false;
const sample = () => {
  try {
    if (!WIN) return tally(execFileSync("ps", ["-axo", "pid=,ppid=,rss=,comm="], { encoding: "utf8" }));
    if (busy) return; busy = true;
    execFile("powershell", ["-NoProfile", "-NonInteractive", "-Command", WIN_PS], { encoding: "utf8", windowsHide: true }, (e, out) => { busy = false; if (!e) tally(out); });
  } catch { /* ignore */ }
};
const timer = setInterval(sample, WIN ? 2000 : 500);
const PW = process.env.PLAYWRIGHT_DIR;
const pw = PW ? await import(pathToFileURL(path.join(PW, "index.mjs")).href) : await import("playwright");
const args = MODE === "wasm" ? ["--disable-features=WebGPU", "--disable-gpu"] : ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--ignore-gpu-blocklist"];
const browser = await pw.chromium.launch({ channel: "chrome", headless: MODE === "wasm", args });
try {
  const all = fs.readFileSync(arg("--items"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
  const ref = JSON.parse(fs.readFileSync(arg("--ref"), "utf8")).rows;
  const res = { mode: MODE, chrome: browser.version(), dir: DIR, groups: [], rows: [] };
  for (const mname of [...new Set(all.map((r) => r.manifest))]) {
    const ctx = await browser.newContext();
    if (MODE === "wasm") await ctx.addInitScript(() => { Object.defineProperty(Navigator.prototype, "gpu", { get: () => undefined, configurable: true }); });
    const p = await ctx.newPage();
    p.on("pageerror", (e) => console.error("pageerror:", e.message));
    p.on("console", (m) => { const t = m.text(); if (/error|fail|alloc|memory/i.test(t)) console.error("[console]", t.slice(0, 300)); });
    await p.goto(`http://127.0.0.1:${srv.address().port}/`);
    await p.waitForFunction(() => window.ready === true, null, { timeout: 60000 });
    const manifest = JSON.parse(fs.readFileSync(path.join(DIR, mname), "utf8"));
    const b = Object.values(manifest.builds)[0]; b.devices = [MODE]; // test-only: force this device
    const items = all.filter((r) => r.manifest === mname);
    const r = await p.evaluate(([m, its, d]) => window.run(m, its, d), [manifest, items, MODE]);
    sample();
    res.groups.push({ manifest: mname, device: r.device, runtime: r.runtime, wasm: r.wasm, env: r.env, msLoad: r.msLoad, msPerDecision: r.msPerDecision, memLoad: r.memLoad, memRun: r.memRun, n: r.rows.length });
    res.rows.push(...r.rows);
    console.error(JSON.stringify(res.groups.at(-1)));
    await ctx.close();
  }
  let same = 0, dp = 0;
  for (const x of res.rows) { const g = ref.find((y) => y.id === x.id); same += x.choice === g.head; dp = Math.max(dp, ...x.probabilities.map((v, i) => Math.abs(v - g.headProbs[i]))); }
  const ms = res.groups.flatMap((g) => g.msPerDecision);
  Object.assign(res, { items: res.rows.length, sameAsNodeCpu: same, probMaxAbsDiff: dp, allOnDevice: res.groups.every((g) => g.device === MODE),
    secondsPerPrompt: ms.reduce((a, v) => a + v, 0) / 1000 / ms.length, secondsMedian: [...ms].sort((a, b) => a - b)[Math.floor(ms.length / 2)] / 1000,
    peakRssRenderer: peak.renderer, peakRssTotal: peak.total, memMax: Math.max(...res.groups.map((g) => Math.max(g.memLoad ?? 0, g.memRun ?? 0))) });
  fs.writeFileSync(arg("--out"), JSON.stringify(res, null, 1));
  console.log(JSON.stringify({ ...res, rows: undefined, groups: res.groups.map((g) => ({ ...g, msPerDecision: undefined })) }));
} finally { clearInterval(timer); await browser.close(); srv.close(); }
