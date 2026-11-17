// GatedDeltaNet (S5-nano, GenAI 0.19 builder) on onnxruntime-web WebGPU in Chrome (Playwright), on the tiny graph from
// test/tools/make_gdn_graph.py: inputs 1-3 empty exactly as the builder emits them. Compares the outputs with the
// numpy reference in gdn_ref.json. Exits 0 when the session is created and the outputs match.
//
//   node test/smoke/gdn_webgpu.mjs [--ort-dist <onnxruntime-web/dist>] [--fixture test/fixtures/gdn] [--playwright <dir>]
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const ortDist = path.resolve(arg("ort-dist", path.join(HERE, "../../node_modules/onnxruntime-web/dist")));
const fixture = path.resolve(arg("fixture", path.join(HERE, "../fixtures/gdn")));
const PW = arg("playwright", process.env.PLAYWRIGHT_DIR ?? path.join(process.cwd(), "node_modules/playwright"));
const ortVersion = JSON.parse(fs.readFileSync(path.join(ortDist, "../package.json"), "utf8")).version;

const page = `<!doctype html><meta charset="utf-8"><script type="module">
import * as ort from "/ort/ort.webgpu.min.mjs";
ort.env.wasm.numThreads = 1; ort.env.wasm.wasmPaths = "/ort/";
const f16 = (a) => { const b = new Uint16Array(a.length), f = new Float32Array(1), u = new Uint32Array(f.buffer);
  for (let i = 0; i < a.length; i++) { f[0] = a[i]; const x = u[0], s = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112, m = x & 0x7fffff;
    b[i] = e <= 0 ? s : e >= 31 ? s | 0x7c00 : s | (e << 10) | ((m + 0x1000) >> 13); } return b; };
const h2f = (h) => { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff; return e === 0 ? s * 2 ** -14 * (f / 1024) : s * 2 ** (e - 15) * (1 + f / 1024); };
window.run = async () => {
  const ref = await (await fetch("/fx/gdn_ref.json")).json(), d = ref.dims;
  const out = { ort: ort.env.versions?.web, adapter: !!(await navigator.gpu?.requestAdapter()) };
  try {
    const S = await ort.InferenceSession.create("/fx/gdn.onnx", { executionProviders: ["webgpu"], logSeverityLevel: 3 });
    const T = (t, a, dims) => new ort.Tensor(t, a, dims);
    const r = await S.run({ qkv: T("float16", f16(ref.inputs.qkv), [1, d.seq, d.packed]), decay: T("float16", f16(ref.inputs.decay), [1, d.seq, d.hv]),
      beta: T("float16", f16(ref.inputs.beta), [1, d.seq, d.hv]), "past.0.recurrent": T("float32", Float32Array.from(ref.inputs["past.0.recurrent"]), [1, d.hv, d.dv, d.dk]) });
    const o = Array.from(r.out.data instanceof Uint16Array ? r.out.data : new Uint16Array(r.out.data.buffer), h2f), st = Array.from(r["present.0.recurrent"].data);
    const mx = (x, y) => x.reduce((m, v, i) => Math.max(m, Math.abs(v - y[i])), 0);
    Object.assign(out, { session: "ok", maxAbsErrOut: mx(o, ref.expected.out), maxAbsErrState: mx(st, ref.expected["present.0.recurrent"]) });
  } catch (e) { Object.assign(out, { session: "error", error: String(e?.message ?? e).slice(-300) }); }
  return out;
};
window.ready = true;
</script>`;
const types = { ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const srv = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const hdr = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
  if (u === "/") { res.writeHead(200, { ...hdr, "Content-Type": "text/html" }); return res.end(page); }
  const [root, rel] = u.startsWith("/ort/") ? [ortDist, u.slice(5)] : u.startsWith("/fx/") ? [fixture, u.slice(4)] : [null, null];
  const p = root && path.resolve(root, rel);
  if (!p || !p.startsWith(root) || !fs.existsSync(p)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { ...hdr, "Content-Type": types[path.extname(p)] ?? "application/octet-stream" });
  fs.createReadStream(p).pipe(res);
});
await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
const { chromium } = await import(pathToFileURL(path.join(PW, "index.mjs")).href);
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--ignore-gpu-blocklist"] });
const pg = await browser.newPage();
await pg.goto(`http://127.0.0.1:${srv.address().port}/`);
await pg.waitForFunction(() => window.ready === true, null, { timeout: 60000 });
const r = await pg.evaluate(() => window.run());
await browser.close(); srv.close();
r.package = ortVersion; r.chrome = browser.version();
console.log(JSON.stringify(r));
const ok = r.session === "ok" && r.maxAbsErrOut < 2e-3 && r.maxAbsErrState < 1e-4;
console.log(ok ? "SMOKE OK" : "SMOKE FAIL");
process.exit(ok ? 0 : 1);
