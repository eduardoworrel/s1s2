#!/usr/bin/env node
// Chrome check of the decoder questions (language, term) through the audio demo page: parity with the torch references and with the Node CPU
// run, ms per question, memory. NOT part of `npm test`: it loads gigabytes. Run it under the Mac's heavy-job lock:
//   node test/browser/media-chrome-decoder.mjs <weights dir> <reference dir> [--device webgpu|wasm] [--limit N] [--node-results file.json] [--cost N] [--out dir]
// <reference dir>: dec_ref.jsonl and af/ written by recipes/marimba-audio/decoder_refs.py. The server lives and dies inside this process.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const { createMediaServer } = await import(path.join(ROOT, "examples/serve-media.mjs"));
const { compare } = await import(path.join(ROOT, "recipes/marimba-audio/decoder-metrics.mjs"));
const require = createRequire(process.env.PLAYWRIGHT_DIR ? process.env.PLAYWRIGHT_DIR + "/" : process.cwd() + "/");
const { chromium } = require("playwright-core");
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const argv = process.argv.slice(2), opt = (k, d = null) => { const i = argv.indexOf(k); return i < 0 ? d : argv[i + 1]; };
const [weights, refDir] = argv;
if (!weights || !refDir) { console.error("usage: media-chrome-decoder.mjs <weights dir> <reference dir> [--device webgpu|wasm] [--limit N] [--node-results f] [--out dir]"); process.exit(2); }
const device = opt("--device", "webgpu"), limit = Number(opt("--limit", 1e9)), OUT = path.resolve(opt("--out", path.join(ROOT, ".capturas-multimodal")));
fs.mkdirSync(OUT, { recursive: true });
const rss = (re = "Google Chrome") => { try { return Number(execSync(`ps -axo rss,command | grep -i '${re}' | grep -v grep | awk '{s+=$1} END{print s/1024}'`).toString().trim().replace(",", ".")); } catch { return NaN; } };

const srv = createMediaServer({ weights: path.resolve(weights), extra: { "/lab/dec/": path.resolve(refDir), "/lab/metrics/": path.join(ROOT, "recipes/marimba-audio") } });
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${srv.address().port}`;
const browser = await chromium.launch({ executablePath: CHROME, headless: !process.env.HEADFUL, args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan,WebGPU,SharedArrayBuffer", "--no-first-run"] });
const result = { device, chrome: browser.version(), when: new Date().toISOString(), errors: [], rss: {} };
console.log("Chrome", result.chrome, "at", base, "device", device);
const refRows = fs.readFileSync(path.join(refDir, "dec_ref.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).slice(0, limit);
try {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await ctx.newPage();
  page.on("console", (m) => { if (m.type() === "error") result.errors.push("console: " + m.text().slice(0, 300)); });
  page.on("pageerror", (e) => result.errors.push("pageerror: " + String(e.message ?? e).slice(0, 300)));
  page.on("request", (r) => { const u = r.url(); if (!u.startsWith(base) && !/^(blob|data):/.test(u)) result.errors.push("external request: " + u); });
  await page.goto(`${base}/audio-turno/index.html?lang=en&device=${device}`);
  await page.waitForFunction(() => window.s1s2demo, null, { timeout: 30000 });
  result.rss.beforeLoadMB = rss();
  const t0 = Date.now();
  await page.click("#load");
  await page.waitForFunction(() => /Model ready|Error|error/.test(document.querySelector("#status")?.textContent ?? ""), null, { timeout: 30 * 60 * 1000, polling: 1000 });
  const status = await page.textContent("#status");
  if (!/Model ready/.test(status)) throw new Error("load failed: " + status);
  result.loadWallMs = Date.now() - t0; result.rss.afterLoadMB = rss(); result.rss.gpuProcessAfterLoadMB = rss("Google Chrome.*type=gpu-process");
  result.tasks = await page.evaluate(() => window.s1s2demo.model.tasks);
  // decoder parity through the library in the page, from torch's audio_features (windows in dec_ref.jsonl)
  const rows = await page.evaluate(async (ref) => {
    const { rowFromAnswers } = await import("/lab/metrics/decoder-metrics.mjs");
    const m = window.s1s2demo.model, out = [];
    const f32 = async (u) => new Float32Array(await (await fetch(u)).arrayBuffer());
    {  // warm-up (shader compilation, first runs)
      const r = ref[0]; await m.decodeFeatures(await f32(`/lab/dec/af/${r.id}.f32`), [{ task: "language" }, { task: "term", term: Object.keys(r.terms)[0] }]);
    }
    for (const r of ref) {
      const af = await f32(`/lab/dec/af/${r.id}.f32`), terms = Object.keys(r.terms), timing = {}, t0 = performance.now();
      const ans = await m.decodeFeatures(af, [{ task: "language" }, ...terms.map((t) => ({ task: "term", term: t }))], timing);
      out.push(rowFromAnswers(r.id, ans, terms, timing, performance.now() - t0));
    }
    return out;
  }, refRows);
  result.vsTorch = compare(refRows, rows, { mode: "features", baseline: "torch" });
  console.log(JSON.stringify(result.vsTorch.summary, null, 1));
  const nr = opt("--node-results");
  if (nr) {
    const node = JSON.parse(fs.readFileSync(nr, "utf8")).results, byId = new Map(node.map((r) => [r.id, r]));
    // Chrome against Node CPU: the Node rows play the part of the reference (language scores are log-probabilities there too)
    const asRef = rows.map((r) => byId.get(r.id)).filter(Boolean).map((n) => ({ id: n.id, language: n.scores, hyps: n.hyps, terms: n.terms }));
    result.vsNode = compare(asRef, rows, { mode: "features", baseline: "node-cpu" });
    console.log(JSON.stringify(result.vsNode.summary, null, 1));
  }
  // cost of a window of audio: mel + tower (+ the left-aligned tower pass of a short clip) + language + term, split into the parts, N joined 8 s windows and N short clips
  const costN = Number(opt("--cost", 0));
  if (costN) result.cost = await page.evaluate(async ({ ref, N }) => {
    const m = window.s1s2demo.model, dec = m.parts.decoder, f32 = async (u) => new Float32Array(await (await fetch(u)).arrayBuffer()), med = (x) => [...x].sort((a, b) => a - b)[x.length >> 1];
    const pick = (re) => ref.filter((r) => re.test(r.id)).slice(0, N), out = {};
    for (const [name, rows] of [["full8s", pick(/^j/)], ["short", pick(/^c/)]]) {
      const T = { mel: [], tower: [], leftTower: [], open: [], language: [], greedy: [], beam: [], term: [], greedyTokens: [], total: [] };
      for (const r of rows) {
        const win = await f32(`/lab/dec/af/${r.id}.win.f32`), t0 = performance.now();
        const enc = await m.encode({ samples: win.subarray(0, r.n).slice(), sampleRate: 16000 });
        const tl = {}; await m.read(enc, ["language"], tl);
        const tt = {}; await m.read(enc, [{ task: "term", term: "palavra" }], tt);
        T.total.push(performance.now() - t0); T.mel.push(enc.ms.mel); T.tower.push(enc.ms.tower); T.leftTower.push(tl.leftTower ?? 0);
        T.open.push(tl.decoderOpen); T.language.push(tl.language); T.term.push(tt.term);
        const af = r.n >= 128000 ? enc.taps.audio_features : (await enc.left).audio_features, w = await dec.open(af);
        let a = performance.now(); const g = await w.greedy(); T.greedy.push(performance.now() - a); T.greedyTokens.push(g.length);
        a = performance.now(); await w.beam(); T.beam.push(performance.now() - a); await w.close();
      }
      out[name] = Object.fromEntries(Object.entries(T).map(([k, v]) => [k, med(v)])); out[name].n = rows.length;
    }
    return out;
  }, { ref: refRows, N: costN });
  if (result.cost) console.log(JSON.stringify(result.cost));
  // one full 8 s window through the page: mel + tower + (left tower) + language + term
  await page.click('button.ex[data-src="clips/clip-1.wav,clips/clip-2.wav"]');
  await page.waitForFunction(() => /ms/.test(document.querySelector("#cost")?.textContent ?? ""), null, { timeout: 5 * 60 * 1000 });
  result.ui = await page.evaluate(() => ({ cost: document.querySelector("#cost").textContent, cards: [...document.querySelectorAll(".card")].map((c) => (c.dataset.task ?? "") + ": " + c.querySelector(".big")?.textContent + " | " + c.querySelector(".muted")?.textContent) }));
  // the word box: a word that is not there, then a word of the first transcript
  const askUi = async (w) => { await page.fill("#term", w); const t = Date.now(); await page.click("#ask"); await page.waitForFunction((x) => window.s1s2demo.term && window.s1s2demo.term.__w === x, w, { timeout: 5 * 60 * 1000 }).catch(() => {}); return Date.now() - t; };
  await page.evaluate(() => { const a = window.s1s2demo.ask; window.s1s2demo.ask = async () => { const w = document.querySelector("#term").value.trim(), r = await a(); if (r) r.__w = w; return r; }; document.querySelector("#ask").onclick = () => window.s1s2demo.ask(); });
  result.ui.askMissing = { ms: await askUi("zzzxq"), card: await page.evaluate(() => document.querySelector('[data-task="term"] .big').textContent + " | " + document.querySelector('[data-task="term"] .muted').textContent + " | " + document.querySelector("#hyp").textContent) };
  const hyp = await page.evaluate(() => window.s1s2demo.term.hypotheses[0]);
  const word = (hyp.normalize("NFKD").replace(/[^\p{L}]+/gu, " ").split(" ").filter((x) => x.length >= 4)[0]) ?? "";
  result.ui.askPresent = { word, ms: await askUi(word), card: await page.evaluate(() => document.querySelector('[data-task="term"] .big').textContent + " | " + document.querySelector('[data-task="term"] .muted').textContent + " | " + document.querySelector("#hyp").textContent) };
  await page.setViewportSize({ width: 390, height: 844 }); await page.waitForTimeout(500);
  result.phone = await page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth }));
  await page.screenshot({ path: path.join(OUT, `decoder-${device}-390.png`), fullPage: true });
  await page.setViewportSize({ width: 1280, height: 900 });
  result.rss.peakAfterMB = rss(); result.rss.gpuProcessMB = rss("Google Chrome.*type=gpu-process");
  await page.screenshot({ path: path.join(OUT, `decoder-${device}-1280.png`), fullPage: true });
  await page.close();
  result.rows = rows;
} catch (e) { result.fatal = String(e?.stack ?? e); console.error(result.fatal); }
finally {
  const f = path.join(OUT, `decoder-chrome-${device}.json`);
  fs.writeFileSync(f, JSON.stringify(result, null, 1) + "\n");
  console.log("wrote", f, "errors:", result.errors.length);
  await browser.close(); srv.close();
}
