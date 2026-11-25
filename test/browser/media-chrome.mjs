#!/usr/bin/env node
// Chrome / WebGPU check of the audio demo: parity with the torch references, ms per decision, memory, captures (desktop 1280,
// phone 390), no horizontal scroll, no console error. NOT part of `npm test`: it loads gigabytes. Run it under the Mac's heavy-job lock:
//   node test/browser/media-chrome.mjs audio <weights dir> <reference dir> [--out dir]
// <weights dir>: holds marimba-audio/ (examples/README.md). <reference dir>: the reference directory (audio: audio/port/ref).
// The server lives and dies inside this process; nothing is sent anywhere.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { execSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const { createMediaServer } = await import(pathToFileURL(path.join(ROOT, "examples/serve-media.mjs")).href);
const require = createRequire(process.env.PLAYWRIGHT_DIR ? process.env.PLAYWRIGHT_DIR + "/" : process.cwd() + "/")   // a directory where playwright-core is installed;
const { chromium } = require("playwright-core");
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const [what, weights, refDir] = process.argv.slice(2);
const oi = process.argv.indexOf("--out");
const OUT = oi > 0 ? path.resolve(process.argv[oi + 1]) : path.join(ROOT, ".captures-media");
if (!["audio"].includes(what) || !weights || !refDir) { console.error("usage: media-chrome.mjs audio <weights dir> <reference dir> [--out dir]"); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });

const rss = (re = "Google Chrome") => { try { return Number(execSync(`ps -axo rss,command | grep -i '${re}' | grep -v grep | awk '{s+=$1} END{print s/1024}'`).toString().trim().replace(",", ".")); } catch { return NaN; } };   // awk prints a decimal comma in pt_BR
const gpuRss = () => rss("Google Chrome.*type=gpu-process");

const srv = createMediaServer({ weights: path.resolve(weights), extra: { "/reference/audio/": path.join(refDir, "audio/port/ref") } });
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${srv.address().port}`;

const browser = await chromium.launch({ executablePath: CHROME, headless: !process.env.HEADFUL, args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan,WebGPU,SharedArrayBuffer", "--no-first-run", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] });   // fake microphone: the Record button runs without a device
const result = { what, chrome: browser.version(), when: new Date().toISOString(), errors: [], rss: {} };
console.log("Chrome", result.chrome, "at", base);

async function newPage(viewport) {
  const ctx = await browser.newContext({ viewport, permissions: ["microphone"] });    // the fake device still needs the permission in headless
  const page = await ctx.newPage();
  page.on("console", (m) => { if (m.type() === "error") result.errors.push("console: " + m.text().slice(0, 300)); });
  page.on("pageerror", (e) => result.errors.push("pageerror: " + String(e.message ?? e).slice(0, 300)));
  page.on("response", (r) => { if (r.status() >= 400) result.errors.push(`http ${r.status()} ${r.url()}`); });
  page.on("request", (r) => { const u = r.url(); if (!u.startsWith(base) && !/^(blob|data):/.test(u)) result.errors.push("external request: " + u); });   // nothing may leave the page
  page.on("requestfailed", (r) => result.errors.push(`requestfailed ${r.url()} ${r.failure()?.errorText}`));
  return page;
}
const noHScroll = (page) => page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth, bodyScrollWidth: document.body.scrollWidth }));
async function shoot(page, name) {
  const o = await noHScroll(page);
  await page.screenshot({ path: path.join(OUT, name), fullPage: true });
  return { file: name, ...o, ok: o.scrollWidth <= o.innerWidth };
}
const waitReady = (page, re, ms = 20 * 60 * 1000) => page.waitForFunction((r) => new RegExp(r, "i").test(document.querySelector("#status")?.textContent ?? ""), re, { timeout: ms, polling: 1000 });

try {
  if (what === "audio") {
    const page = await newPage({ width: 1280, height: 900 });
    await page.goto(`${base}/audio-turn/index.html?lang=en`);
    await page.waitForFunction(() => window.s1s2demo, null, { timeout: 30000 });
    result.rss.beforeLoadMB = rss();
    const t0 = Date.now();
    await page.click("#load");
    await waitReady(page, "Model ready|Loading.*(Error|error)|Error");
    const status = await page.textContent("#status");
    if (!/Model ready/.test(status)) throw new Error("load failed: " + status);
    result.loadWallMs = Date.now() - t0; result.device = await page.evaluate(() => window.s1s2demo.model.device);
    result.rss.afterLoadMB = rss(); result.rss.gpuProcessAfterLoadMB = gpuRss();
    // parity through the library, in the page: 56 windows set A (right-aligned, n real samples) and set B (3 clips)
    result.parity = await page.evaluate(async () => {
      const f32 = async (u) => new Float32Array(await (await fetch(u)).arrayBuffer());
      const ref = await (await fetch("/reference/audio/ref.json")).json(), winA = await f32("/reference/audio/winA.f32"), winB = await f32("/reference/audio/winB.f32"), W = 128000;
      const m = window.s1s2demo.model, md = {}, ag = {}, T = { mel: [], tower: [], readers: [], ms: [], two: [] };
      const up = (k, d) => { md[k] = Math.max(md[k] ?? 0, d); }, ac = (k, ok) => { ag[k] = (ag[k] ?? 0) + (ok ? 1 : 0); };
      await m.decide({ audio: { samples: winA.subarray(W - ref.A[0].n, W), sampleRate: 16000 }, questions: ["voice_activity"] });   // warm-up (shader compilation)
      for (let i = 0; i < 56; i++) {
        const a = ref.A[i], b = ref.B[i];
        const dA = await m.decide({ audio: { samples: winA.subarray(i * W + W - a.n, (i + 1) * W), sampleRate: 16000 }, questions: ["voice_activity", "turn_end"] });
        const dB = await m.decide({ audio: { samples: winB.subarray(i * W, (i + 1) * W), sampleRate: 16000 }, questions: ["voice_activity", "turn_end", "same_speaker", "speaker_change", "speaker_count"] });
        const g = (d, t) => d.answers.find((x) => x.task === t);
        up("voice_activity", Math.abs(g(dA, "voice_activity").probability - a.vad_p)); up("turn_end", Math.abs(g(dA, "turn_end").probability - a.turn_p));
        up("same_speaker", Math.abs(g(dB, "same_speaker").probability - b.same_p)); up("speaker_change", Math.abs(g(dB, "speaker_change").probability - b.change_p));
        const cnt = g(dB, "speaker_count").probabilities.map((x) => x.probability); up("speaker_count", Math.max(...cnt.map((v, k) => Math.abs(v - b.count_p[k]))));
        ac("voice_activity@0.225", (g(dA, "voice_activity").decision === "yes") === (a.vad_p >= 0.225)); ac("turn_end@0.5", (g(dA, "turn_end").decision === "yes") === (a.turn_p >= 0.5));
        ac("same_speaker@0.5", (g(dB, "same_speaker").decision === "yes") === (b.same_p >= 0.5)); ac("speaker_change@0.5", (g(dB, "speaker_change").decision === "yes") === (b.change_p >= 0.5));
        ac("speaker_count argmax", cnt.indexOf(Math.max(...cnt)) === b.count_p.indexOf(Math.max(...b.count_p)));
        T.mel.push(dB.cost.parts.mel); T.tower.push(dB.cost.parts.tower); T.readers.push(dB.cost.parts.readers); T.ms.push(dB.cost.ms); T.two.push(dA.cost.ms);
      }
      const med = (x) => [...x].sort((p, q) => p - q)[x.length >> 1];
      return { n: 56, maxAbsDiffP: md, agreement: ag, medianMs: { mel: med(T.mel), tower: med(T.tower), readers_5_tasks: med(T.readers), total_5_tasks: med(T.ms), total_2_tasks: med(T.two) } };
    });
    console.log(JSON.stringify(result.parity));
    // the page, through its own buttons
    await page.click('button.ex[data-src="clips/clip-1.wav,clips/clip-2.wav"]');
    await page.waitForFunction(() => /ms/.test(document.querySelector("#cost")?.textContent ?? ""), null, { timeout: 60000 });
    result.ui = { two: await page.evaluate(() => ({ cost: document.querySelector("#cost").textContent, cards: [...document.querySelectorAll(".card")].map((c) => c.dataset.task + ": " + c.querySelector(".big").textContent + " | " + c.querySelector(".muted").textContent) })) };
    result.captures = [await shoot(page, "audio-1280-8s.png")];
    await page.click('button.ex[data-src="clips/clip-1.wav"]');
    await page.waitForFunction(() => /clip-1/.test(document.querySelector("#what")?.textContent ?? ""), null, { timeout: 60000 }); await page.waitForTimeout(800);
    result.ui.clip1 = await page.evaluate(() => [...document.querySelectorAll(".card")].map((c) => c.dataset.task + ": " + c.querySelector(".big").textContent + " | " + c.querySelector(".muted").textContent));
    result.captures.push(await shoot(page, "audio-1280-clip1.png"));
    await page.click('button.ex[data-src="clips/noise.wav"]'); await page.waitForTimeout(1500);
    result.ui.noise = await page.evaluate(() => [...document.querySelectorAll(".card")].slice(0, 2).map((c) => c.dataset.task + ": " + c.querySelector(".big").textContent + " | " + c.querySelector(".muted").textContent));
    await page.click("#rec");                                                       // microphone path (Chrome's fake device: a beep train), 8 s
    await page.waitForFunction(() => /^mic:|not available|indispon/i.test((document.querySelector("#what")?.textContent ?? "") + (document.querySelector("#status")?.textContent ?? "")), null, { timeout: 60000 }).catch(() => {}); await page.waitForTimeout(1500);
    result.ui.micStatus = await page.textContent("#status");
    result.ui.mic = await page.evaluate(() => ({ what: document.querySelector("#what").textContent, cards: [...document.querySelectorAll(".card")].map((c) => c.dataset.task + ": " + c.querySelector(".big").textContent + " | " + c.querySelector(".muted").textContent) }));
    await page.setViewportSize({ width: 390, height: 844 });
    await page.click('button.ex[data-src="clips/clip-1.wav,clips/clip-2.wav"]'); await page.waitForTimeout(1500);
    result.captures.push(await shoot(page, "audio-390-8s.png"));
    result.rss.peakAfterMB = rss(); result.rss.gpuProcessMB = gpuRss();
    await page.close();
  }
} catch (e) { result.fatal = String(e?.stack ?? e); console.error(result.fatal); }
finally {
  const f = path.join(OUT, `${what}-chrome.json`);
  fs.writeFileSync(f, JSON.stringify(result, null, 1) + "\n");
  console.log("wrote", f, "errors:", result.errors.length);
  await browser.close(); srv.close();
}
