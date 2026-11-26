#!/usr/bin/env node
// Node CPU parity of the s1s2 audio model against the torch Marimba references of the reference run that made the weights (56 FLEURS pt-BR clips).
// Heavy (loads the 746 MB tower, several GB of memory). Usage:
//   node recipes/marimba-audio/node-parity.mjs <reference dir> [--write-fixtures <dir>] [--out result.json]
// <reference dir> holds port/ref/{ref.json,winA.f32,winB.f32,melA.f32,melB.f32} and result.json (layout in README.md).
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const { loadMedia, createLogMel } = await import(pathToFileURL(path.join(ROOT, "dist/media.js")).href);
const args = process.argv.slice(2);
const refDir = path.resolve(args[0]);
const opt = (k) => { const i = args.indexOf(k); return i < 0 ? null : args[i + 1]; };
const fixDir = opt("--write-fixtures"), outFile = opt("--out");
const R = path.join(refDir, "port/ref");
const f32 = (n) => { const b = fs.readFileSync(path.join(R, n)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const ref = JSON.parse(fs.readFileSync(path.join(R, "ref.json")));
const refRows = JSON.parse(fs.readFileSync(path.join(refDir, "result.json")));
const winA = f32("winA.f32"), winB = f32("winB.f32"), melA = f32("melA.f32"), melB = f32("melB.f32");
const WIN = 128000, MEL = 128 * 800;

const t0 = performance.now();
const model = await loadMedia({ manifest: path.join(ROOT, "manifests/marimba-audio.json"), device: "cpu", localDirs: { "eduardoworrel/Marimba-ONNX-S1S2": path.join(ROOT, ".weights/marimba-audio") }, onStatus: (m) => console.log("[load]", m) });
const loadMs = performance.now() - t0;
console.log("loaded", Math.round(loadMs), "ms; tasks", model.tasks.join(", "));
const maxd = {}, agree = {}, timing = { mel: [], tower: [], readers: [], ms: [] };
const upd = (k, d) => { maxd[k] = Math.max(maxd[k] ?? 0, d); };
const ag = (k, ok) => { agree[k] = (agree[k] ?? 0) + (ok ? 1 : 0); };
const fixtures = [];
const mel = createLogMel(model.spec.preprocess);
let melMax = 0;
for (let i = 0; i < 56; i++) {
  const a = ref.A[i], b = ref.B[i];
  const wa = winA.subarray(i * WIN, (i + 1) * WIN), wb = winB.subarray(i * WIN, (i + 1) * WIN);
  const dA = await model.decide({ audio: { samples: wa.subarray(WIN - a.n), sampleRate: 16000 }, questions: ["voice_activity", "turn_end"] });
  const dB = await model.decide({ audio: { samples: wb, sampleRate: 16000 }, questions: ["voice_activity", "turn_end", "same_speaker", "speaker_change", "speaker_count"] });
  const g = (d, t) => d.answers.find((x) => x.task === t);
  upd("voice_activity_p(A)", Math.abs(g(dA, "voice_activity").probability - a.vad_p));
  upd("turn_end_p(A)", Math.abs(g(dA, "turn_end").probability - a.turn_p));
  upd("turn_end_uncalibrated(A)", Math.abs(g(dA, "turn_end").uncalibrated - a.turn_score));
  upd("voice_activity_p(B)", Math.abs(g(dB, "voice_activity").probability - b.vad_p));
  upd("turn_end_p(B)", Math.abs(g(dB, "turn_end").probability - b.turn_p));
  upd("same_speaker_p", Math.abs(g(dB, "same_speaker").probability - b.same_p));
  upd("speaker_change_p", Math.abs(g(dB, "speaker_change").probability - b.change_p));
  const cnt = g(dB, "speaker_count").probabilities.map((x) => x.probability);
  upd("speaker_count_p", Math.max(...cnt.map((v, k) => Math.abs(v - b.count_p[k]))));
  ag("voice_activity@0.225(A)", (g(dA, "voice_activity").decision === "yes") === (a.vad_p >= 0.225));
  ag("turn_end@0.5(A)", (g(dA, "turn_end").decision === "yes") === (a.turn_p >= 0.5));
  ag("same_speaker@0.5", (g(dB, "same_speaker").decision === "yes") === (b.same_p >= 0.5));
  ag("speaker_change@0.5", (g(dB, "speaker_change").decision === "yes") === (b.change_p >= 0.5));
  ag("speaker_count argmax", cnt.indexOf(Math.max(...cnt)) === b.count_p.indexOf(Math.max(...b.count_p)));
  const rr = refRows.find((x) => x.fn === a.fn);
  if (rr) upd("voice_activity_p vs result.json", Math.abs(g(dA, "voice_activity").probability - rr.vad));
  const m = mel(wa); let md = 0; for (let k = 0; k < MEL; k++) md = Math.max(md, Math.abs(m[k] - melA[i * MEL + k])); melMax = Math.max(melMax, md);
  for (const d of [dA, dB]) { timing.mel.push(d.cost.parts.mel); timing.tower.push(d.cost.parts.tower); timing.readers.push(d.cost.parts.readers); timing.ms.push(d.cost.ms); }
  if (i % 8 === 0) console.log(i, "vad", g(dA, "voice_activity").probability.toFixed(5), "ref", a.vad_p.toFixed(5), "turn", g(dA, "turn_end").probability.toFixed(5), "ref", a.turn_p.toFixed(5));
  if (fixDir && i === 0) {
    const eA = await model.encode({ samples: wa.subarray(WIN - a.n), sampleRate: 16000 }), eB = await model.encode({ samples: wb, sampleRate: 16000 });
    fixtures.push({ set: "A", i, ref: a, win: wa, melRef: melA.subarray(i * MEL, (i + 1) * MEL), e: eA }, { set: "B", i, ref: b, win: wb, melRef: melB.subarray(i * MEL, (i + 1) * MEL), e: eB });
  }
}
const med = (x) => [...x].sort((p, q) => p - q)[x.length >> 1];
const res = { n: 56, device: model.device, backend: model.backend, loadMs, maxAbsDiff: { ...maxd, "log-mel (JS vs torch, A)": melMax }, agreement: agree,
  medianMs: { mel: med(timing.mel), tower: med(timing.tower), readers: med(timing.readers), total: med(timing.ms) }, node: process.version };
console.log(JSON.stringify(res, null, 1));
if (outFile) fs.writeFileSync(outFile, JSON.stringify(res, null, 1) + "\n");
if (fixDir) {
  fs.mkdirSync(fixDir, { recursive: true });
  const f16 = Math.f16round, gz = (name, arr) => fs.writeFileSync(path.join(fixDir, name + ".gz"), zlib.gzipSync(Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength), { level: 9 }));
  const meta = { note: "Windows of FLEURS pt-BR dev clips (google/fleurs, CC-BY-4.0; see ATTRIBUTION.md), the torch Marimba references (ref.json entries) and the tower outputs of the s1s2 JS mel + ONNX tower on CPU. h12/h18 are stored float16-rounded (the readers round to float16 first); h9 is exact float32.", windows: [] };
  for (const f of fixtures) {
    const k = f.set + f.i;
    gz(k + ".win.f32", f.win); gz(k + ".mel_torch.f32", f.melRef); gz(k + ".h9.f32", f.e.taps.h9);
    gz(k + ".h12.f32", Float32Array.from(f.e.taps.h12, f16)); gz(k + ".h18.f32", Float32Array.from(f.e.taps.h18, f16));
    meta.windows.push({ key: k, set: f.set, nValid: f.e.nValid, ref: f.ref });
  }
  fs.writeFileSync(path.join(fixDir, "meta.json"), JSON.stringify(meta, null, 1) + "\n");
}
await model.dispose();
