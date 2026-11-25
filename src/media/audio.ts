// Audio media model: log-mel in JS -> frozen tower (ONNX, one pass for any number of questions) -> readers.
// Loads a media manifest's `audio` block (pinned files, sha256-checked like the text packages) and answers typed
// questions about one window of audio. Design and limits: spec/MULTIMODAL.md.

import { Files } from "../files.js";
import { createGate } from "../backend.js";
import { sigmoid, predicateFromP } from "../readout.js";
import { parseSafetensors, type SafeTensors } from "../heads/safetensors.js";
import { detectPlatform, webgpuStatus, importOptional } from "../runtime.js";
import { createLogMel, type MelExtractor } from "./mel.js";
import { createPooledReader, frameMask, rightAlign, type PooledReaderRun } from "./audioReaders.js";
import type { AudioSpec, MediaAnswer, MediaDecision, MediaDecisionRequest, SpeechHeadReader, UnavailableAnswer } from "./types.js";

/** What the audio model needs from a runtime: run the tower on a mel, run a speech head. Tests inject fakes. */
export interface AudioRuntime {
  backend: string;
  device: string;
  /** mel: [nMels * frames] row-major. Returns every tap by output name, as [frames * dim] float32. */
  tower(mel: Float32Array): Promise<Record<string, Float32Array>>;
  /** Score of a speech head from a tap ([frames * dim]). */
  speech(name: string, tap: Float32Array, mask: Float32Array): Promise<number>;
  dispose(): Promise<void>;
}

export interface AudioParts {
  spec: AudioSpec;
  weights: Record<string, SafeTensors>;
  /** Speech heads: name -> question vector. */
  questions: Record<string, Float32Array>;
  runtime: AudioRuntime;
  /** Mel filters (default: generated). */
  filters?: Float32Array;
}

/** What one window costs once: tower outputs by tap (as [frames * dim]), real samples in the window, the mel. */
export interface AudioEncoding { taps: Record<string, Float32Array>; nValid: number; win: number; mel: Float32Array; ms: { mel: number; tower: number } }

export class AudioModel {
  readonly modality = "audio" as const;
  private mel: MelExtractor;
  private readers = new Map<string, { run?: PooledReaderRun; speech?: SpeechHeadReader }>();
  private gate = createGate("audio");

  constructor(private parts: AudioParts) {
    const { spec } = parts;
    this.mel = createLogMel(spec.preprocess, parts.filters);
    for (const [task, r] of Object.entries(spec.readers)) {
      if (!spec.tower.outputs[r.tap]) throw new Error(`audio reader ${task}: tap ${r.tap} is not a tower output (${Object.keys(spec.tower.outputs).join(", ")})`);
      if (r.kind === "speech-head") {
        const q = parts.questions[task];
        if (!q || q.length !== r.question.dim) throw new Error(`audio reader ${task}: question vector has ${q?.length} values, expected ${r.question.dim}`);
        this.readers.set(task, { speech: r });
      } else {
        const st = parts.weights[task];
        if (!st) throw new Error(`audio reader ${task}: weights not loaded`);
        this.readers.set(task, { run: createPooledReader(task, r, st, (tap) => spec.tower.outputs[tap]) });
      }
    }
  }

  get tasks(): string[] { return [...this.readers.keys()]; }
  get device(): string { return this.parts.runtime.device; }
  get backend(): string { return this.parts.runtime.backend; }

  get spec(): AudioSpec { return this.parts.spec; }

  private checkAudio(a: MediaDecisionRequest["audio"]): asserts a is NonNullable<MediaDecisionRequest["audio"]> {
    if (!a || !(a.samples instanceof Float32Array)) throw new Error("decide: audio.samples (Float32Array) is required");
    const sr = this.parts.spec.preprocess.sampleRate;
    if (a.sampleRate !== sr) throw new Error(`decide: audio is ${a.sampleRate} Hz, this package takes ${sr} Hz mono; resample first (spec/MULTIMODAL.md, limits)`);
    if (a.samples.length <= this.parts.spec.preprocess.nFft / 2) throw new Error("decide: audio is too short");
    // only the last window is read: a NaN or Infinity in it would turn every mel frame, and every probability, into NaN
    const x = a.samples, from = Math.max(0, x.length - Math.round(this.parts.spec.preprocess.windowSeconds * sr));
    for (let i = from; i < x.length; i++) if (!Number.isFinite(x[i])) throw new Error(`decide: audio.samples[${i}] is ${x[i]}; samples must be finite numbers`);
  }

  /** The shared part of every question: log-mel and ONE tower pass. Readers can then run many times on the taps (`read`). */
  async encode(audio: NonNullable<MediaDecisionRequest["audio"]>): Promise<AudioEncoding> {
    this.checkAudio(audio);
    const { spec } = this.parts, sr = spec.preprocess.sampleRate;
    const rel = await this.gate.acquire();
    try {
      const t0 = performance.now();
      const win = Math.round(spec.preprocess.windowSeconds * sr);
      const { window, nValid } = rightAlign(audio.samples, win);
      const mel = this.mel(window), t1 = performance.now();
      const taps = await this.parts.runtime.tower(mel), t2 = performance.now();
      return { taps, nValid, win, mel, ms: { mel: t1 - t0, tower: t2 - t1 } };
    } finally { rel(); }
  }

  /** Answers for tasks on an encoding (no tower pass: this is the cost of an extra question). */
  async read(enc: AudioEncoding, questions: MediaDecisionRequest["questions"]): Promise<MediaAnswer[]> {
    const tasks = this.tasksOf(questions);
    const { spec } = this.parts, sr = spec.preprocess.sampleRate;
    const answers: MediaAnswer[] = [];
    for (const t of tasks) {
      const r = this.readers.get(t)!;
      if (r.run) answers.push(r.run.read(enc.taps, { win: enc.win, sampleRate: sr, nValid: enc.nValid }));
      else answers.push(await this.speech(t, r.speech!, enc.taps, enc.nValid, enc.win));
    }
    return answers;
  }

  private tasksOf(questions: MediaDecisionRequest["questions"]): string[] {
    if (!Array.isArray(questions) || !questions.length) throw new Error("decide: questions must be a non-empty array of task names");
    const tasks = questions.map((q) => (typeof q === "string" ? q : (q.task ?? q.name) as string));
    for (const t of tasks) if (!this.readers.has(t)) throw new Error(`decide: unknown audio task ${JSON.stringify(t)} (this package has: ${this.tasks.join(", ")})`);
    return tasks;
  }

  /** Answers for the requested tasks on one window of audio: one tower pass, then each reader. */
  async decide(req: MediaDecisionRequest): Promise<MediaDecision> {
    this.tasksOf(req.questions);
    const t0 = performance.now();
    const enc = await this.encode(req.audio!), t1 = performance.now();
    const answers = await this.read(enc, req.questions), t3 = performance.now();
    return { answers, cost: { ms: t3 - t0, parts: { mel: enc.ms.mel, tower: enc.ms.tower, readers: t3 - t1 }, passes: 1, backend: this.backend, device: this.device } };
  }

  private async speech(task: string, r: SpeechHeadReader, taps: Record<string, Float32Array>, nValid: number, win: number): Promise<MediaAnswer> {
    const mask = frameMask(nValid, win, r.frames);
    if (!mask.some((v) => v > 0)) { const u: UnavailableAnswer = { name: task, task, modality: "audio", available: false, reason: "no frame of real audio" }; return u; }
    const score = await this.parts.runtime.speech(task, taps[r.tap], mask);
    if (!Number.isFinite(score)) throw new Error(`audio reader ${task}: head score is not finite`);
    const raw = score / r.temperature, p = sigmoid(raw), a = predicateFromP(task, p, "head", raw);
    return { ...a, modality: "audio", task, decision: p >= r.threshold ? "yes" : "no", threshold: r.threshold, available: true };
  }

  async dispose(): Promise<void> { await this.gate.dispose(() => this.parts.runtime.dispose()); }
}

// ---- loading from a manifest ---------------------------------------------------------------------------------------

export interface AudioLoadOptions {
  device?: "auto" | "webgpu" | "wasm" | "cpu";
  modules?: { ort?: any };
  files: Files;
  repo: string;
  revision: string;
  onStatus?: (m: string) => void;
}

const asF32 = (u: Uint8Array): Float32Array => new Float32Array(u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength));
const ab = (u: Uint8Array): ArrayBuffer => u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer;

export async function loadAudio(spec: AudioSpec, o: AudioLoadOptions): Promise<AudioModel> {
  const status = o.onStatus ?? (() => {});
  const platform = detectPlatform();
  let device = o.device ?? "auto";
  if (device === "auto") device = platform === "node" ? "cpu" : (await webgpuStatus()).ok ? "webgpu" : "wasm";
  if (!spec.tower.devices.includes(device as any)) throw new Error(`audio tower lists devices ${spec.tower.devices.join(", ")}, not ${device}`);
  if ((platform === "node") !== (device === "cpu")) throw new Error(`audio: device ${device} does not run on ${platform}`);
  const ort: any = o.modules?.ort ?? (platform === "node"
    ? await importOptional("onnxruntime-node", () => import(/* @vite-ignore */ "onnxruntime-node"), "ONNX Runtime in Node")
    : await importOptional("onnxruntime-web", () => import("onnxruntime-web/webgpu"), "ONNX Runtime in the browser"));
  const { files, repo, revision } = o;

  status("loading audio readers…");
  const weights: Record<string, SafeTensors> = {}, questions: Record<string, Float32Array> = {}, speechBytes: Record<string, Uint8Array> = {};
  for (const [task, r] of Object.entries(spec.readers)) {
    if (r.kind === "speech-head") {
      speechBytes[task] = await files.bytes(repo, revision, r.graph);
      questions[task] = asF32(await files.bytes(repo, revision, r.question));
    } else weights[task] = parseSafetensors(ab(await files.bytes(repo, revision, r.weights)));
  }
  status(`loading audio tower (${device})…`);
  const towerSrc = await files.large(repo, revision, spec.tower.file);
  const opts = { executionProviders: [device === "cpu" ? "cpu" : device], graphOptimizationLevel: "all", logSeverityLevel: 3 };
  const tower = await ort.InferenceSession.create(towerSrc, opts);
  const heads: Record<string, any> = {};
  try {
    for (const [task, b] of Object.entries(speechBytes)) heads[task] = await ort.InferenceSession.create(b, opts);
  } catch (e) { await tower.release?.(); throw e; }

  const runtime: AudioRuntime = {
    backend: `onnxruntime-${platform === "node" ? "node" : "web"}`, device,
    async tower(mel) {
      const inp = spec.tower.input;
      const out = await tower.run({ [inp.name]: new ort.Tensor("float32", mel, inp.shape) });
      const taps: Record<string, Float32Array> = {};
      for (const [name, t] of Object.entries(spec.tower.outputs)) {
        const x = out[name];
        if (!x) throw new Error(`audio tower: output ${name} missing`);
        const data = x.data as Float32Array;
        if (data.length !== t.frames * t.dim) throw new Error(`audio tower: output ${name} has ${data.length} values, expected ${t.frames} x ${t.dim}`);
        taps[name] = data;
      }
      return taps;
    },
    async speech(task, tap, mask) {
      const r = spec.readers[task] as SpeechHeadReader, tp = spec.tower.outputs[r.tap];
      const res = await heads[task].run({
        [r.inputs.tap]: new ort.Tensor("float32", tap, [tp.frames, tp.dim]),
        [r.inputs.mask]: new ort.Tensor("float32", mask, [1, r.frames]),
        [r.inputs.question]: new ort.Tensor("float32", questions[task], [1, r.question.dim]),
      });
      return Number(res[r.inputs.output].data[0]);
    },
    async dispose() { for (const h of Object.values(heads)) await h.release?.(); await tower.release?.(); },
  };
  return new AudioModel({ spec, weights, questions, runtime });
}
