// Audio media model: log-mel in JS -> frozen tower (ONNX, one pass for any number of questions) -> readers.
// Loads a media manifest's `audio` block (pinned files, sha256-checked like the text packages) and answers typed
// questions about one window of audio. Design and limits: spec/MULTIMODAL.md.

import { Files } from "../files.js";
import { createGate } from "../backend.js";
import { sigmoid, predicateFromP, choiceFromScores } from "../readout.js";
import { embeddingTableFromBytes, httpRangeTable, fileTable, type EmbeddingTable } from "../backends/embeddings.js";
import { nodeFloat16Workaround } from "../model.js";
import { parseSafetensors, type SafeTensors } from "../heads/safetensors.js";
import { detectPlatform, webgpuStatus, importOptional } from "../runtime.js";
import { createLogMel, type MelExtractor } from "./mel.js";
import { createPooledReader, frameMask, rightAlign, type PooledReaderRun } from "./audioReaders.js";
import { DecoderEngine, DecoderWindow, ortDecoder } from "./decoder.js";
import { termFraction, normalizeTerm } from "./decoderText.js";
import { SMALL_FILE_LIMIT } from "../files.js";
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
  /** The Qwen3-ASR decoder, when the manifest lists decoder readers and it was loaded. */
  decoder?: DecoderEngine;
  /** Why the decoder is not there: `"off"` (loadMedia option `decoder: false`) or the message of the load error (`decoder: "auto"`). */
  decoderMissing?: string;
}

/** What `model.decoder` says: whether the language and term questions can be answered, and if not, why. */
export interface DecoderStatus { loaded: boolean; reason?: string }

/** What one window costs once: tower outputs by tap (as [frames * dim]), real samples in the window, the mel. */
export interface AudioEncoding {
  taps: Record<string, Float32Array>; nValid: number; win: number; mel: Float32Array; ms: { mel: number; tower: number };
  /** The right-aligned window the tower read (kept so the decoder lane can re-align it to the left). */
  window: Float32Array;
  /** Tower taps of the left-aligned window (decoder questions on less than a full window); computed once. */
  left?: Promise<Record<string, Float32Array>>;
}

/** One question of a request, normalised. */
interface Ask { task: string; term?: string; options?: string[] }

export class AudioModel {
  readonly modality = "audio" as const;
  private mel: MelExtractor;
  private readers = new Map<string, { run?: PooledReaderRun; speech?: SpeechHeadReader; decoder?: "language" | "term" }>();
  private gate = createGate("audio");
  private inflight = new Set<Promise<unknown>>();
  private closing = false;
  private disposing: Promise<void> | null = null;

  constructor(private parts: AudioParts) {
    const { spec } = parts;
    this.mel = createLogMel(spec.preprocess, parts.filters);
    for (const [task, r] of Object.entries(spec.readers)) {
      if (!spec.tower.outputs[r.tap]) throw new Error(`audio reader ${task}: tap ${r.tap} is not a tower output (${Object.keys(spec.tower.outputs).join(", ")})`);
      if (r.kind === "decoder-language" || r.kind === "decoder-term") {
        if (!spec.decoder) throw new Error(`audio reader ${task}: kind ${r.kind} needs audio.decoder in the manifest`);
        if (parts.decoder) this.readers.set(task, { decoder: r.kind === "decoder-language" ? "language" : "term" });
      } else if (r.kind === "speech-head") {
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

  /** Whether the decoder (language, term) is loaded; when it is listed in the manifest but is not, the reason. Absent from a manifest without one. */
  get decoder(): DecoderStatus | undefined {
    if (!this.parts.spec.decoder) return undefined;
    return this.parts.decoder ? { loaded: true } : { loaded: false, reason: this.parts.decoderMissing ?? "not loaded" };
  }

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
      return { taps, nValid, win, mel, window, ms: { mel: t1 - t0, tower: t2 - t1 } };
    } finally { rel(); }
  }

  /** Answers for tasks on an encoding (no tower pass for the readers; decoder questions on less than a full window run the tower once more). */
  async read(enc: AudioEncoding, questions: MediaDecisionRequest["questions"], timing?: Record<string, number>): Promise<MediaAnswer[]> {
    const asks = this.asksOf(questions);
    const { spec } = this.parts, sr = spec.preprocess.sampleRate;
    const answers: MediaAnswer[] = new Array(asks.length);
    const dec: number[] = [];
    for (let i = 0; i < asks.length; i++) {
      const a = asks[i], r = this.readers.get(a.task)!;
      if (r.run) answers[i] = r.run.read(enc.taps, { win: enc.win, sampleRate: sr, nValid: enc.nValid });
      else if (r.speech) answers[i] = await this.speech(a.task, r.speech, enc.taps, enc.nValid, enc.win);
      else dec.push(i);
    }
    if (dec.length) {
      const sub = await this.decoderAnswers(dec.map((i) => asks[i]), () => this.decoderFeatures(enc, (spec.readers[asks[dec[0]].task] as any).tap, timing), timing);
      dec.forEach((i, k) => { answers[i] = sub[k]; });
    }
    return answers;
  }

  /**
   * Decoder questions (language, term) answered from the tower's final vectors of a window with the audio at index 0 ([104 * 1024],
   * what `encode` yields for a full window). For tests and offline features; `read` and `decide` do this for you.
   */
  async decodeFeatures(features: Float32Array, questions: MediaDecisionRequest["questions"], timing?: Record<string, number>): Promise<MediaAnswer[]> {
    const asks = this.asksOf(questions);
    if (asks.some((a) => !this.readers.get(a.task)!.decoder)) throw new Error("decodeFeatures: only decoder tasks (language, term) can be answered from features");
    return this.decoderAnswers(asks, async () => features, timing);
  }

  /** Decoder questions in flight are counted: `dispose` waits for them instead of closing the session under a running `feed`. */
  private decoderAnswers(asks: Ask[], features: () => Promise<Float32Array>, timing?: Record<string, number>): Promise<MediaAnswer[]> {
    if (this.closing) return Promise.reject(new Error("audio: backend already disposed"));
    const run = this.runDecoderAnswers(asks, features, timing);
    this.inflight.add(run);
    const done = () => { this.inflight.delete(run); };
    run.then(done, done);
    return run;
  }

  private async runDecoderAnswers(asks: Ask[], features: () => Promise<Float32Array>, timing?: Record<string, number>): Promise<MediaAnswer[]> {
    const tm = (k: string, t0: number) => { if (timing) timing[k] = (timing[k] ?? 0) + performance.now() - t0; };
    const out: MediaAnswer[] = [];
    let hyps: string[] | null = null;
    const scoreCache = new Map<string, { codes: string[]; scores: number[] }>();
    const t0 = performance.now();
    const win: DecoderWindow = await this.parts.decoder!.open(await features());
    tm("decoderOpen", t0);
    try {
      for (const a of asks) {
        if (this.readers.get(a.task)!.decoder === "language") {
          const t1 = performance.now(), key = (a.options ?? []).join(",");
          let sc = scoreCache.get(key);
          if (!sc) { sc = await win.language(a.options); scoreCache.set(key, sc); }
          tm("language", t1);
          const c = choiceFromScores(a.task, sc.codes, sc.scores, "logits");
          out.push({ ...c, modality: "audio", task: a.task, decision: c.choice, available: true });
        } else {
          const t1 = performance.now();
          if (!hyps) hyps = await win.hypotheses();
          tm("term", t1);
          const p = termFraction(a.term!, hyps);
          out.push({ ...predicateFromP(a.task, p, "logits"), modality: "audio", task: a.task, decision: p >= 0.5 ? "yes" : "no", threshold: 0.5, hypotheses: hyps, available: true });
        }
      }
    } finally { await win.close(); }
    return out;
  }

  /** Tower output of the window with the audio at index 0 (what the decoder lane reads); the right-aligned pass when the window is full. */
  private async decoderFeatures(enc: AudioEncoding, tap: string, timing?: Record<string, number>): Promise<Float32Array> {
    if (enc.nValid >= enc.win) return enc.taps[tap];
    if (!enc.left) {
      enc.left = (async () => {
        const rel = await this.gate.acquire();
        try {
          const t0 = performance.now();
          const w = new Float32Array(enc.win);
          w.set(enc.window.subarray(enc.win - enc.nValid));
          const taps = await this.parts.runtime.tower(this.mel(w));
          if (timing) timing.leftTower = performance.now() - t0;
          return taps;
        } finally { rel(); }
      })();
    }
    return (await enc.left)[tap];
  }

  private asksOf(questions: MediaDecisionRequest["questions"]): Ask[] {
    if (!Array.isArray(questions) || !questions.length) throw new Error("decide: questions must be a non-empty array of task names");
    return questions.map((q) => {
      const o = typeof q === "string" ? { task: q } : q, task = (o.task ?? o.name) as string;
      const r = this.readers.get(task);
      if (!r) {
        const need = this.parts.spec.readers[task];
        if (need && (need.kind === "decoder-language" || need.kind === "decoder-term")) {
          const why = this.parts.decoderMissing && this.parts.decoderMissing !== "off" ? `failed to load: ${this.parts.decoderMissing}` : "was not loaded (loadMedia option decoder: false)";
          throw new Error(`decide: task ${JSON.stringify(task)} needs the decoder, which ${why}`);
        }
        throw new Error(`decide: unknown audio task ${JSON.stringify(task)} (this package has: ${this.tasks.join(", ")})`);
      }
      if (r.decoder === "term" && !(typeof o.term === "string" && normalizeTerm(o.term))) throw new Error(`decide: task ${JSON.stringify(task)} needs { task, term } with a word that has letters or digits`);
      if (r.decoder === "language" && o.options !== undefined) {
        if (!(Array.isArray(o.options) && o.options.length >= 2 && new Set(o.options).size === o.options.length)) throw new Error(`decide: language options must be at least two distinct codes`);
        const known = this.parts.spec.decoder!.languages, bad = o.options.filter((c) => typeof c !== "string" || !Object.hasOwn(known, c));
        if (bad.length) throw new Error(`decide: language options ${bad.map((c) => JSON.stringify(c)).join(", ")} are not codes of this package (${Object.keys(known).join(", ")})`);
      }
      return { task, term: o.term, options: o.options };
    });
  }

  /** Answers for the requested tasks on one window of audio: one tower pass, then each reader. */
  async decide(req: MediaDecisionRequest): Promise<MediaDecision> {
    this.asksOf(req.questions);
    const t0 = performance.now();
    const enc = await this.encode(req.audio!), t1 = performance.now();
    const timing: Record<string, number> = {};
    const answers = await this.read(enc, req.questions, timing), t3 = performance.now();
    return { answers, cost: { ms: t3 - t0, parts: { mel: enc.ms.mel, tower: enc.ms.tower, readers: t3 - t1, ...timing }, passes: 1 + (timing.leftTower !== undefined ? 1 : 0), backend: this.backend, device: this.device } };
  }

  private async speech(task: string, r: SpeechHeadReader, taps: Record<string, Float32Array>, nValid: number, win: number): Promise<MediaAnswer> {
    const mask = frameMask(nValid, win, r.frames);
    if (!mask.some((v) => v > 0)) { const u: UnavailableAnswer = { name: task, task, modality: "audio", available: false, reason: "no frame of real audio" }; return u; }
    const score = await this.parts.runtime.speech(task, taps[r.tap], mask);
    if (!Number.isFinite(score)) throw new Error(`audio reader ${task}: head score is not finite`);
    const raw = score / r.temperature, p = sigmoid(raw), a = predicateFromP(task, p, "head", raw);
    return { ...a, modality: "audio", task, decision: p >= r.threshold ? "yes" : "no", threshold: r.threshold, available: true };
  }

  /** Waits for the decoder questions in flight (the session is released only after them), then releases the decoder and the tower. */
  dispose(): Promise<void> {
    return this.disposing ??= (async () => {
      this.closing = true;
      await Promise.allSettled([...this.inflight]);
      await this.gate.dispose(async () => { try { await this.parts.decoder?.close(); } finally { await this.parts.runtime.dispose(); } });
    })();
  }
}

// ---- loading from a manifest ---------------------------------------------------------------------------------------

export interface AudioLoadOptions {
  device?: "auto" | "webgpu" | "wasm" | "cpu";
  modules?: { ort?: any };
  files: Files;
  repo: string;
  revision: string;
  onStatus?: (m: string) => void;
  /**
   * The decoder (language, term questions) of a manifest that has one. `"auto"` (default): load it; if that fails (not enough GPU memory, a server
   * without Range requests, a build for another device) the model runs without it, `onStatus` and `model.decoder` say why, and only a language or
   * term question throws. `true`: a failed load throws. `false`: nothing of the decoder is loaded.
   */
  decoder?: boolean | "auto";
  fetch?: typeof fetch;
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
    if (r.kind === "decoder-language" || r.kind === "decoder-term") continue;
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

  let decoder: DecoderEngine | undefined, decoderMissing: string | undefined;
  const mode = o.decoder ?? "auto";
  if (spec.decoder && mode === false) decoderMissing = "off";
  else if (spec.decoder) {
    try { decoder = await loadDecoder(spec.decoder, { ort, files, repo, revision, device: device as any, platform, status, fetch: o.fetch }); }
    catch (e) {
      if (mode === true) { await tower.release?.(); for (const h of Object.values(heads)) await h.release?.(); throw e; }
      decoderMissing = e instanceof Error ? e.message : String(e);
      status(`decoder not loaded, language and term are unavailable: ${decoderMissing}`);
    }
  }

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
  return new AudioModel({ spec, weights, questions, runtime, decoder, decoderMissing });
}

async function loadDecoder(d: NonNullable<AudioSpec["decoder"]>, o: { ort: any; files: Files; repo: string; revision: string; device: "webgpu" | "wasm" | "cpu"; platform: string; status: (m: string) => void; fetch?: typeof fetch }): Promise<DecoderEngine> {
  const { ort, files, repo, revision, device } = o;
  const build = device === "webgpu" ? { graph: d.graph, data: d.data } : d.cpuBuild;
  if (!build || !d.devices.includes(device)) throw new Error(`audio decoder lists devices ${d.devices.join(", ")}, not ${device}`);
  const H = d.model.hidden, base = (p: string) => p.split("/").pop()!;
  o.status("loading decoder tokens…");
  const tokens = await files.json(repo, revision, d.tokens);
  if (!Array.isArray(tokens?.tokens) || tokens.tokens.length !== d.model.vocab) throw new Error(`decoder tokens: expected ${d.model.vocab} pieces`);
  o.status("loading decoder embeddings…");
  let table: EmbeddingTable;
  if (o.platform === "node") table = await fileTable(await files.local(repo, revision, d.embeddings), H);
  else if ((d.embeddings.size ?? Infinity) <= SMALL_FILE_LIMIT) table = embeddingTableFromBytes(await files.bytes(repo, revision, d.embeddings), H);
  else table = httpRangeTable(await files.large(repo, revision, d.embeddings), d.embeddings.size!, H, o.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a)));
  o.status(`loading decoder (${device})…`);
  let session: any;
  try {
    if (o.platform === "node") {
      await files.local(repo, revision, build.data);
      nodeFloat16Workaround(ort);
      session = await ort.InferenceSession.create(await files.local(repo, revision, build.graph), { executionProviders: ["cpu"], graphOptimizationLevel: "all", logSeverityLevel: 3 });
    } else {
      const graph = await files.bytes(repo, revision, build.graph), dataUrl = await files.large(repo, revision, build.data);
      session = await ort.InferenceSession.create(graph, {
        executionProviders: [device === "webgpu" ? { name: "webgpu", validationMode: "disabled" } : "wasm"],
        externalData: [{ path: base(build.data.path), data: dataUrl }],
        graphOptimizationLevel: "all", enableGraphCapture: false, logSeverityLevel: 3,
        ...(device === "webgpu" ? { preferredOutputLocation: "gpu-buffer" } : {}),
      });
    }
  } catch (e) { await table.close?.(); throw e; }
  const be = ortDecoder(session, ort, { hidden: H, vocab: d.model.vocab, kvHeadDim: d.model.headDim, layers: d.model.layers, kvHeads: d.model.kvHeads });
  return new DecoderEngine(be, table, d, tokens);
}
