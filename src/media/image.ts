// Image decision reader (spec/MULTIMODAL.md section 10): vision graph once, text prefix once, one independent branch per question over the prefix cache,
// scores = hidden(last) . embedding rows of the option letters, softmax over the options supplied.

import type { Files } from "../files.js";
import { detectPlatform, webgpuStatus, importOptional } from "../runtime.js";
import { embeddingTable, fileTable, httpRangeTable, embeddingTableFromBytes, f16ToF32, type EmbeddingTable } from "../backends/embeddings.js";
import { tokenizerFromJson, type Tok } from "../tok.js";
import { f32ToF16Bits } from "./decoder.js";
import { positionEmbeddings, prepareImage, prefixText, suffixText, mropePositions, nextPosition, optionScores, rgbaToRgb, MIN_OPTIONS, MAX_OPTIONS } from "./imagePrep.js";
import type { ImageSpec, MediaCost } from "./types.js";

export interface ImageInput { rgb: Uint8Array | Uint8ClampedArray; width: number; height: number; channels?: 3 | 4 }
export interface ImageQuestion { name?: string; instruction: string; options: string[] }
export interface ImageRequest { image: ImageInput; questions: ImageQuestion[]; context?: string }
export interface ImageAnswer {
  name: string; modality: "image"; task: "choice"; available: true; path: "s1";
  decision: string; choice: number; probability: number; probabilities: number[]; logits: number[]; margin: number; rawMargin: number; options: string[];
}
export interface ImageDecision { answers: ImageAnswer[]; cost: MediaCost & { imageTokens: number; grid: [number, number, number]; resized: [number, number] } }

export interface ImageLoadOptions {
  device?: "auto" | "webgpu" | "wasm" | "cpu";
  /** Run the vision graph on another provider than the text graph (e.g. "wasm" with the text on WebGPU: see spec section 10.8, the WebGPU Gather of the vision graph's position embedding). */
  visionDevice?: "webgpu" | "wasm" | "cpu";
  modules?: { ort?: any; transformers?: any };
  files: Files;
  repo: string;
  revision: string;
  onStatus?: (m: string) => void;
  /** Replace the tokenizer built from the manifest's tokenizer.json (tests). */
  tokenizer?: Tok;
  /** Run every question as one full pass over prefix + suffix, with no prefix cache (slower; the diagnostic for a provider whose attention with a past is wrong). */
  noCache?: boolean;
  /** Diagnostics: called with the prefix inputs of each decide() (embeddings, positions, DeepStack rows) before the text graph runs. */
  onPrefix?: (p: { embeds: Float32Array; positions: BigInt64Array; deepstack: Float32Array[]; length: number }) => void;
  fetch?: typeof fetch;
}

const base = (p: string) => p.split("/").pop()!;
const u16 = (u: Uint8Array) => new Uint16Array(u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength));

export class ImageModel {
  constructor(private spec: ImageSpec, private ort: any, private vision: any, private text: any, private table: EmbeddingTable, private tok: Tok, private rows: Float32Array,
              readonly device: string, private noCache = false, private posTable: Float32Array | null = null, private onPrefix?: ImageLoadOptions["onPrefix"]) {}

  private tensor(f: Float32Array, dims: number[]): any {
    const T = this.ort.Tensor;
    if (this.spec.text.ioDtype === "float32") return new T("float32", f, dims);
    const u = new Uint16Array(f.length);
    for (let i = 0; i < f.length; i++) u[i] = f32ToF16Bits(f[i]);
    return new T("float16", u, dims);
  }
  private async toF32(t: any): Promise<Float32Array> {
    const d = typeof t.getData === "function" ? await t.getData(true) : t.data;
    if (d instanceof Float32Array) return d;
    if (d instanceof Uint16Array) return f16ToF32(d);
    return Float32Array.from(d as ArrayLike<number>, Number);
  }
  private async embed(ids: number[]): Promise<Float32Array> { return f16ToF32(await this.table.rows(ids)); }

  async decide(req: ImageRequest): Promise<ImageDecision> {
    const t0 = Date.now(), parts: Record<string, number> = {}, lap = (k: string, t: number) => { parts[k] = Date.now() - t; };
    const { spec } = this, H = spec.text.model.hidden, T = this.ort.Tensor, pp = spec.preprocess;
    if (!req.questions?.length) throw new Error("decide: at least one question");
    for (const q of req.questions) if (!Array.isArray(q.options) || q.options.length < Math.max(MIN_OPTIONS, spec.readout.minOptions) || q.options.length > Math.min(MAX_OPTIONS, spec.readout.maxOptions))
      throw new Error(`a question takes ${spec.readout.minOptions} to ${spec.readout.maxOptions} options, got ${q.options?.length}`);
    const im = req.image, ch = im.channels ?? 3;
    const rgb = ch === 4 ? rgbaToRgb(im.rgb) : im.rgb instanceof Uint8Array ? im.rgb : Uint8Array.from(im.rgb);
    // 1. pixels -> vision graph (once)
    let t = Date.now();
    const prep = prepareImage(rgb, im.height, im.width, { patchSize: pp.patchSize, mergeSize: pp.mergeSize, temporalPatchSize: pp.temporalPatchSize, mean: pp.mean, std: pp.std, minPixels: pp.minPixels, maxPixels: pp.maxPixels });
    lap("preprocess", t); t = Date.now();
    const vfeeds: Record<string, any> = {
      [spec.vision.inputs.pixelValues]: this.tensor(prep.pixelValues, [prep.rows, prep.dim]),
      [spec.vision.inputs.grid]: new T("int64", BigInt64Array.from(prep.grid.map(BigInt)), [1, 3]),
    };
    if (spec.vision.posEmbeds) vfeeds[spec.vision.posEmbeds.input] = this.tensor(positionEmbeddings(this.posTable!, spec.vision.posEmbeds.side, this.posTable!.length / spec.vision.posEmbeds.side ** 2, prep.grid, pp.mergeSize), [prep.rows, this.posTable!.length / spec.vision.posEmbeds.side ** 2]);
    const vo = await this.vision.run(vfeeds);
    const feats = await this.toF32(vo[spec.vision.outputs.features]);
    const ds = await Promise.all(spec.vision.outputs.deepstack.map((n) => this.toF32(vo[n])));
    lap("vision", t); t = Date.now();
    const nImg = prep.mergedTokens;
    if (feats.length !== nImg * H) throw new Error(`vision graph returned ${feats.length / H} rows for ${nImg} image tokens`);
    // 2. prefix: ids, embeddings with the image rows scattered in, DeepStack tensors, positions; one pass; keep the cache
    const preIds = this.tok.encode(prefixText(req.context ?? "", nImg));
    const isImg = preIds.map((x) => x === spec.prompt.imagePad);
    const prefixEmb = await this.embed(preIds);
    let k = 0;
    for (let i = 0; i < preIds.length; i++) if (isImg[i]) { prefixEmb.set(feats.subarray(k * H, (k + 1) * H), i * H); k++; }
    if (k !== nImg) throw new Error(`prompt has ${k} image pads for ${nImg} image tokens`);
    const pos = mropePositions(isImg, prep.grid, pp.mergeSize);
    if (this.noCache) return this.decideFull(req, prep, feats, ds, preIds, prefixEmb, parts, t0);
    const S = preIds.length, nL = spec.text.model.layers, kvH = spec.text.model.kvHeads, hd = spec.text.model.headDim;
    const feeds: Record<string, any> = {
      [spec.text.inputs.embeds]: this.tensor(prefixEmb, [1, S, H]),
      [spec.text.inputs.attentionMask]: new T("int64", new BigInt64Array(S).fill(1n), [1, S]),
      [spec.text.inputs.positionIds]: new T("int64", BigInt64Array.from([...pos[0], ...pos[1], ...pos[2]], (v) => BigInt(v)), [3, 1, S]),
    };
    spec.text.inputs.deepstack.forEach((name, d) => {
      const z = new Float32Array(S * H);
      let r = 0;
      for (let i = 0; i < S; i++) if (isImg[i]) { z.set(ds[d].subarray(r * H, (r + 1) * H), i * H); r++; }
      feeds[name] = this.tensor(z, [1, S, H]);
    });
    if (this.onPrefix) this.onPrefix({ embeds: prefixEmb, positions: BigInt64Array.from([...pos[0], ...pos[1], ...pos[2]], (v) => BigInt(v)), deepstack: spec.text.inputs.deepstack.map((n) => feeds[n].data ?? new Float32Array(0)), length: S });
    const empty = () => this.tensor(new Float32Array(0), [1, kvH, 0, hd]);
    const past = (l: number, kv: "key" | "value") => (kv === "key" ? spec.text.cache.pastKey : spec.text.cache.pastValue).replace("%d", String(l));
    const present = (l: number, kv: "key" | "value") => (kv === "key" ? spec.text.cache.presentKey : spec.text.cache.presentValue).replace("%d", String(l));
    for (let l = 0; l < nL; l++) { feeds[past(l, "key")] = empty(); feeds[past(l, "value")] = empty(); }
    const fetches = [...Array(nL).keys()].flatMap((l) => [present(l, "key"), present(l, "value")]);
    const pre = await this.text.run(feeds, fetches);
    lap("prefix", t);
    // 3. branches: each over the same prefix cache (inputs are never modified by the runtime), zero DeepStack, positions continue the prefix
    const base0 = nextPosition(pos), answers: ImageAnswer[] = [];
    t = Date.now();
    for (let qi = 0; qi < req.questions.length; qi++) {
      const q = req.questions[qi];
      const sIds = this.tok.encode(suffixText(q.instruction, q.options)), L = sIds.length;
      const f2: Record<string, any> = {
        [spec.text.inputs.embeds]: this.tensor(await this.embed(sIds), [1, L, H]),
        [spec.text.inputs.attentionMask]: new T("int64", new BigInt64Array(S + L).fill(1n), [1, S + L]),
        [spec.text.inputs.positionIds]: new T("int64", BigInt64Array.from({ length: 3 * L }, (_, i) => BigInt(base0 + (i % L))), [3, 1, L]),
      };
      for (const name of spec.text.inputs.deepstack) f2[name] = this.tensor(new Float32Array(L * H), [1, L, H]);
      for (let l = 0; l < nL; l++) { f2[past(l, "key")] = pre[present(l, "key")]; f2[past(l, "value")] = pre[present(l, "value")]; }
      const r = await this.text.run(f2, [spec.text.output.hidden]);
      const h = await this.toF32(r[spec.text.output.hidden]);
      const last = h.subarray((L - 1) * H, L * H);
      const sc = optionScores(last, this.rows, q.options.length, spec.readout.temperature);
      answers.push({ name: q.name ?? `q${qi}`, modality: "image", task: "choice", available: true, path: "s1", decision: q.options[sc.choice], choice: sc.choice, probability: sc.probabilities[sc.choice],
        probabilities: sc.probabilities, logits: sc.logits, margin: sc.margin, rawMargin: sc.rawMargin, options: q.options });
    }
    lap("branches", t);
    for (const v of Object.values(pre)) (v as any)?.dispose?.();
    return { answers, cost: { ms: Date.now() - t0, parts, passes: 1 + 1 + req.questions.length, backend: "onnxruntime", device: this.device, imageTokens: nImg, grid: prep.grid, resized: prep.resized } };
  }

  /** One full pass per question (no cache): the same inputs as the cached path, concatenated. */
  private async decideFull(req: ImageRequest, prep: any, feats: Float32Array, ds: Float32Array[], preIds: number[], prefixEmb: Float32Array, parts: Record<string, number>, t0: number): Promise<ImageDecision> {
    const { spec } = this, H = spec.text.model.hidden, T = this.ort.Tensor, nL = spec.text.model.layers, kvH = spec.text.model.kvHeads, hd = spec.text.model.headDim;
    const answers: ImageAnswer[] = [];
    for (let qi = 0; qi < req.questions.length; qi++) {
      const q = req.questions[qi], sIds = this.tok.encode(suffixText(q.instruction, q.options)), ids = [...preIds, ...sIds], S = ids.length;
      const isImg = ids.map((x) => x === spec.prompt.imagePad), emb = new Float32Array(S * H);
      emb.set(prefixEmb); emb.set(await this.embed(sIds), preIds.length * H);
      const pos = mropePositions(isImg, prep.grid, spec.preprocess.mergeSize);
      const feeds: Record<string, any> = { [spec.text.inputs.embeds]: this.tensor(emb, [1, S, H]), [spec.text.inputs.attentionMask]: new T("int64", new BigInt64Array(S).fill(1n), [1, S]),
        [spec.text.inputs.positionIds]: new T("int64", BigInt64Array.from([...pos[0], ...pos[1], ...pos[2]], (v) => BigInt(v)), [3, 1, S]) };
      spec.text.inputs.deepstack.forEach((name, d) => { const z = new Float32Array(S * H); let r = 0; for (let i = 0; i < S; i++) if (isImg[i]) { z.set(ds[d].subarray(r * H, (r + 1) * H), i * H); r++; } feeds[name] = this.tensor(z, [1, S, H]); });
      for (let l = 0; l < nL; l++) for (const n of [spec.text.cache.pastKey, spec.text.cache.pastValue]) feeds[n.replace("%d", String(l))] = this.tensor(new Float32Array(0), [1, kvH, 0, hd]);
      const r = await this.text.run(feeds, [spec.text.output.hidden]);
      const h = await this.toF32(r[spec.text.output.hidden]);
      const sc = optionScores(h.subarray((S - 1) * H, S * H), this.rows, q.options.length, spec.readout.temperature);
      answers.push({ name: q.name ?? `q${qi}`, modality: "image", task: "choice", available: true, path: "s1", decision: q.options[sc.choice], choice: sc.choice, probability: sc.probabilities[sc.choice], probabilities: sc.probabilities, logits: sc.logits, margin: sc.margin, rawMargin: sc.rawMargin, options: q.options });
    }
    return { answers, cost: { ms: Date.now() - t0, parts, passes: 1 + req.questions.length, backend: "onnxruntime", device: this.device, imageTokens: prep.mergedTokens, grid: prep.grid, resized: prep.resized } };
  }

  async dispose(): Promise<void> {
    await this.vision.release?.(); await this.text.release?.(); await this.table.close?.();
  }
}

export async function loadImage(spec: ImageSpec, o: ImageLoadOptions): Promise<ImageModel> {
  const status = o.onStatus ?? (() => {});
  const platform = detectPlatform();
  let device = o.device ?? "auto";
  if (device === "auto") device = platform === "node" ? "cpu" : (await webgpuStatus()).ok ? "webgpu" : "wasm";
  for (const [n, d] of [["vision", spec.vision.devices], ["text", spec.text.devices]] as const) if (!d.includes(device as any)) throw new Error(`image ${n} lists devices ${d.join(", ")}, not ${device}`);
  if ((platform === "node") !== (device === "cpu")) throw new Error(`image: device ${device} does not run on ${platform}`);
  const ort: any = o.modules?.ort ?? (platform === "node"
    ? await importOptional("onnxruntime-node", () => import(/* @vite-ignore */ "onnxruntime-node"), "ONNX Runtime in Node")
    : await importOptional("onnxruntime-web", () => import("onnxruntime-web/webgpu"), "ONNX Runtime in the browser"));
  const { files, repo, revision } = o, H = spec.text.model.hidden;
  status("loading tokenizer…");
  let tok = o.tokenizer;
  if (!tok) {
    const tjs = o.modules?.transformers ?? await importOptional("@huggingface/transformers", () => import("@huggingface/transformers"), "transformers.js (tokenizer)");
    tok = tokenizerFromJson(tjs, await files.json(repo, revision, spec.tokenizer.tokenizer), await files.json(repo, revision, spec.tokenizer.config));
  }
  const rowsBytes = await files.bytes(repo, revision, spec.readout.rows);
  const rows = f16ToF32(u16(rowsBytes));
  status("loading embeddings…");
  const ef = spec.embeddings;
  if (ef.format !== "f16") throw new Error("image: embeddings format q4-block128 is specified but not implemented in the runtime");
  let table: EmbeddingTable;
  if (platform === "node") table = await fileTable(await files.local(repo, revision, ef.file), H);
  else if ((ef.file.size ?? Infinity) <= 64 * 1024 * 1024) table = embeddingTableFromBytes(await files.bytes(repo, revision, ef.file), H);
  else table = httpRangeTable(await files.large(repo, revision, ef.file), ef.file.size!, H, o.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a)));
  const vdev = o.visionDevice ?? device;
  if (!spec.vision.devices.includes(vdev as any)) throw new Error(`image vision lists devices ${spec.vision.devices.join(", ")}, not ${vdev}`);
  const mk = async (g: { graph: any; data?: any[] }, extra: Record<string, any> = {}, dev: string = device) => {
    const data = g.data ?? [];
    if (platform === "node") {
      for (const d of data) await files.local(repo, revision, d);
      return ort.InferenceSession.create(await files.local(repo, revision, g.graph), { executionProviders: ["cpu"], graphOptimizationLevel: "all", logSeverityLevel: 3, ...extra });
    }
    const bytes = await files.bytes(repo, revision, g.graph);
    const externalData = await Promise.all(data.map(async (d: any) => ({ path: base(d.path), data: await files.large(repo, revision, d) })));
    return ort.InferenceSession.create(bytes, { executionProviders: [dev === "webgpu" ? { name: "webgpu", validationMode: "disabled" } : "wasm"], externalData, graphOptimizationLevel: "all", enableGraphCapture: false, logSeverityLevel: 3, ...extra });
  };
  status(`loading vision (${device})…`);
  const vision = await mk(spec.vision, {}, vdev);
  status(`loading text (${device})…`);
  const L = spec.text.model.layers;
  const loc: Record<string, string> = { [spec.text.output.hidden]: "cpu" };
  if (device === "webgpu") for (let l = 0; l < L; l++) for (const kv of ["Key", "Value"] as const) loc[spec.text.cache[`present${kv}` as "presentKey"].replace("%d", String(l))] = "gpu-buffer";
  const text = await mk(spec.text, device === "webgpu" ? { preferredOutputLocation: loc } : {});
  void embeddingTable;
  const posTable = spec.vision.posEmbeds ? f16ToF32(u16(await files.bytes(repo, revision, spec.vision.posEmbeds.table))) : null;
  return new ImageModel(spec, ort, vision, text, table, tok, rows, device, !!o.noCache, posTable, o.onPrefix);
}
