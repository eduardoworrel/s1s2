// Load a model from its manifest: validate, pick a build, fetch + verify pinned files, build the
// tokenizer, the backend and the heads, then assert hidden size / vocabulary / marker tokens.

import { validateManifest, headFor, HEAD_ROLES, type Manifest, type HeadRole, type RuntimeName, type BuildSpec } from "./manifest.js";
import { Files, SMALL_FILE_LIMIT } from "./files.js";
import { tokenizerFromJson, type Tok } from "./tok.js";
import { assertManifestTokens, taskTokens } from "./encoding.js";
import { loadAnyHead, type AnyHead } from "./heads/index.js";
import { loadNorm, type NormWeights } from "./heads/norm.js";
import { canShareState, type Backend } from "./backend.js";
import { decide, generate, type Engine, type DecideOptions, type DecisionSession, type GenerateOpts, type GenerateResult } from "./decide.js";
import type { DecisionRequest } from "./questions.js";
import { detectPlatform, detectWebGPU, detectCuda, planBuilds, importOptional, type Caps, type Device, type DeviceChoice, type Plan } from "./runtime.js";
import { genaiFromSession, createGenai, type GenaiOptions } from "./backends/genai.js";
import { embeddingTableFromBytes, httpRangeTable, fileTable, type EmbeddingTable } from "./backends/embeddings.js";
import { createDecodeOnly } from "./backends/decodeOnly.js";
import { transformersJsFromModel } from "./backends/transformersjs.js";
import { createWebllm, type Recovery } from "./backends/webllm.js";
import { createLiteRt, embedFromF16Reader, httpRangeReader, LITERTJS_VERSION } from "./backends/litert.js";

export interface LoadOptions {
  /** Manifest object, or a URL / Node path to its JSON. */
  manifest: Manifest | Record<string, unknown> | string;
  /** Build name from the manifest; overrides runtime. */
  build?: string;
  runtime?: RuntimeName | "auto";
  device?: DeviceChoice;
  /** Modules to use instead of importing them. */
  modules?: { ort?: any; transformers?: any; webllm?: any; litert?: any };
  /** Node: download cache. */
  cacheDir?: string;
  /** Node: repo id -> local directory mirroring that repo (read-only; sha256-checked once). */
  localDirs?: Record<string, string>;
  /** Self-hosting: repo id -> base URL. */
  baseUrls?: Record<string, string>;
  allowUnverified?: boolean;
  /** Load the heads listed by the build (default true). */
  heads?: boolean;
  maxSafeTokens?: number;
  skipVersionCheck?: boolean;
  genai?: Pick<GenaiOptions, "chunk" | "verifyEp" | "epCache" | "maxCpuNodes">;
  decodeOnly?: { pastLen?: number; providers?: string[] };
  litert?: { wasmUrl?: string; precision?: "fp32" | "fp16"; quiet?: boolean };
  webllm?: { initProgressCallback?: (p: any) => void; appConfig?: any };
  fetch?: typeof fetch;
  onStatus?: (msg: string) => void;
}

const NODE_ORT = "onnxruntime-node", WEBLLM_PKG = "@mlc-ai/web-llm", LITERT_PKG = "@litertjs/core", TJS_PKG = "@huggingface/transformers";
const dynImport = (s: string): Promise<any> => import(/* @vite-ignore */ s);

async function readManifest(src: LoadOptions["manifest"], platform: string, f: typeof fetch): Promise<Manifest> {
  if (typeof src !== "string") return validateManifest(src);
  if (platform === "node" && !/^https?:/.test(src)) {
    const fs = await dynImport("node:fs"), url = await dynImport("node:url");
    return validateManifest(JSON.parse(fs.readFileSync(src.startsWith("file:") ? url.fileURLToPath(src) : src, "utf8")));
  }
  const r = await f(src);
  if (!r.ok) throw new Error(`manifest ${src}: HTTP ${r.status}`);
  return validateManifest(await r.json());
}

export class S12 {
  readonly manifest: Manifest;
  readonly build: string;
  readonly runtime: RuntimeName;
  readonly device: Device;
  /** Why "auto" did not use its first candidate. */
  fallbackReason?: string;
  private engine: Engine;
  private buildLoader?: (build: string) => Promise<S12>;
  private escalationBuilds = new Map<string, Promise<S12>>();

  private constructor(m: Manifest, plan: Plan, engine: Engine) {
    this.manifest = m; this.build = plan.build; this.runtime = plan.runtime; this.device = plan.device;
    this.engine = { ...engine, build: plan.build, resolveBuild: (b) => this.escalationBuild(b) };
  }

  /** How `to: { build }` escalation loads another build (default after S12.load: S12.load with the same options
   *  and that build). Each build is loaded once, on first use, and disposed with this instance. A second graph in
   *  memory is a real cost (an int8 graph is about twice the int4 one). */
  withBuildLoader(load: (build: string) => Promise<S12>): this { this.buildLoader = load; return this; }

  /** The instance used for escalation to `build` (loaded on first use). */
  escalationBuild(build: string): Promise<S12> {
    if (build === this.build) return Promise.reject(new Error(`margin policy: escalation target "${build}" is the build already answering`));
    if (!this.manifest.builds[build]) return Promise.reject(new Error(`margin policy: unknown build "${build}"`));
    let p = this.escalationBuilds.get(build);
    if (!p) {
      if (!this.buildLoader) return Promise.reject(new Error(`margin policy: no loader for build "${build}" (S12.fromBackend: call withBuildLoader)`));
      p = this.buildLoader(build);
      p.catch(() => this.escalationBuilds.delete(build));
      this.escalationBuilds.set(build, p);
    }
    return p;
  }

  /** Whether session.generate() continues from the decision prefill on this backend. */
  get sharedPrefill(): boolean { return canShareState(this.engine.backend); }
  get hiddenOutput(): string | null { return this.engine.backend.hiddenName; }
  get maxSafeTokens(): number { return this.engine.maxSafeTokens; }
  get backend(): Backend { return this.engine.backend; }
  get tokenizer(): Tok { return this.engine.tok; }
  /** Question types (and "evidence") with a loaded head. */
  get heads(): HeadRole[] { return Object.keys(this.engine.heads) as HeadRole[]; }

  countTokens(text: string): number { return this.engine.tok.encode(text).length; }

  /** System 1: typed decisions in one forward pass per question. Returns answers + a session for System 2. */
  decide(req: DecisionRequest, opts?: DecideOptions): Promise<DecisionSession> { return decide(this.engine, req, opts); }

  /** System 2: generation for one user turn. */
  generate(prompt: string, opts?: GenerateOpts): Promise<GenerateResult> { return generate(this.engine, prompt, opts); }

  async dispose(): Promise<void> {
    const extra = [...this.escalationBuilds.values()];
    this.escalationBuilds.clear();
    for (const p of extra) { try { await (await p).dispose(); } catch { /* failed loads have nothing to free */ } }
    await this.engine.backend.dispose();
  }

  /** Build an instance around an existing backend + tokenizer (tests, custom runtimes). */
  static fromBackend(manifest: Manifest | Record<string, unknown>, build: string, backend: Backend, tok: Tok, heads: Partial<Record<HeadRole, AnyHead>> = {}, device: Device = "cpu", maxSafeTokens?: number, norm?: NormWeights): S12 {
    const m = validateManifest(manifest);
    const b = m.builds[build];
    if (!b) throw new Error(`unknown build "${build}"`);
    assertManifestTokens(tok, m);
    for (const [t, h] of Object.entries(heads)) { const hs = h && ("cfg" in h ? h.cfg.hidden_size : h.hiddenSize); if (h && hs !== m.hiddenSize) throw new Error(`head for ${t}: hiddenSize ${hs} != manifest ${m.hiddenSize}`); }
    return new S12(m, { build, runtime: b.runtime, device }, { manifest: m, tok, backend, heads, norm, maxSafeTokens: maxSafeTokens ?? b.maxSafeTokens ?? m.defaults.maxSafeTokens ?? 2048 });
  }

  static async load(opts: LoadOptions): Promise<S12> {
    const status = opts.onStatus ?? (() => {});
    const platform = detectPlatform();
    const f = opts.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
    const m = await readManifest(opts.manifest, platform, f);
    const files = new Files({ platform, cacheDir: opts.cacheDir, localDirs: opts.localDirs, baseUrls: opts.baseUrls, allowUnverified: opts.allowUnverified, fetch: opts.fetch, onStatus: status });
    const mods = opts.modules ?? {};

    const webgpu = platform === "browser" ? await detectWebGPU() : false;
    let ort: any = mods.ort ?? null, cuda = false;
    const wantsOrt = (r?: string) => !r || r === "auto" || r === "genai" || r === "decode-only";
    const req = opts.build ? m.builds[opts.build]?.runtime : opts.runtime;
    if (platform === "node" && (wantsOrt(req) || opts.device === "cuda")) {
      try { ort ??= await importOptional(NODE_ORT, () => dynImport(NODE_ORT), "ONNX Runtime in Node"); cuda = detectCuda(ort); }
      catch (e) { if (req === "genai" || req === "decode-only") throw e; }
    }
    const caps: Caps = { platform, webgpu, cuda };
    const plans = planBuilds(m, { build: opts.build, runtime: opts.runtime, device: opts.device }, caps);

    status("loading tokenizer…");
    const tjs: any = mods.transformers ?? await importOptional(TJS_PKG, () => import("@huggingface/transformers"), "the tokenizer");
    const tk = m.tokenizer;
    const tok = tokenizerFromJson(tjs, await files.json(tk.repo, tk.revision, tk.files.tokenizer), await files.json(tk.repo, tk.revision, tk.files.config));
    assertManifestTokens(tok, m);

    const loadOrt = async () => {
      if (ort) return ort;
      ort = platform === "node" ? await importOptional(NODE_ORT, () => dynImport(NODE_ORT), "ONNX Runtime in Node")
        : await importOptional("onnxruntime-web", () => import("onnxruntime-web/webgpu"), "ONNX Runtime in the browser");
      return ort;
    };
    const eos = new Set(m.eos.map((t) => t.id));
    const basename = (p: string) => p.split("/").pop()!;

    const loaders: Record<RuntimeName, (b: BuildSpec, d: Device) => Promise<Backend>> = {
      genai: async (b, d) => {
        const o = await loadOrt();
        let kvHeadDim = b.kvHeadDim;
        if (b.files.genaiConfig) kvHeadDim = (await files.json(b.repo, b.revision, b.files.genaiConfig))?.model?.decoder?.head_size ?? kvHeadDim;
        // exclude_embeds builds: the token embedding table is a separate pinned file (role "embeddings", fp16 [vocab, hidden]).
        let embeddings: EmbeddingTable | undefined;
        const ef = b.files.embeddings;
        if (ef) {
          status("loading embeddings table…");
          if (ef.size === undefined) throw new Error(`build ${b.repo}: files.embeddings needs a pinned size`);
          if (platform === "node") embeddings = await fileTable(await files.local(b.repo, b.revision, ef), m.hiddenSize);
          else if (ef.size <= SMALL_FILE_LIMIT) embeddings = embeddingTableFromBytes(await files.bytes(b.repo, b.revision, ef), m.hiddenSize);
          else embeddings = httpRangeTable(await files.large(b.repo, b.revision, ef), ef.size, m.hiddenSize, f);
        }
        const gopts: GenaiOptions = { ...opts.genai, eos, hiddenOutput: b.hiddenOutput, kvHeadDim, ort: o, skipVersionCheck: opts.skipVersionCheck, embeddings };
        status(`loading genai graph (${d})…`);
        try {
          if (platform === "browser") {
            const graph = await files.bytes(b.repo, b.revision, b.files.model);
            const dataUrl = await files.large(b.repo, b.revision, b.files.data);
            return await createGenai(graph, dataUrl, basename(b.files.data.path), { ...gopts, modelKey: b.files.model.sha256 });
          }
          await files.local(b.repo, b.revision, b.files.data);
          nodeFloat16Workaround(o);
          const S = await o.InferenceSession.create(await files.local(b.repo, b.revision, b.files.model), { executionProviders: [d], graphOptimizationLevel: "all", logSeverityLevel: 3 });
          try { return genaiFromSession(S, o, gopts); } catch (e) { try { await S.release?.(); } catch { /* ignore */ } throw e; }
        } catch (e) { await embeddings?.close?.(); throw e; }
      },
      "decode-only": async (b, d) => {
        const o = await loadOrt();
        const providers = opts.decodeOnly?.providers ?? [d];
        status(`loading decode-only graph (${providers.join(",")})…`);
        let S: any;
        if (platform === "node") {
          await files.local(b.repo, b.revision, b.files.data);
          S = await o.InferenceSession.create(await files.local(b.repo, b.revision, b.files.model), { executionProviders: providers, logSeverityLevel: 3 });
        } else {
          const graph = await files.bytes(b.repo, b.revision, b.files.model);
          S = await o.InferenceSession.create(graph, { executionProviders: providers, externalData: [{ path: basename(b.files.data.path), data: await files.large(b.repo, b.revision, b.files.data) }] });
        }
        try { return await createDecodeOnly(S, { ort: o, eos, hiddenOutput: b.hiddenOutput, pastLen: opts.decodeOnly?.pastLen, skipVersionCheck: platform === "node" || opts.skipVersionCheck }); }
        catch (e) { try { await S.release?.(); } catch { /* ignore */ } throw e; }
      },
      "transformers.js": async (b, d) => {
        status(`loading transformers.js graph (${b.dtype ?? "default dtype"}, ${d})…`);
        const env = tjs.env;
        let modelId = b.repo;
        if (platform === "node") {
          const path = await dynImport("node:path");
          const locals = await Promise.all(Object.values(b.files).map((p) => files.local(b.repo, b.revision, p)));
          const modelPath = locals[Object.keys(b.files).indexOf("model")];
          const root = path.resolve(path.dirname(modelPath), path.relative(path.dirname(b.files.model.path), "."));
          env.allowRemoteModels = false; env.allowLocalModels = true;
          env.localModelPath = path.dirname(root) + path.sep; modelId = path.basename(root);
        } else {
          for (const p of Object.values(b.files)) { if ((p.size ?? Infinity) <= 1e6) await files.bytes(b.repo, b.revision, p); else await files.large(b.repo, b.revision, p); }
          env.allowRemoteModels = true;
        }
        const model = await tjs.AutoModelForCausalLM.from_pretrained(modelId, { dtype: b.dtype, device: d, revision: platform === "node" ? undefined : b.revision });
        return transformersJsFromModel(model, tjs.Tensor, { eos, hiddenOutput: b.hiddenOutput });
      },
      webllm: async (b) => {
        const webllm = mods.webllm ?? await importOptional(WEBLLM_PKG, () => dynImport(WEBLLM_PKG), 'runtime "webllm"');
        if (!b.modelId) throw new Error(`build ${b.repo}: webllm needs modelId`);
        await files.bytes(b.repo, b.revision, b.files.config);
        let recovery: Recovery | undefined;
        if (b.files.recoveryRows && b.files.recoveryP) {
          status("loading WebLLM hidden-state recovery…");
          const rows = await files.bytes(b.repo, b.revision, b.files.recoveryRows), P = await files.bytes(b.repo, b.revision, b.files.recoveryP);
          const ab = (u: Uint8Array) => u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength);
          recovery = { rows: new Int32Array(ab(rows)), P: b.files.recoveryP.path.endsWith(".f16") ? new Uint16Array(ab(P)) : new Float32Array(ab(P)) };
        }
        const libUrl = b.files.lib ? await files.large(b.repo, b.revision, b.files.lib) : undefined;
        const modelUrl = (opts.baseUrls?.[b.repo] ?? `https://huggingface.co/${b.repo}/resolve/${b.revision}/`);
        const appConfig = opts.webllm?.appConfig ?? { model_list: [{ model: modelUrl, model_id: b.modelId, model_lib: libUrl }] };
        status("loading WebLLM build…");
        return createWebllm(webllm, { modelId: b.modelId, modelUrl, libUrl, appConfig, tok, recovery, eos, initProgressCallback: opts.webllm?.initProgressCallback });
      },
      litert: async (b, d) => {
        const lrt = mods.litert ?? await importOptional(LITERT_PKG, () => dynImport(LITERT_PKG), 'runtime "litert"');
        status(`loading LiteRT.js ${LITERTJS_VERSION} runtime…`);
        try { await lrt.loadLiteRt(opts.litert?.wasmUrl ?? `https://cdn.jsdelivr.net/npm/@litertjs/core@${LITERTJS_VERSION}/wasm/`, { jspi: !!(globalThis as any).WebAssembly?.Suspending }); }
        catch (e: any) { if (!/already loading|already loaded/i.test(String(e?.message ?? e))) throw e; }
        const lman = await files.json(b.repo, b.revision, b.files.manifest);
        if (lman.hidden !== m.hiddenSize) throw new Error(`litert bundle hidden ${lman.hidden} != manifest hiddenSize ${m.hiddenSize}`);
        const graph = await files.bytes(b.repo, b.revision, b.files.graph);
        const weightsUrl = await files.large(b.repo, b.revision, b.files.weights);
        const embedUrl = await files.large(b.repo, b.revision, b.files.embed);
        status("streaming LiteRT weights into WebGPU…");
        return createLiteRt(lrt, graph, lman, embedFromF16Reader(httpRangeReader(embedUrl), lman.hidden), {
          eos, hiddenOutput: b.hiddenOutput, accelerator: d === "webgpu" ? "webgpu" : "wasm", gpuPrecision: opts.litert?.precision, quiet: opts.litert?.quiet,
          weights: lman.externalized === false ? undefined : () => {
            let body: ReadableStreamDefaultReader<Uint8Array> | null = null;
            return new ReadableStream<Uint8Array>({
              async pull(c) { body ??= (await f(weightsUrl)).body!.getReader(); const { done, value } = await body.read(); if (done) c.close(); else c.enqueue(value); },
              cancel() { body?.cancel(); },
            });
          },
        });
      },
    };

    const reasons: string[] = [];
    for (const [i, plan] of plans.entries()) {
      if (i > 0) status(`${reasons.at(-1)} -> ${plan.build} (${plan.device})`);
      const b = m.builds[plan.build];
      let backend: Backend | null = null;
      try {
        if (b.attention === "noncausal") throw new Error(`build ${plan.build}: its heads need a non-causal decision pass (attention "noncausal", spec 7); s1s2 0.1 runtimes are causal`);
        backend = await loaders[plan.runtime](b, plan.device);
        const heads = opts.heads === false ? {} : await loadHeads(m, b, files, status);
        const norm = opts.heads === false || !b.norm ? undefined : await loadBuildNorm(m, b, files);
        await assertBackend(m, b, backend);
        const s = new S12(m, plan, { manifest: m, tok, backend, heads, norm, maxSafeTokens: opts.maxSafeTokens ?? b.maxSafeTokens ?? m.defaults.maxSafeTokens ?? 2048 });
        if (reasons.length) s.fallbackReason = reasons.join("; ");
        s.withBuildLoader((build) => S12.load({ ...opts, build, runtime: undefined }));
        status(`ready: ${plan.build} (${plan.runtime}, ${plan.device})`);
        return s;
      } catch (e: any) {
        if (backend) { try { await backend.dispose(); } catch { /* ignore */ } }
        if (plans.length === 1 || e?.name === "IntegrityError" || e?.name === "ManifestError") throw e;
        reasons.push(`${plan.build} (${plan.device}) unavailable: ${e?.message ?? String(e)}`);
        if (i === plans.length - 1) { const err = new Error(`no build loaded. ${reasons.join("; ")}`); (err as any).cause = e; throw err; }
      }
    }
    throw new Error("no build candidate");
  }
}

/**
 * onnxruntime-node 1.30 + Node >= 24 (native Float16Array): onnxruntime-common stores "float16" tensor data as a
 * Float16Array, which the native binding reads as 0 bytes ("not enough space: expected N, got 0") as soon as a
 * non-empty fp16 tensor (the KV cache) is fed back. The common type map is decided once, at the first tensor; hide
 * Float16Array while that happens so fp16 data stays in Uint16Array, then restore the global. Throws if the map
 * was already fixed with Float16Array (a tensor was created earlier in this process).
 */
export function nodeFloat16Workaround(ort: any): void {
  const g = globalThis as any;
  const F16 = g.Float16Array;
  if (typeof F16 === "undefined") return;
  let t: any;
  try { delete g.Float16Array; t = new ort.Tensor("float16", new Uint16Array(1), [1]); }
  finally { g.Float16Array = F16; }
  if (!(t.data instanceof Uint16Array))
    throw new Error("onnxruntime-node: float16 tensors already use Float16Array in this process (the native binding reads them as empty). Load the model before creating any ort tensor, or run with globalThis.Float16Array removed.");
}

async function loadHeads(m: Manifest, b: BuildSpec, files: Files, status: (s: string) => void): Promise<Partial<Record<HeadRole, AnyHead>>> {
  const out: Partial<Record<HeadRole, AnyHead>> = {};
  const byName = new Map<string, AnyHead>();
  for (const t of HEAD_ROLES) {
    const h = headFor(b, t);
    if (!h) continue;
    const seen = byName.get(h.name);
    if (seen) { out[t] = seen; continue; }
    status(`loading head ${h.name}…`);
    const bytes = await files.bytes(b.repo, b.revision, h.spec.file);
    const w = loadAnyHead(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer, h.spec);
    const hs = "cfg" in w ? w.cfg.hidden_size : w.hiddenSize;
    if (hs !== m.hiddenSize) throw new Error(`head ${h.spec.file.path}: hiddenSize ${hs} != manifest hiddenSize ${m.hiddenSize}`);
    out[t] = w; byName.set(h.name, w);
  }
  return out;
}

async function loadBuildNorm(m: Manifest, b: BuildSpec, files: Files): Promise<NormWeights> {
  const bytes = await files.bytes(b.repo, b.revision, b.norm!.file);
  const nw = loadNorm(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer, b.norm!);
  if (nw.hiddenSize !== m.hiddenSize) throw new Error(`norm ${b.norm!.file.path}: hiddenSize ${nw.hiddenSize} != manifest hiddenSize ${m.hiddenSize}`);
  return nw;
}

/** One warmup forward: hidden width == manifest hiddenSize, vocabulary covers every manifest token. */
export async function assertBackend(m: Manifest, b: BuildSpec, be: Backend): Promise<void> {
  const tok = m.warmup?.id ?? m.eos[0].id;
  const r = await be.prefill([tok], { positions: be.hiddenName ? [0] : [] });
  if (be.hiddenName) {
    const h = r.hiddens[0];
    if (!h || h.length !== m.hiddenSize) throw new Error(`build ${b.repo}@${b.revision}: hidden output ${be.hiddenName} has width ${h?.length}, manifest hiddenSize is ${m.hiddenSize}`);
  }
  const ids = [m.logits.predicate.yes.id, m.logits.predicate.no.id, ...m.eos.map((t) => t.id), ...(m.logits.score.tokens ?? []).map((t) => t.id), ...taskTokens(m).map((t) => t.id)];
  const mx = Math.max(...ids);
  if (r.lastLogits.length <= mx) throw new Error(`build ${b.repo}@${b.revision}: vocabulary ${r.lastLogits.length} does not cover manifest token ${mx}`);
}
