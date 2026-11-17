// Runtime/device planning (ported from ct0-edge runtime.ts; builds come from the manifest).

import type { RuntimeName, Manifest } from "./manifest.js";
import { RUNTIMES } from "./manifest.js";

export type Device = "webgpu" | "wasm" | "cpu" | "cuda";
export type DeviceChoice = Device | "auto";
export type Platform = "browser" | "node";
export interface Caps { platform: Platform; webgpu: boolean; cuda: boolean }
export interface Plan { build: string; runtime: RuntimeName; device: Device }

const DEVICES: readonly Device[] = ["webgpu", "wasm", "cpu", "cuda"];

export function detectPlatform(g: any = globalThis): Platform {
  const isNode = !!g.process?.versions?.node;
  const hasDom = typeof g.window !== "undefined" && typeof g.document !== "undefined";
  const isWorker = typeof g.WorkerGlobalScope !== "undefined" && typeof g.importScripts === "function";
  return isNode && !hasDom && !isWorker ? "node" : "browser";
}

export async function detectWebGPU(nav: any = (globalThis as any).navigator): Promise<boolean> {
  try { return !!(await nav?.gpu?.requestAdapter?.()); } catch { return false; }
}

export function detectCuda(ortNode: any): boolean {
  try { return !!ortNode?.listSupportedBackends?.().some((b: any) => b?.name === "cuda"); } catch { return false; }
}

export const GENAI_NODE_CPU_ERROR = 'runtime "genai" does not run on onnxruntime-node CPU for this build (CT0\'s int4 GenAI graph has no CPU kernels for its linear-attention ops); in Node use "decode-only" or "transformers.js", genai on CUDA, or a build whose manifest lists device "cpu".';

function checkDevice(platform: Platform, device: Device): void {
  if (!DEVICES.includes(device)) throw new Error(`unknown device "${device}". Use "auto", ${DEVICES.map((d) => `"${d}"`).join(", ")}.`);
  if (platform === "browser" && device === "cuda") throw new Error('device "cuda" is Node-only.');
  if (platform === "node" && (device === "webgpu" || device === "wasm")) throw new Error(`device "${device}" is browser-only.`);
}

/** Device for a runtime on this platform (throws with the reason when impossible). */
export function deviceFor(runtime: RuntimeName, want: DeviceChoice, caps: Caps): Device {
  const { platform } = caps;
  const dev: DeviceChoice = platform === "browser" && want === "cpu" ? "wasm" : want;
  if (dev !== "auto") checkDevice(platform, dev);
  const needGpu = (d: Device) => { if (d === "webgpu" && !caps.webgpu) throw new Error(`runtime "${runtime}" on WebGPU, but WebGPU is not available.`); return d; };
  const needCuda = (d: Device) => { if (d === "cuda" && !caps.cuda) throw new Error(`runtime "${runtime}" on CUDA, but onnxruntime-node lists no CUDA provider.`); return d; };
  switch (runtime) {
    case "genai":
      if (platform === "browser") { if (dev === "wasm") throw new Error('runtime "genai" needs WebGPU in the browser.'); return needGpu("webgpu"); }
      if (dev === "cpu") throw new Error(GENAI_NODE_CPU_ERROR);
      if (!caps.cuda) throw new Error(GENAI_NODE_CPU_ERROR + " No CUDA provider was found.");
      return "cuda";
    case "webllm":
    case "litert":
      if (platform === "node") throw new Error(`runtime "${runtime}" runs only in the browser (WebGPU).`);
      if (dev === "wasm") throw new Error(`runtime "${runtime}" needs WebGPU.`);
      return needGpu("webgpu");
    case "decode-only":
    case "transformers.js":
      if (dev !== "auto") return platform === "browser" ? needGpu(dev) : needCuda(dev);
      return platform === "browser" ? (caps.webgpu ? "webgpu" : "wasm") : "cpu";
  }
}

/** Ordered (runtime, device) candidates for "auto" (same order as ct0-edge). */
export function autoOrder(device: DeviceChoice, caps: Caps): { runtime: RuntimeName; device: Device }[] {
  const { platform } = caps;
  const dev: DeviceChoice = platform === "browser" && device === "cpu" ? "wasm" : device;
  if (dev !== "auto") checkDevice(platform, dev);
  if (platform === "browser") {
    if (dev === "wasm" || !caps.webgpu) return [{ runtime: "decode-only", device: "wasm" }, { runtime: "transformers.js", device: "wasm" }];
    return [{ runtime: "genai", device: "webgpu" }, { runtime: "decode-only", device: "webgpu" }, { runtime: "transformers.js", device: "webgpu" }];
  }
  const nodeCpu = [{ runtime: "transformers.js" as const, device: "cpu" as const }, { runtime: "decode-only" as const, device: "cpu" as const }];
  if (dev === "cpu" || !caps.cuda) return nodeCpu;
  return [{ runtime: "genai", device: "cuda" }, { runtime: "transformers.js", device: "cuda" }, ...nodeCpu];
}

/** Candidate plans: an explicit build, an explicit runtime, or "auto" over the manifest builds. */
export function planBuilds(m: Manifest, o: { build?: string; runtime?: RuntimeName | "auto"; device?: DeviceChoice }, caps: Caps): Plan[] {
  const device = o.device ?? "auto";
  const builds = Object.entries(m.builds);
  if (o.build !== undefined) {
    const b = m.builds[o.build];
    if (!b) throw new Error(`unknown build "${o.build}"; manifest has ${Object.keys(m.builds).join(", ")}`);
    if (o.runtime && o.runtime !== "auto" && o.runtime !== b.runtime) throw new Error(`build "${o.build}" is runtime ${b.runtime}, not ${o.runtime}`);
    // Opt-in: a genai build whose manifest lists "cpu" runs on onnxruntime-node CPU (plain MatMulNBits/GQA graphs, e.g. Llama).
    if (b.runtime === "genai" && caps.platform === "node" && b.devices.includes("cpu") && (device === "cpu" || (device === "auto" && !caps.cuda)))
      return [{ build: o.build, runtime: "genai", device: "cpu" }];
    const d = deviceFor(b.runtime, device, caps);
    if (!b.devices.includes(d)) throw new Error(`build "${o.build}" does not list device ${d} (has ${b.devices.join(", ")})`);
    return [{ build: o.build, runtime: b.runtime, device: d }];
  }
  if (o.runtime && o.runtime !== "auto") {
    if (!RUNTIMES.includes(o.runtime)) throw new Error(`unknown runtime "${o.runtime}"`);
    const d = deviceFor(o.runtime, device, caps);
    const hit = builds.find(([, b]) => b.runtime === o.runtime && b.devices.includes(d));
    if (!hit) throw new Error(`manifest has no ${o.runtime} build for device ${d}`);
    return [{ build: hit[0], runtime: o.runtime, device: d }];
  }
  const out: Plan[] = [];
  for (const c of autoOrder(device, caps)) {
    const hit = builds.find(([, b]) => b.runtime === c.runtime && b.devices.includes(c.device));
    if (hit) out.push({ build: hit[0], ...c });
  }
  if (!out.length) throw new Error(`no build in the manifest runs here (${caps.platform}, device ${device})`);
  return out;
}

export async function importOptional<T = any>(pkg: string, importer: () => Promise<T>, why: string): Promise<T> {
  try { return await importer(); }
  catch (e: any) {
    const msg = String(e?.message ?? e);
    const notFound = e?.code === "ERR_MODULE_NOT_FOUND" || e?.code === "MODULE_NOT_FOUND" || /Cannot find (package|module)|Failed to resolve module specifier|Failed to fetch dynamically imported module/.test(msg);
    if (!notFound) throw e;
    const err = new Error(`${why} needs the package "${pkg}", which is not installed: npm i ${pkg}`);
    (err as any).cause = e;
    throw err;
  }
}
