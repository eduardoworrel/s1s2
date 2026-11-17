// Runtime/device planning (ported from ct0-edge runtime.ts; builds come from the manifest).

import type { RuntimeName, Manifest } from "./manifest.js";
import { RUNTIMES } from "./manifest.js";

export type Device = "webgpu" | "wasm" | "cpu" | "cuda";
export type DeviceChoice = Device | "auto";
export type Platform = "browser" | "node";
export interface Caps {
  platform: Platform; webgpu: boolean; cuda: boolean;
  /** Why WebGPU is not available (browser), for error messages and `S1S2.fallbackReason`. */
  webgpuReason?: string;
}
export interface Plan { build: string; runtime: RuntimeName; device: Device }

const DEVICES: readonly Device[] = ["webgpu", "wasm", "cpu", "cuda"];

export function detectPlatform(g: any = globalThis): Platform {
  const isNode = !!g.process?.versions?.node;
  const hasDom = typeof g.window !== "undefined" && typeof g.document !== "undefined";
  const isWorker = typeof g.WorkerGlobalScope !== "undefined" && typeof g.importScripts === "function";
  return isNode && !hasDom && !isWorker ? "node" : "browser";
}

export async function detectWebGPU(nav: any = (globalThis as any).navigator): Promise<boolean> {
  return (await webgpuStatus(nav)).ok;
}

/** WebGPU availability with the reason when absent: navigator.gpu missing, or requestAdapter() gave no adapter. */
export async function webgpuStatus(nav: any = (globalThis as any).navigator): Promise<{ ok: boolean; reason?: string }> {
  if (typeof nav?.gpu?.requestAdapter !== "function") return { ok: false, reason: "navigator.gpu is missing" };
  try { return (await nav.gpu.requestAdapter()) ? { ok: true } : { ok: false, reason: "navigator.gpu.requestAdapter() returned no adapter" }; }
  catch (e: any) { return { ok: false, reason: `navigator.gpu.requestAdapter() failed: ${e?.message ?? String(e)}` }; }
}

/** Why a genai build may or may not run on the browser CPU. */
export const GENAI_BROWSER_CPU_NOTE = 'A genai build runs on the browser CPU (onnxruntime-web WASM) only when its manifest lists device "wasm", i.e. the packager measured that path. Graphs with linear-attention ops cannot: onnxruntime-web has no WASM kernel for com.microsoft.GatedDeltaNet (Qwen3.5-family graphs such as S5-nano / CT0), so those builds need WebGPU.';

const noGpu = (caps: Caps) => `WebGPU is not available (${caps.webgpuReason ?? "no WebGPU adapter"})`;

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
      if (platform === "browser") { if (dev === "wasm") throw new Error('runtime "genai" on the browser CPU (wasm) needs a build whose manifest lists device "wasm". ' + GENAI_BROWSER_CPU_NOTE); return needGpu("webgpu"); }
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
    if (dev === "wasm" || !caps.webgpu) return [{ runtime: "genai", device: "wasm" }, { runtime: "decode-only", device: "wasm" }, { runtime: "transformers.js", device: "wasm" }];
    return [{ runtime: "genai", device: "webgpu" }, { runtime: "decode-only", device: "webgpu" }, { runtime: "transformers.js", device: "webgpu" }];
  }
  const nodeCpu = [{ runtime: "transformers.js" as const, device: "cpu" as const }, { runtime: "decode-only" as const, device: "cpu" as const }];
  if (dev === "cpu" || !caps.cuda) return nodeCpu;
  return [{ runtime: "genai", device: "cuda" }, { runtime: "transformers.js", device: "cuda" }, ...nodeCpu];
}

/** Candidate plans: an explicit build, an explicit runtime, or "auto" over the manifest builds.
 *  `cpuFallback: false`: with device "auto", never move to the CPU (WASM) when WebGPU is missing (throws instead). */
export function planBuilds(m: Manifest, o: { build?: string; runtime?: RuntimeName | "auto"; device?: DeviceChoice; cpuFallback?: boolean }, caps: Caps): Plan[] {
  const device = o.device ?? "auto";
  const builds = Object.entries(m.builds);
  const browser = caps.platform === "browser";
  const wantWasm = browser && (device === "wasm" || device === "cpu");
  const autoNoGpu = browser && device === "auto" && !caps.webgpu;
  if (autoNoGpu && o.cpuFallback === false)
    throw new Error(`${noGpu(caps)}, and cpuFallback is false: no automatic move to the browser CPU (WASM). Pass device "wasm" to ask for it.`);
  const listDevices = () => builds.map(([n, b]) => `${n} (${b.runtime}: ${b.devices.join(", ")})`).join("; ");
  if (o.build !== undefined) {
    const b = m.builds[o.build];
    if (!b) throw new Error(`unknown build "${o.build}"; manifest has ${Object.keys(m.builds).join(", ")}`);
    if (o.runtime && o.runtime !== "auto" && o.runtime !== b.runtime) throw new Error(`build "${o.build}" is runtime ${b.runtime}, not ${o.runtime}`);
    // Opt-in: a genai build whose manifest lists "cpu" runs on onnxruntime-node CPU (plain MatMulNBits/GQA graphs, e.g. Llama).
    if (b.runtime === "genai" && caps.platform === "node" && b.devices.includes("cpu") && (device === "cpu" || (device === "auto" && !caps.cuda)))
      return [{ build: o.build, runtime: "genai", device: "cpu" }];
    // Opt-in: a genai build whose manifest lists "wasm" runs on the browser CPU (onnxruntime-web WASM); "auto" falls back to it without WebGPU.
    if (b.runtime === "genai" && (wantWasm || autoNoGpu)) {
      if (b.devices.includes("wasm")) return [{ build: o.build, runtime: "genai", device: "wasm" }];
      throw new Error(`build "${o.build}" cannot run here: ${autoNoGpu ? noGpu(caps) + ", and " : ""}its manifest lists device ${b.devices.join(", ")}, not "wasm". ${GENAI_BROWSER_CPU_NOTE}`);
    }
    const d = deviceFor(b.runtime, device, caps);
    if (!b.devices.includes(d)) throw new Error(`build "${o.build}" does not list device ${d} (has ${b.devices.join(", ")})`);
    return [{ build: o.build, runtime: b.runtime, device: d }];
  }
  if (o.runtime && o.runtime !== "auto") {
    if (!RUNTIMES.includes(o.runtime)) throw new Error(`unknown runtime "${o.runtime}"`);
    if (o.runtime === "genai" && (wantWasm || autoNoGpu)) {
      const hit = builds.find(([, b]) => b.runtime === "genai" && b.devices.includes("wasm"));
      if (hit) return [{ build: hit[0], runtime: "genai", device: "wasm" }];
      throw new Error(`manifest has no genai build for device wasm${autoNoGpu ? ` (${noGpu(caps)})` : ""}; builds: ${listDevices()}. ${GENAI_BROWSER_CPU_NOTE}`);
    }
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
  if (!out.length) {
    const why = browser && (autoNoGpu || wantWasm) ? ` ${autoNoGpu ? noGpu(caps) + "; " : ""}no build lists device "wasm" (builds: ${listDevices()}). ${GENAI_BROWSER_CPU_NOTE}` : "";
    throw new Error(`no build in the manifest runs here (${caps.platform}, device ${device}).${why}`);
  }
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
