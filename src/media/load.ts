// loadMedia: read a media manifest (modalities ["audio"]; "image" is specified, runtime not built), check its pinned files and return the model.

import { Files, type FilesOptions } from "../files.js";
import { validateMediaManifest } from "./manifest.js";
import type { MediaManifest } from "./types.js";
import { detectPlatform } from "../runtime.js";
import { loadAudio, AudioModel } from "./audio.js";
import { loadImage, ImageModel } from "./image.js";

export interface MediaLoadOptions {
  /** Manifest object, or a URL / Node path to its JSON. */
  manifest: MediaManifest | Record<string, unknown> | string;
  /** Which media of a manifest that lists several. Default: its first. */
  modality?: "audio" | "image";
  device?: "auto" | "webgpu" | "wasm" | "cpu";
  /** Modules to use instead of importing them: onnxruntime. */
  modules?: { ort?: any; transformers?: any };
  cacheDir?: string;
  /** Node: repo id -> local directory mirroring that repo (sha256-checked once). */
  localDirs?: Record<string, string>;
  /** Browser: repo id -> base URL serving that repo's files. */
  baseUrls?: Record<string, string>;
  /** Accept files over 300 MB from non-Hub hosts without a browser-side hash (the demo server hashes them at start-up). */
  allowUnverified?: boolean;
  /** The decoder (language, term questions) of a manifest that has one: `"auto"` (default) loads it and goes on without it if that fails, `true` fails, `false` loads nothing. */
  decoder?: boolean | "auto";
  fetch?: typeof fetch;
  onStatus?: (msg: string) => void;
}

async function readMediaManifest(src: MediaLoadOptions["manifest"], platform: string, f: typeof fetch): Promise<MediaManifest> {
  if (typeof src !== "string") return validateMediaManifest(src);
  if (platform === "node" && !/^https?:/.test(src)) {
    const fs: any = await import(/* @vite-ignore */ "node:fs"), url: any = await import(/* @vite-ignore */ "node:url");
    return validateMediaManifest(JSON.parse(fs.readFileSync(src.startsWith("file:") ? url.fileURLToPath(src) : src, "utf8")));
  }
  const r = await f(src);
  if (!r.ok) throw new Error(`manifest ${src}: HTTP ${r.status}`);
  return validateMediaManifest(await r.json());
}

export async function loadMedia(o: MediaLoadOptions & { modality: "image"; tokenizer?: any }): Promise<ImageModel>;
export async function loadMedia(o: MediaLoadOptions): Promise<AudioModel>;
export async function loadMedia(o: MediaLoadOptions & { tokenizer?: any }): Promise<AudioModel | ImageModel> {
  const platform = detectPlatform();
  const f = o.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  const m = await readMediaManifest(o.manifest, platform, f);
  const which = o.modality ?? m.modalities.find((x) => x === "audio" || x === "image");
  if (which === "image") {
    if (!m.modalities.includes("image")) throw new Error(`loadMedia: this manifest takes ${m.modalities.join(", ")}, not image`);
    const files = new Files({ platform, cacheDir: o.cacheDir, localDirs: o.localDirs, baseUrls: o.baseUrls, allowUnverified: o.allowUnverified, fetch: o.fetch, onStatus: o.onStatus });
    return loadImage(m.image!, { device: o.device, modules: o.modules, files, repo: m.repo, revision: m.revision, onStatus: o.onStatus, tokenizer: (o as any).tokenizer, fetch: o.fetch });
  }
  if (which !== "audio") throw new Error("loadMedia: the manifest lists no audio modality");
  if (!m.modalities.includes(which)) throw new Error(`loadMedia: this manifest takes ${m.modalities.join(", ")}, not ${which}`);
  const fo: FilesOptions = { platform, cacheDir: o.cacheDir, localDirs: o.localDirs, baseUrls: o.baseUrls, allowUnverified: o.allowUnverified, fetch: o.fetch, onStatus: o.onStatus };
  const files = new Files(fo);
  return loadAudio(m.audio!, { device: o.device, modules: o.modules, files, repo: m.repo, revision: m.revision, onStatus: o.onStatus, decoder: o.decoder, fetch: o.fetch });
}
