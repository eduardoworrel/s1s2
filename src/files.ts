// Pinned-file access with integrity checks.
//
// Policy (every mismatch throws):
//  - URLs are pinned: https://huggingface.co/<repo>/resolve/<revision>/<path>.
//  - Small files (tokenizer, configs, heads, litert manifest/graph, recovery): bytes are
//    sha256-hashed after download, in the browser and in Node.
//  - Large weights in the browser (handed to ONNX Runtime / WebLLM / LiteRT by URL): the HF API
//    paths-info at the pinned revision must report the manifest sha256 and size (metadata check;
//    the bytes themselves are not re-hashed in the browser). Non-HF hosts: pass
//    `allowUnverified: true` to accept large files without a check.
//  - Node: every file is downloaded once into the cache (<cacheDir>/<revision>/<repo>/<path>),
//    hashed while streaming; files from `localDirs` are hashed in full once and remembered in
//    <cacheDir>/.verified/ (keyed by path, size, mtime). Local files are only read.

import { fileUrl, type PinnedFile } from "./manifest.js";

export const SMALL_FILE_LIMIT = 300 * 1024 * 1024;

const nodeImport = (spec: string): Promise<any> => import(/* @vite-ignore */ spec);

export async function sha256Hex(bytes: Uint8Array | ArrayBuffer): Promise<string> {
  const subtle = (globalThis as any).crypto?.subtle;
  if (!subtle) throw new Error("sha256: WebCrypto (crypto.subtle) is not available in this context");
  const d = new Uint8Array(await subtle.digest("SHA-256", bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

export class IntegrityError extends Error {
  constructor(msg: string) { super(msg); this.name = "IntegrityError"; }
}

export interface FilesOptions {
  platform: "node" | "browser";
  /** Node: download cache (default ~/.cache/s1s2, or $S1S2_CACHE; the deprecated $S12_CACHE is still read). */
  cacheDir?: string;
  /** Node: repo id -> local directory mirroring the repo layout (read-only; hashed once; missing files are downloaded to the cache). */
  localDirs?: Record<string, string>;
  /** Self-hosting: repo id -> base URL replacing https://huggingface.co/<repo>/resolve/<revision>/. */
  baseUrls?: Record<string, string>;
  /** Accept large browser files from non-HF hosts without a sha check. */
  allowUnverified?: boolean;
  fetch?: typeof fetch;
  onStatus?: (msg: string) => void;
}

export class Files {
  private f: typeof fetch;
  private status: (m: string) => void;
  constructor(private o: FilesOptions) {
    this.f = o.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
    this.status = o.onStatus ?? (() => {});
  }

  url(repo: string, revision: string, p: PinnedFile): string {
    return fileUrl(repo, revision, p.path, this.o.baseUrls?.[repo]);
  }

  /** Bytes of a pinned file, sha256-verified. */
  async bytes(repo: string, revision: string, p: PinnedFile): Promise<Uint8Array> {
    if (this.o.platform === "node") {
      const fs = await nodeImport("node:fs");
      const b: Uint8Array = fs.readFileSync(await this.local(repo, revision, p));
      return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    }
    const url = this.url(repo, revision, p);
    const r = await this.f(url);
    if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
    const b = new Uint8Array(await r.arrayBuffer());
    await checkBytes(b, p, url);
    return b;
  }

  async json(repo: string, revision: string, p: PinnedFile): Promise<any> {
    return JSON.parse(new TextDecoder().decode(await this.bytes(repo, revision, p)));
  }

  /** Browser: URL of a large file after the paths-info check; Node: verified local path. */
  async large(repo: string, revision: string, p: PinnedFile): Promise<string> {
    if (this.o.platform === "node") return this.local(repo, revision, p);
    const url = this.url(repo, revision, p);
    if (/^https:\/\//.test(p.path) || this.o.baseUrls?.[repo]) {
      // Not on the Hub: fetch and hash when small enough (the browser cache serves the second fetch).
      if ((p.size ?? Infinity) <= SMALL_FILE_LIMIT) { await this.bytes(repo, revision, p); return url; }
      if (this.o.allowUnverified) { this.status(`unverified (allowUnverified): ${url}`); return url; }
      throw new IntegrityError(`${url}: large file not on the Hugging Face Hub (or size not pinned), so its sha256 cannot be checked without downloading it; pass allowUnverified: true to accept it`);
    }
    await this.pathsInfo(repo, revision, [p]);
    return url;
  }

  /** HF API metadata check: at `revision`, each path has the manifest sha256 (LFS oid) and size. */
  async pathsInfo(repo: string, revision: string, ps: PinnedFile[]): Promise<void> {
    const api = `https://huggingface.co/api/models/${repo}/paths-info/${revision}`;
    const r = await this.f(api, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ paths: ps.map((p) => p.path) }) });
    if (!r.ok) throw new IntegrityError(`${api}: HTTP ${r.status} (revision ${revision} missing?)`);
    const info: any[] = await r.json();
    for (const p of ps) {
      const it = info.find((x) => x.path === p.path);
      if (!it) throw new IntegrityError(`${repo}@${revision}: ${p.path} does not exist`);
      const sha = it.lfs?.oid;
      if (!sha) throw new IntegrityError(`${repo}@${revision}: ${p.path} is not an LFS file; its sha256 is not reported, download it as a small file instead`);
      if (sha !== p.sha256) throw new IntegrityError(`${repo}@${revision}: ${p.path} sha256 ${sha} != manifest ${p.sha256}`);
      if (p.size !== undefined && it.size !== p.size) throw new IntegrityError(`${repo}@${revision}: ${p.path} size ${it.size} != manifest ${p.size}`);
    }
  }

  async cacheDir(): Promise<string> {
    if (this.o.cacheDir) return this.o.cacheDir;
    const env = (globalThis as any).process?.env ?? {};
    if (env.S1S2_CACHE) return env.S1S2_CACHE;
    if (env.S12_CACHE) return env.S12_CACHE; // deprecated name
    const os = await nodeImport("node:os"), path = await nodeImport("node:path");
    return path.join(env.XDG_CACHE_HOME ?? path.join(os.homedir(), ".cache"), "s1s2");
  }

  /** Node: verified local path (local dir, cache, or download). */
  async local(repo: string, revision: string, p: PinnedFile): Promise<string> {
    const fs = await nodeImport("node:fs"), path = await nodeImport("node:path");
    const cache = await this.cacheDir();
    const localDir = this.o.localDirs?.[repo];
    if (localDir) {
      const file = path.resolve(localDir, p.path);
      // A file present in the local mirror is used (after the sha256 check); a missing one is downloaded to the cache.
      if (fs.existsSync(file)) { await verifyFileOnce(file, p, cache, this.status); return file; }
    }
    if (/^https:\/\//.test(p.path)) {
      const u = new URL(p.path);
      return this.download(p.path, path.join(cache, "external", u.hostname, ...u.pathname.split("/").filter(Boolean)), p);
    }
    return this.download(this.url(repo, revision, p), path.join(cache, revision, ...repo.split("/"), ...p.path.split("/")), p);
  }

  private async download(url: string, dest: string, p: PinnedFile): Promise<string> {
    const fs = await nodeImport("node:fs"), path = await nodeImport("node:path"), crypto = await nodeImport("node:crypto");
    if (fs.existsSync(dest)) {
      await verifyFileOnce(dest, p, await this.cacheDir(), this.status);
      return dest;
    }
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    const part = `${dest}.part-${(globalThis as any).process?.pid ?? 0}`;
    try {
      const r = await this.f(url);
      if (!r.ok || !r.body) throw new Error(`download ${url}: HTTP ${r.status}`);
      const total = Number(r.headers.get("content-length")) || p.size || 0;
      const h = crypto.createHash("sha256"), out = fs.createWriteStream(part);
      let got = 0, shown = -1;
      for await (const chunk of r.body as any as AsyncIterable<Uint8Array>) {
        h.update(chunk); got += chunk.length;
        if (!out.write(chunk)) await new Promise((res) => out.once("drain", res));
        const pct = total ? Math.floor((100 * got) / total / 10) * 10 : -1;
        if (pct !== shown && pct >= 0 && total > 50e6) { shown = pct; this.status(`downloading ${path.basename(dest)}: ${pct}%`); }
      }
      await new Promise<void>((res, rej) => out.end((e: any) => (e ? rej(e) : res())));
      const sha = h.digest("hex");
      if (sha !== p.sha256) throw new IntegrityError(`${url}: sha256 ${sha} != manifest ${p.sha256}`);
      if (p.size !== undefined && got !== p.size) throw new IntegrityError(`${url}: size ${got} != manifest ${p.size}`);
      fs.renameSync(part, dest);
      await markVerified(dest, p, await this.cacheDir());
    } catch (e) {
      try { fs.rmSync(part, { force: true }); } catch { /* ignore */ }
      throw e;
    }
    return dest;
  }
}

export async function checkBytes(b: Uint8Array, p: PinnedFile, where: string): Promise<void> {
  if (p.size !== undefined && b.byteLength !== p.size) throw new IntegrityError(`${where}: size ${b.byteLength} != manifest ${p.size}`);
  const sha = await sha256Hex(b);
  if (sha !== p.sha256) throw new IntegrityError(`${where}: sha256 ${sha} != manifest ${p.sha256}`);
}

async function markerPath(cache: string, p: PinnedFile) {
  const path = await nodeImport("node:path");
  return path.join(cache, ".verified", p.sha256);
}

async function markVerified(file: string, p: PinnedFile, cache: string) {
  const fs = await nodeImport("node:fs"), path = await nodeImport("node:path");
  const st = fs.statSync(file), mk = await markerPath(cache, p);
  fs.mkdirSync(path.dirname(mk), { recursive: true });
  const prev = fs.existsSync(mk) ? JSON.parse(fs.readFileSync(mk, "utf8")) : [];
  prev.push({ path: fs.realpathSync(file), size: st.size, mtimeMs: st.mtimeMs });
  fs.writeFileSync(mk, JSON.stringify(prev));
}

/** Full sha256 of a local file, once; later loads trust the marker while path/size/mtime match. */
export async function verifyFileOnce(file: string, p: PinnedFile, cache: string, status: (m: string) => void = () => {}): Promise<void> {
  const fs = await nodeImport("node:fs"), crypto = await nodeImport("node:crypto");
  const st = fs.statSync(file), real = fs.realpathSync(file), mk = await markerPath(cache, p);
  if (p.size !== undefined && st.size !== p.size) throw new IntegrityError(`${file}: size ${st.size} != manifest ${p.size}`);
  if (fs.existsSync(mk)) {
    const seen: any[] = JSON.parse(fs.readFileSync(mk, "utf8"));
    if (seen.some((x) => x.path === real && x.size === st.size && x.mtimeMs === st.mtimeMs)) return;
  }
  if (st.size > 50e6) status(`hashing ${file} (${(st.size / 1e9).toFixed(2)} GB, once)…`);
  const h = crypto.createHash("sha256");
  for await (const chunk of fs.createReadStream(file, { highWaterMark: 8 << 20 })) h.update(chunk);
  const sha = h.digest("hex");
  if (sha !== p.sha256) throw new IntegrityError(`${file}: sha256 ${sha} != manifest ${p.sha256}`);
  await markVerified(file, p, cache);
}
