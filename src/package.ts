// S1S2 package validator (Node only). See spec/s1s2-package.md.
//
// Two layers per manifest: the JSON Schema (spec/s1s2-manifest.schema.json, structure) and validateManifest
// (src/manifest.ts, normative cross-field rules). Package mode adds file checks: every pinned file present under the
// package root is sha256- and size-checked, head.bin files are parsed with loadHead and their hiddenSize compared with
// the manifest's, genai_config.json hidden/head sizes are compared with the manifest. A pinned file that is not present
// is reported as "not present" (warning), or as an error with `strict`.

import { ForecastReader } from "./forecast.js";
import { validateManifest, ManifestError, type Manifest, type PinnedFile } from "./manifest.js";
import { validateMediaManifest, mediaProblems } from "./media/manifest.js";
import type { MediaManifest } from "./media/types.js";
import { headScaleSource } from "./readout.js";
import { loadAnyHead } from "./heads/index.js";
import { loadNorm } from "./heads/norm.js";

const nodeImport = (spec: string): Promise<any> => import(/* @vite-ignore */ spec);

export const PACKAGE_SPEC_VERSION = "0.2-draft";

export interface ValidateOptions {
  /** Missing pinned files are errors (default: warnings). */
  strict?: boolean;
  /** repo id -> directory holding that repo's files (default: the manifest's directory, for every repo). */
  roots?: Record<string, string>;
  /** Skip file checks (manifest only). */
  manifestOnly?: boolean;
  /** Parsed JSON Schema (default: spec/s1s2-manifest.schema.json next to dist/). */
  schema?: object;
}

export interface ManifestReport {
  file: string;
  id?: string;
  ok: boolean;
  errors: string[];
  warnings: string[];
  /** Files whose sha256 (and size) were checked. */
  checked: string[];
}

let ajvCache: { schema: object; validate: any } | null = null;

async function schemaValidator(schema?: object) {
  if (ajvCache && (!schema || ajvCache.schema === schema)) return ajvCache.validate;
  const s = schema ?? (await defaultSchema());
  const mod = await nodeImport("ajv/dist/2020.js");
  const Ajv = mod.default?.default ?? mod.default ?? mod;
  const ajv = new Ajv({ allErrors: true, strict: false });
  const validate = ajv.compile(s);
  ajvCache = { schema: s, validate };
  return validate;
}

export async function defaultSchema(): Promise<object> {
  const fs = await nodeImport("node:fs"), url = await nodeImport("node:url"), path = await nodeImport("node:path");
  const here = path.dirname(url.fileURLToPath(import.meta.url));
  for (const c of [path.join(here, "..", "spec", "s1s2-manifest.schema.json"), path.join(here, "..", "..", "spec", "s1s2-manifest.schema.json")])
    if (fs.existsSync(c)) return JSON.parse(fs.readFileSync(c, "utf8"));
  throw new Error("s1s2: spec/s1s2-manifest.schema.json not found next to the library");
}

/** Schema errors of a parsed manifest, as "path: message" strings. */
export async function schemaProblems(m: unknown, schema?: object): Promise<string[]> {
  const validate = await schemaValidator(schema);
  if (validate(m)) return [];
  const out = new Set<string>();
  for (const e of validate.errors ?? []) {
    // if/then/else and oneOf wrappers repeat the leaf error; keep the leaves.
    if (["if", "oneOf", "anyOf", "allOf"].includes(e.keyword) && (validate.errors ?? []).length > 1) continue;
    const at = (e.instancePath || "/").replace(/^\//, "").replace(/\//g, ".") || "(root)";
    const extra = e.keyword === "additionalProperties" ? ` (${e.params.additionalProperty})` : e.keyword === "enum" ? ` (${e.params.allowedValues.join(", ")})` : "";
    out.add(`schema: ${at}: ${e.message}${extra}`);
  }
  return [...out];
}

async function sha256File(file: string): Promise<string> {
  const fs = await nodeImport("node:fs"), crypto = await nodeImport("node:crypto");
  const h = crypto.createHash("sha256");
  for await (const chunk of fs.createReadStream(file, { highWaterMark: 8 << 20 })) h.update(chunk);
  return h.digest("hex");
}

/** Validate one manifest file (and, unless manifestOnly, the package files next to it). */
export async function validateManifestFile(file: string, o: ValidateOptions = {}): Promise<ManifestReport> {
  const fs = await nodeImport("node:fs"), path = await nodeImport("node:path");
  const r: ManifestReport = { file, ok: false, errors: [], warnings: [], checked: [] };
  let m: any;
  try { m = JSON.parse(fs.readFileSync(file, "utf8")); } catch (e: any) { r.errors.push(`json: ${e?.message ?? e}`); return r; }
  r.id = typeof m?.id === "string" ? m.id : undefined;
  r.errors.push(...(await schemaProblems(m, o.schema)));
  let man: Manifest | null = null, media: MediaManifest | null = null;
  const isMedia = Array.isArray(m?.modalities) && !m.modalities.includes("text");
  try { if (isMedia) media = validateMediaManifest(m); else man = validateManifest(m); }
  catch (e) { if (e instanceof ManifestError) r.errors.push(...e.problems.map((x) => `manifest: ${x}`)); else throw e; }
  if (man && (m.modalities !== undefined || m.audio !== undefined)) r.errors.push(...mediaProblems(m, false).map((x) => `manifest: ${x}`));   // the media blocks of a text + audio manifest
  if (man && !o.manifestOnly) await checkFiles(man, path.dirname(path.resolve(file)), o, r);
  if (media && !o.manifestOnly) await checkMediaFiles(media, path.dirname(path.resolve(file)), o, r);
  r.ok = r.errors.length === 0;
  return r;
}

/** Media packages: every pinned file present under the root is size- and sha256-checked (missing ones are warnings, errors with strict). */
async function checkMediaFiles(m: MediaManifest, dir: string, o: ValidateOptions, r: ManifestReport): Promise<void> {
  const fs = await nodeImport("node:fs"), path = await nodeImport("node:path");
  const pins: [string, PinnedFile][] = [];
  const a = m.audio;
  if (a) {
    pins.push(["audio.tower.file", a.tower.file]);
    for (const [n, rd] of Object.entries(a.readers)) {
      if (rd.kind === "speech-head") { pins.push([`audio.readers.${n}.graph`, rd.graph], [`audio.readers.${n}.question`, rd.question]); }
      else if (rd.kind === "decoder-language" || rd.kind === "decoder-term") continue;
      else pins.push([`audio.readers.${n}.weights`, rd.weights]);
    }
    if (a.decoder) {
      for (const k of ["graph", "data", "embeddings", "tokens"] as const) pins.push([`audio.decoder.${k}`, a.decoder[k]]);
      if (a.decoder.cpuBuild) for (const k of ["graph", "data"] as const) pins.push([`audio.decoder.cpuBuild.${k}`, a.decoder.cpuBuild[k]]);
    }
  }
  const root = o.roots?.[m.repo] ?? dir, seen = new Set<string>();
  for (const [what, p] of pins) {
    if (/^https:\/\//.test(p.path)) { r.warnings.push(`${what}: ${p.path} is a URL (not checked locally)`); continue; }
    const f = path.resolve(root, p.path), key = `${f}\0${p.sha256}`;
    if (seen.has(key)) continue;
    seen.add(key);
    if (!fs.existsSync(f)) { (o.strict ? r.errors : r.warnings).push(`${what}: ${m.repo}/${p.path} not present${o.strict ? "" : " (not checked)"}`); continue; }
    const st = fs.statSync(f);
    if (p.size !== undefined && st.size !== p.size) { r.errors.push(`${what}: ${p.path} size ${st.size} != manifest ${p.size}`); continue; }
    const sha = await sha256File(f);
    if (sha !== p.sha256) r.errors.push(`${what}: ${p.path} sha256 ${sha} != manifest ${p.sha256}`); else r.checked.push(path.relative(dir, f) || p.path);
  }
}

async function checkFiles(m: Manifest, dir: string, o: ValidateOptions, r: ManifestReport): Promise<void> {
  const fs = await nodeImport("node:fs"), path = await nodeImport("node:path");
  const seen = new Map<string, boolean>();
  const root = (repo: string) => o.roots?.[repo] ?? dir;
  /** Check one pinned file; returns the local path when present and matching. */
  const check = async (repo: string, p: PinnedFile, what: string): Promise<string | null> => {
    if (/^https:\/\//.test(p.path)) { r.warnings.push(`${what}: ${p.path} is a URL (not checked locally)`); return null; }
    const f = path.resolve(root(repo), p.path);
    const key = `${f}\0${p.sha256}\0${p.size ?? ""}`;
    if (seen.has(key)) return seen.get(key) ? f : null;
    if (!fs.existsSync(f)) {
      (o.strict ? r.errors : r.warnings).push(`${what}: ${repo}/${p.path} not present${o.strict ? "" : " (not checked)"}`);
      seen.set(key, false);
      return null;
    }
    const st = fs.statSync(f);
    let good = true;
    if (p.size !== undefined && st.size !== p.size) { r.errors.push(`${what}: ${p.path} size ${st.size} != manifest ${p.size}`); good = false; }
    else {
      const sha = await sha256File(f);
      if (sha !== p.sha256) { r.errors.push(`${what}: ${p.path} sha256 ${sha} != manifest ${p.sha256}`); good = false; }
      else r.checked.push(path.relative(dir, f) || p.path);
    }
    seen.set(key, good);
    return good ? f : null;
  };
  await check(m.tokenizer.repo, m.tokenizer.files.tokenizer, "tokenizer.files.tokenizer");
  await check(m.tokenizer.repo, m.tokenizer.files.config, "tokenizer.files.config");
  for (const [bn, b] of Object.entries(m.builds)) {
    for (const [role, p] of Object.entries(b.files)) {
      const f = await check(b.repo, p, `builds.${bn}.files.${role}`);
      if (f && role === "genaiConfig") {
        try {
          const g = JSON.parse(fs.readFileSync(f, "utf8"))?.model?.decoder ?? {};
          if (g.hidden_size !== undefined && g.hidden_size !== m.hiddenSize) r.errors.push(`builds.${bn}: genai_config hidden_size ${g.hidden_size} != manifest hiddenSize ${m.hiddenSize}`);
          if (b.kvHeadDim !== undefined && g.head_size !== undefined && g.head_size !== b.kvHeadDim) r.errors.push(`builds.${bn}: genai_config head_size ${g.head_size} != kvHeadDim ${b.kvHeadDim}`);
          if (g.filename !== undefined && b.files.model && path.basename(b.files.model.path) !== g.filename) r.errors.push(`builds.${bn}: genai_config decoder.filename ${g.filename} != files.model ${b.files.model.path}`);
        } catch (e: any) { r.errors.push(`builds.${bn}.files.genaiConfig: ${e?.message ?? e}`); }
      }
    }
    for (const [hn, h] of Object.entries(b.heads ?? {})) {
      const f = await check(b.repo, h.file, `builds.${bn}.heads.${hn}`);
      if (!f) continue;
      try {
        const buf = fs.readFileSync(f);
        const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        const hd = loadAnyHead(ab, { ...h, layer: undefined });
        const hs = "cfg" in hd ? hd.cfg.hidden_size : hd.hiddenSize;
        if (hs !== m.hiddenSize) r.errors.push(`builds.${bn}.heads.${hn}: head hiddenSize ${hs} != manifest ${m.hiddenSize}`);
        if (h.layer !== undefined && h.layer !== "final") r.warnings.push(`builds.${bn}.heads.${hn}: reads layer ${h.layer}; s1s2 0.1 runs final-layer heads only`);
        if ((h.kind ?? "pointer") === "pointer" && (h.format ?? "head.bin") === "head.bin") {
          const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf.buffer, buf.byteOffset + 4, buf.readUInt32LE(0))));
          if (headScaleSource(meta, h.scale) === "default")
            r.warnings.push(`builds.${bn}.heads.${hn}: no scale in the head.bin header nor in the manifest; read with 1/sqrt(headDim) = ${(hd as any).scale} (alias3-web semantics, spec 6.1). State it ("scale": 1 when 1/sqrt(headDim) is folded into q/qb)`);
        }
      } catch (e: any) { r.errors.push(`builds.${bn}.heads.${hn}: head.bin does not parse: ${e?.message ?? e}`); }
    }
    if (b.norm) {
      const f = await check(b.repo, b.norm.file, `builds.${bn}.norm`);
      if (f) try {
        const buf = fs.readFileSync(f);
        const nw = loadNorm(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), b.norm);
        if (nw.hiddenSize !== m.hiddenSize) r.errors.push(`builds.${bn}.norm: hiddenSize ${nw.hiddenSize} != manifest ${m.hiddenSize}`);
      } catch (e: any) { r.errors.push(`builds.${bn}.norm: norm.bin does not parse: ${e?.message ?? e}`); }
    }
    if (m.forecast && b.forecastOutput !== undefined) {
      const fh = m.forecast.head;
      const f = await check(fh.repo ?? b.repo, fh.file, `builds.${bn}.forecast.head`);
      if (f) try {
        const buf = fs.readFileSync(f);
        ForecastReader.load(m.forecast, new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength), m.hiddenSize);
      } catch (e: any) { r.errors.push(`builds.${bn}.forecast.head: ${e?.message ?? e}`); }
      if (m.forecast.trainedOn.baseId !== m.id) r.errors.push(`forecast.trainedOn.baseId ${m.forecast.trainedOn.baseId} != manifest id ${m.id}`);
    }
    if (b.heads && b.hiddenOutput === null) r.errors.push(`builds.${bn}: has heads but hiddenOutput is null (heads read the hidden state)`);
  }
  if (m.evidence && !Object.values(m.builds).some((b) => Object.values(b.heads ?? {}).some((h) => h.types.includes("evidence"))))
    r.warnings.push("evidence: declared, but no build has a head of type \"evidence\"");
}

/** Manifest files for a path: a .json file, or manifest*.json in a directory (else one level of subdirectories). */
export async function findManifests(p: string): Promise<string[]> {
  const fs = await nodeImport("node:fs"), path = await nodeImport("node:path");
  if (!fs.existsSync(p)) throw new Error(`${p}: no such file or directory`);
  if (fs.statSync(p).isFile()) return [p];
  const here = (d: string) => fs.readdirSync(d).filter((f: string) => /^manifest.*\.json$/.test(f)).sort().map((f: string) => path.join(d, f));
  const own = here(p);
  if (own.length) return own;
  const out: string[] = [];
  for (const d of fs.readdirSync(p).sort()) {
    const full = path.join(p, d);
    if (fs.statSync(full).isDirectory()) out.push(...here(full));
  }
  return out;
}
