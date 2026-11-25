// Validation of the media blocks of a manifest (`modalities`, `audio`). Cross-field rules a JSON Schema cannot
// say (a reader's tap exists, spans fit the window, tensor names are given) live here; the structure is in
// spec/s1s2-manifest.schema.json. validateMediaManifest / validateManifest (src/manifest.ts) call mediaProblems.

import { MODALITIES, type MediaManifest } from "./types.js";
import { ManifestError, MANIFEST_SCHEMA, provenanceProblems } from "../manifest.js";

const HEX64 = /^[0-9a-f]{64}$/, SHA1 = /^[0-9a-f]{40}$/;
const isObj = (x: any) => !!x && typeof x === "object" && !Array.isArray(x);
const str = (x: any) => typeof x === "string" && x.length > 0;
const num = (x: any) => typeof x === "number" && Number.isFinite(x);
const pos = (x: any) => num(x) && x > 0;
const int = (x: any) => Number.isInteger(x) && x > 0;

export function pinnedProblems(x: any, at: string, p: string[]): void {
  if (!isObj(x)) { p.push(`${at}: expected {path, sha256}`); return; }
  const segs = str(x.path) ? x.path.replace(/\\/g, "/").split("/") : [];   // "\" separates paths on Windows
  if (!(str(x.path) && (/^https:\/\//.test(x.path) || (!x.path.startsWith("/") && !x.path.startsWith("\\") && !/^[a-z]+:/i.test(x.path))) && !segs.includes(".."))) p.push(`${at}.path: repo-relative path or absolute https URL required`);
  if (!(typeof x.sha256 === "string" && HEX64.test(x.sha256))) p.push(`${at}.sha256: 64 lowercase hex chars required`);
  if (x.size !== undefined && !(Number.isInteger(x.size) && x.size >= 0)) p.push(`${at}.size: non-negative integer`);
}

/** Problems of the media blocks of a manifest (empty = fine). `mediaOnly`: the manifest has no text sections. */
export function mediaProblems(m: any, mediaOnly: boolean): string[] {
  const p: string[] = [];
  const mod = m.modalities;
  if (mod !== undefined) {
    if (!Array.isArray(mod) || !mod.length || !mod.every((x: any) => MODALITIES.includes(x)) || new Set(mod).size !== mod.length) p.push(`modalities: a non-empty list of distinct values among ${MODALITIES.join(", ")}`);
  }
  const has = (k: string) => Array.isArray(mod) && mod.includes(k);
  if (mediaOnly) {
    if (!Array.isArray(mod) || !has("audio")) p.push('modalities: a media manifest (no text sections) lists "audio"');
    if (has("text")) p.push('modalities: lists "text": this is a text manifest (use load()); loadMedia reads media-only packages');
    if (!str(m.repo)) p.push("repo: string (where the pinned files live)");
    if (!(typeof m.revision === "string" && (SHA1.test(m.revision) || m.revision === "local"))) p.push('revision: a 40-hex commit sha, or "local" for an unpublished package');
  }
  if (m.audio !== undefined && !has("audio")) p.push('audio: present, but modalities does not list "audio"');
  if (has("audio")) { if (m.audio === undefined) p.push('audio: required by modalities "audio"'); else audioProblems(m.audio, p); }
  return p;
}

function spanOk(s: any, at: string, win: number, p: string[]) {
  if (!(isObj(s) && num(s.from) && num(s.to) && s.from >= 0 && s.to > s.from && s.to <= win)) p.push(`${at}: {from, to} in seconds with 0 <= from < to <= windowSeconds (${win})`);
}

function audioProblems(a: any, p: string[]): void {
  if (!isObj(a)) { p.push("audio: object"); return; }
  const pre = a.preprocess;
  let win = 0;
  if (!isObj(pre) || pre.kind !== "log-mel") p.push('audio.preprocess.kind: "log-mel"');
  else {
    if (pre.sampleRate !== 16000) p.push("audio.preprocess.sampleRate: 16000 (the only rate measured)");
    for (const k of ["nFft", "hop", "nMels", "frames"]) if (!int(pre[k])) p.push(`audio.preprocess.${k}: positive integer`);
    if (!pos(pre.windowSeconds)) p.push("audio.preprocess.windowSeconds: positive number");
    else if (Math.abs(pre.windowSeconds * pre.sampleRate - Math.round(pre.windowSeconds * pre.sampleRate)) > 1e-6) p.push("audio.preprocess.windowSeconds: a whole number of samples at sampleRate");
    if (int(pre.nFft) && pre.nFft % 2) p.push("audio.preprocess.nFft: even (reflect padding of nFft/2)");
    if (pre.window !== "hann-periodic" || pre.padding !== "reflect" || pre.filters !== "slaney") p.push('audio.preprocess: window "hann-periodic", padding "reflect", filters "slaney"');
    const L = pre.log;
    if (!isObj(L) || !pos(L.floor) || !pos(L.dynamicRange) || !num(L.shift) || !pos(L.divide)) p.push("audio.preprocess.log: {floor, dynamicRange, shift, divide}");
    if (pos(pre.windowSeconds) && int(pre.hop) && int(pre.frames) && Math.floor(Math.round(pre.windowSeconds * pre.sampleRate) / pre.hop) !== pre.frames)
      p.push(`audio.preprocess.frames: ${pre.frames} != floor(windowSeconds * sampleRate / hop) = ${Math.floor(Math.round(pre.windowSeconds * pre.sampleRate) / pre.hop)}`);
    win = pre.windowSeconds;
  }
  const t = a.tower;
  if (!isObj(t) || t.runtime !== "onnx") p.push('audio.tower.runtime: "onnx"');
  else {
    pinnedProblems(t.file, "audio.tower.file", p);
    if (!(str(t.input?.name) && Array.isArray(t.input.shape) && t.input.shape.length === 2 && t.input.shape.every(int))) p.push("audio.tower.input: {name, shape: [nMels, frames]}");
    else if (isObj(pre) && (t.input.shape[0] !== pre.nMels || t.input.shape[1] !== pre.frames)) p.push(`audio.tower.input.shape: [${t.input.shape}] != [nMels, frames] = [${pre.nMels}, ${pre.frames}]`);
    if (!isObj(t.outputs) || !Object.keys(t.outputs).length) p.push("audio.tower.outputs: at least one tap");
    else for (const [k, o] of Object.entries<any>(t.outputs)) if (!(isObj(o) && int(o.frames) && int(o.dim) && (int(o.layer) || o.layer === "final"))) p.push(`audio.tower.outputs.${k}: {layer: int | "final", frames, dim}`);
    if (!Array.isArray(t.devices) || !t.devices.length || !t.devices.every((d: any) => ["webgpu", "wasm", "cpu"].includes(d))) p.push('audio.tower.devices: non-empty list of "webgpu" | "wasm" | "cpu"');
  }
  if (!isObj(a.readers) || !Object.keys(a.readers).length) { p.push("audio.readers: at least one task"); return; }
  for (const [name, r] of Object.entries<any>(a.readers)) {
    const at = `audio.readers.${name}`;
    if (!isObj(r)) { p.push(`${at}: object`); continue; }
    if (!str(r.tap) || !isObj(t?.outputs) || !t.outputs[r.tap]) p.push(`${at}.tap: one of the tower outputs`);
    if (r.kind === "speech-head") {
      if (r.type !== "predicate") p.push(`${at}.type: "predicate"`);
      pinnedProblems(r.graph, `${at}.graph`, p); pinnedProblems(r.question, `${at}.question`, p);
      if (!int(r.question?.dim) || !str(r.question?.prompt)) p.push(`${at}.question: {path, sha256, dim, prompt}`);
      if (!(isObj(r.inputs) && ["tap", "mask", "question", "output"].every((k) => str(r.inputs[k])))) p.push(`${at}.inputs: {tap, mask, question, output} graph names`);
      if (!int(r.frames)) p.push(`${at}.frames: positive integer`);
      if (!pos(r.temperature)) p.push(`${at}.temperature: positive number`);
      if (!(num(r.threshold) && r.threshold > 0 && r.threshold < 1)) p.push(`${at}.threshold: number in (0, 1)`);
      continue;
    }
    if (r.kind !== "pooled-linear" && r.kind !== "pooled-mlp") { p.push(`${at}.kind: "pooled-linear" | "pooled-mlp" | "speech-head"`); continue; }
    pinnedProblems(r.weights, `${at}.weights`, p);
    if (!isObj(r.tensors)) p.push(`${at}.tensors: names of the tensors in the weights file`);
    const tn = r.tensors ?? {};
    const need = r.kind === "pooled-linear" ? ["coef", "intercept"] : ["layer0Weight", "layer0Bias", "layer1Weight", "layer1Bias"];
    for (const k of need) if (!str(tn[k])) p.push(`${at}.tensors.${k}: tensor name`);
    if (r.kind === "pooled-linear") {
      if (r.type === "choice") { if (!(Array.isArray(r.labels) && r.labels.length >= 2 && r.labels.every(str))) p.push(`${at}.labels: at least 2 labels for a choice reader`); }
      else if (r.type !== "predicate") p.push(`${at}.type: "predicate" | "choice"`);
      if (!!tn.mean !== !!tn.scale) p.push(`${at}.tensors: mean and scale go together`);
    } else if (r.type !== "predicate") p.push(`${at}.type: "predicate"`);
    const c = r.calibration;
    if (!isObj(c) || !["platt", "temperature", "none"].includes(c.kind)) p.push(`${at}.calibration.kind: "platt" | "temperature" | "none"`);
    else if (c.kind === "platt" && !(str(c.coefficient) && str(c.intercept))) p.push(`${at}.calibration: platt needs {coefficient, intercept} tensor names`);
    else if (c.kind === "temperature" && !str(c.tensor)) p.push(`${at}.calibration: temperature needs {tensor}`);
    if (r.type === "choice" && isObj(c) && c.kind === "platt") p.push(`${at}.calibration: a choice reader uses temperature or none`);
    if (r.type === "predicate" && isObj(c) && c.kind === "temperature") p.push(`${at}.calibration: a predicate reader uses platt or none`);
    const g = r.pooling;
    if (!isObj(g)) { p.push(`${at}.pooling: object`); continue; }
    if (g.kind === "tail-mean") { if (!(pos(g.seconds) && win && g.seconds <= win)) p.push(`${at}.pooling.seconds: positive, at most windowSeconds`); }
    else if (g.kind === "pair") { spanOk(g.a, `${at}.pooling.a`, win, p); spanOk(g.b, `${at}.pooling.b`, win, p); }
    else if (g.kind === "pair-split") { if (!(pos(g.split) && g.split < win)) p.push(`${at}.pooling.split: seconds inside the window`); }
    else if (g.kind !== "mean-std") p.push(`${at}.pooling.kind: "tail-mean" | "pair" | "pair-split" | "mean-std"`);
    if (r.kind === "pooled-mlp" && g.kind === "mean-std") p.push(`${at}: the MLP reader is specified for pair features`);
  }
}

/** Validate a media package manifest (no text sections; `modalities` lists "audio"). Throws ManifestError. */
export function validateMediaManifest(m: any): MediaManifest {
  const p: string[] = [];
  if (!m || typeof m !== "object" || Array.isArray(m)) throw new ManifestError(["manifest: expected an object"]);
  if (m.schema !== MANIFEST_SCHEMA) p.push(`schema: unsupported schema id ${JSON.stringify(m.schema)}; expected "${MANIFEST_SCHEMA}"`);
  if (typeof m.id !== "string" || !m.id) p.push("id: non-empty string");
  p.push(...mediaProblems(m, true));
  if (m.provenance !== undefined) provenanceProblems(m.provenance, p);
  if (p.length) throw new ManifestError(p);
  return m as MediaManifest;
}
