#!/usr/bin/env node
// s1s2: local tooling for S1S2 model packages (spec/s1s2-package.md).
//
//   s1s2 validate <manifest.json | package dir | dir of package dirs>... [--strict] [--manifest-only] [--root repo=dir]... [--json]
//   s1s2 encode <manifest.json> --tokenizer-dir <dir> --in rows.jsonl --out encoded.jsonl
//
// validate: JSON Schema + validateManifest for every manifest; in package mode (default) also the pinned files found
//   under the manifest's directory (or --root): sha256/size, head.bin parse + hiddenSize, genai_config sizes. Files not
//   present are warnings (errors with --strict). Exit code 1 when any manifest has an error.
// encode: the token ids and read positions s1s2 builds for each row (no model: tokenizer only), so a converter can
//   check that the prompts it trained on are the prompts the runtime sends. Row: {"input": string,
//   "question"?: DecisionsQuestion, "evidence"?: EvidenceRequest}. Output row: {question?: {ids, decide, options, last,
//   labelIds}, evidence?: {ids, decide, options}}.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = (f) => pathToFileURL(path.join(here, "..", "dist", f)).href;

function usage(code = 2) {
  console.error(`usage:
  s1s2 validate <manifest.json | dir>... [--strict] [--manifest-only] [--root repo=dir]... [--json]
  s1s2 encode <manifest.json> --tokenizer-dir <dir> --in rows.jsonl --out encoded.jsonl`);
  process.exit(code);
}

async function validate(args) {
  const paths = [], roots = {};
  let strict = false, manifestOnly = false, json = false;
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--strict") strict = true;
    else if (a === "--manifest-only") manifestOnly = true;
    else if (a === "--json") json = true;
    else if (a === "--root") { const [k, v] = String(args[++i] ?? "").split("="); if (!k || !v) usage(); roots[k] = path.resolve(v); }
    else if (a.startsWith("--")) usage();
    else paths.push(a);
  }
  if (!paths.length) usage();
  const { validateManifestFile, findManifests, PACKAGE_SPEC_VERSION } = await import(dist("package.js"));
  const reports = [];
  for (const p of paths) {
    const files = await findManifests(p);
    if (!files.length) reports.push({ file: p, ok: false, errors: ["no manifest*.json found"], warnings: [], checked: [] });
    for (const f of files) reports.push(await validateManifestFile(f, { strict, manifestOnly, roots }));
  }
  if (json) console.log(JSON.stringify({ spec: PACKAGE_SPEC_VERSION, reports }, null, 1));
  else {
    for (const r of reports) {
      console.log(`${r.ok ? "ok  " : "FAIL"} ${r.file}${r.id ? `  (${r.id})` : ""}${r.checked.length ? `  [${r.checked.length} files checked]` : ""}`);
      for (const e of r.errors) console.log(`     error: ${e}`);
      for (const w of r.warnings) console.log(`     warn:  ${w}`);
    }
    const bad = reports.filter((r) => !r.ok).length;
    console.log(`s1s2 package spec ${PACKAGE_SPEC_VERSION}: ${reports.length - bad}/${reports.length} manifests valid`);
  }
  process.exit(reports.every((r) => r.ok) ? 0 : 1);
}

async function encode(args) {
  const opt = (k) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : undefined; };
  const manPath = args[0], tdir = opt("tokenizer-dir"), inp = opt("in"), out = opt("out");
  if (!manPath || manPath.startsWith("--") || !tdir || !inp || !out) usage();
  const api = await import(dist("index.js"));
  const tjs = await import("@huggingface/transformers");
  // A converter's draft manifest has no build yet: accepted when "no build" is its only problem.
  const raw = JSON.parse(fs.readFileSync(manPath, "utf8"));
  let m;
  try { m = api.validateManifest(raw); }
  catch (e) { if (e?.problems?.length === 1 && /^builds: at least one build/.test(e.problems[0])) m = raw; else throw e; }
  const tok = api.tokenizerFromJson(tjs, JSON.parse(fs.readFileSync(path.join(tdir, "tokenizer.json"), "utf8")), JSON.parse(fs.readFileSync(path.join(tdir, "tokenizer_config.json"), "utf8")));
  api.assertManifestTokens(tok, m);
  const dummyHead = { q: new Float32Array(0), qb: new Float32Array(0), k: new Float32Array(0), kb: new Float32Array(0), headDim: 0, hiddenSize: m.hiddenSize, temperature: 1, scoreOffset: 0 };
  const engine = { manifest: m, tok, backend: { hiddenName: "hidden" }, heads: { evidence: dummyHead }, maxSafeTokens: Number.MAX_SAFE_INTEGER };
  const lines = fs.readFileSync(inp, "utf8").split("\n").filter((l) => l.trim());
  const outRows = [];
  for (const [i, l] of lines.entries()) {
    const row = JSON.parse(l), o = {};
    const state = api.inputToText(row.input);
    try {
      if (row.question) {
        const [q] = api.normalizeQuestions([row.question], m);
        const enc = api.encodeQuestion(tok, m, q, state);
        o.question = { ids: enc.ids, decide: enc.decide, options: enc.options, last: enc.last, labelIds: api.logitIds(engine, q, enc.text) };
      }
      if (row.evidence) {
        const p = api.planEvidence(engine, row.evidence, state);
        o.evidence = { ids: p.enc.ids, decide: p.enc.decide, options: p.enc.options };
      }
    } catch (e) { o.error = String(e?.message ?? e); }
    outRows.push(JSON.stringify(o));
    if ((i + 1) % 500 === 0) console.error(`encode: ${i + 1}/${lines.length}`);
  }
  fs.writeFileSync(out, outRows.join("\n") + "\n");
  const errors = outRows.filter((r) => r.includes('"error"')).length;
  console.log(`encode: ${lines.length} rows -> ${out}${errors ? `, ${errors} with errors` : ""}`);
  process.exit(errors ? 1 : 0);
}

const [cmd, ...rest] = process.argv.slice(2);
if (cmd === "validate") await validate(rest);
else if (cmd === "encode") await encode(rest);
else usage(cmd ? 2 : 0);
