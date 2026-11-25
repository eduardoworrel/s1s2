// S1S2 package validator: JSON Schema + validateManifest + package file checks (spec/s1s2-package.md).
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { schemaProblems, validateManifestFile, findManifests } from "../dist/package.js";
import { validateManifest } from "../dist/index.js";
import { CT0, clone, toyManifest } from "./helpers.mjs";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const repoManifests = readdirSync(join(ROOT, "manifests")).filter((f) => f.endsWith(".json")).map((f) => join(ROOT, "manifests", f));
const sha = (b) => createHash("sha256").update(b).digest("hex");

/** head.bin as s1s2 reads it (u32 LE json length, json, fp16 q, qb, k, kb). */
function headBin(headDim, hiddenSize, extra = { scale: 1 }) {
  const meta = Buffer.from(JSON.stringify({ headDim, hiddenSize, temperature: 1, ...extra }));
  const n = 2 * headDim * hiddenSize + 2 * headDim;
  const b = Buffer.alloc(4 + meta.length + 2 * n);
  b.writeUInt32LE(meta.length, 0); meta.copy(b, 4);
  return b;
}

test("schema and validateManifest accept every manifest shipped in the repo", async () => {
  for (const f of repoManifests) {
    const m = JSON.parse(readFileSync(f, "utf8"));
    assert.deepEqual(await schemaProblems(m), [], f);
    validateManifest(m);
  }
  assert.deepEqual(await schemaProblems(toyManifest()), []);
});

test("schema rejects structural errors validateManifest also rejects", async () => {
  const cases = [
    ["unknown top-level key", (m) => { m.extra = 1; }],
    ["branch instead of revision", (m) => { m.base.revision = "main"; }],
    ["bad sha256", (m) => { m.tokenizer.files.tokenizer.sha256 = "XYZ"; }],
    ["genai build without data", (m) => { delete Object.values(m.builds)[0].files.data; }],
    ["unknown head role", (m) => { Object.values(m.builds)[0].heads = { h: { file: { path: "h.bin", sha256: "a".repeat(64) }, types: ["evidense"] } }; }],
    ["escalation typo", (m) => { m.escalation = { belo: { choice: 0.1 } }; }],
    ["choice map tokens without tokens", (m) => { m.logits.choice = { map: "tokens" }; }],
    ["path escapes the repo", (m) => { m.tokenizer.files.config.path = "../x.json"; }],
  ];
  for (const [what, edit] of cases) {
    const m = clone(CT0); edit(m);
    assert.ok((await schemaProblems(m)).length > 0, `schema should reject: ${what}`);
  }
  for (const [what, edit] of cases.filter(([w]) => w !== "unknown top-level key")) {
    const m = clone(CT0); edit(m);
    assert.throws(() => validateManifest(m), undefined, `validateManifest should reject: ${what}`);
  }
});

test("lab notes, x- extensions, license and provenance are allowed; malformed provenance is not", async () => {
  const m = clone(CT0);
  m.lab = { anything: true }; m["x-vendor"] = 1; m.license = "Apache-2.0";
  m.provenance = { generator: { name: "s1s2-convert", version: "0.1.0" }, createdAt: "2026-11-15T00:00:00Z",
    base: { repo: "a/b", revision: "0".repeat(40), license: "Apache-2.0" }, dataset: { sha256: "a".repeat(64), rows: 3 },
    training: { recipe: "kev", seed: 1 }, builds: { x: { tool: "onnxruntime-genai builder", version: "0.19.0", patches: [{ path: "p.patch", sha256: "b".repeat(64) }] } } };
  assert.deepEqual(await schemaProblems(m), []);
  validateManifest(m);
  const bad = clone(m); bad.provenance.dataset.sha256 = "short"; bad.provenance.generator = { name: "x" };
  assert.ok((await schemaProblems(bad)).length >= 2);
  assert.throws(() => validateManifest(bad), /provenance.generator.*\n.*provenance.dataset|provenance.dataset[\s\S]*provenance.generator|provenance.generator[\s\S]*provenance.dataset/);
});

function makePackage({ headHidden, parent, headMeta } = {}) {
  const dir = parent ? join(parent, "pkg") : mkdtempSync(join(tmpdir(), "s1s2-pkg-"));
  mkdirSync(dir, { recursive: true });
  const m = toyManifest();
  const files = { "tokenizer.json": Buffer.from("{}"), "tokenizer_config.json": Buffer.from("{}"), "model.onnx": Buffer.from("graph"), "model.onnx.data": Buffer.from("data"),
    "heads/choice/head.bin": headBin(2, headHidden ?? m.hiddenSize, headMeta) };
  mkdirSync(join(dir, "heads/choice"), { recursive: true });
  for (const [p, b] of Object.entries(files)) writeFileSync(join(dir, p), b);
  const pin = (p) => ({ path: p, sha256: sha(files[p]), size: files[p].length });
  m.tokenizer.files = { tokenizer: pin("tokenizer.json"), config: pin("tokenizer_config.json") };
  for (const b of Object.values(m.builds)) {
    b.files = { model: pin("model.onnx"), data: pin("model.onnx.data") };
    b.heads = { choice: { file: pin("heads/choice/head.bin"), types: ["choice"] } };
    b.hiddenOutput = "hidden_states";
  }
  writeFileSync(join(dir, "manifest.json"), JSON.stringify(m));
  return { dir, m };
}

test("package mode: files checked by sha256/size, head.bin parsed, hiddenSize compared", async () => {
  const { dir } = makePackage();
  const r = await validateManifestFile(join(dir, "manifest.json"));
  assert.equal(r.ok, true, r.errors.join("\n"));
  assert.equal(r.checked.length, 5);

  writeFileSync(join(dir, "model.onnx.data"), "DATA"); // same size, other bytes
  const r2 = await validateManifestFile(join(dir, "manifest.json"));
  assert.equal(r2.ok, false);
  assert.match(r2.errors.join("\n"), /model.onnx.data sha256/);

  const { dir: d3 } = makePackage({ headHidden: 7 });
  const r3 = await validateManifestFile(join(d3, "manifest.json"));
  assert.match(r3.errors.join("\n"), /head hiddenSize 7 != manifest/);

  const { dir: d4 } = makePackage({ headMeta: {} }); // a head.bin that does not say its scale: alias3-web default, warned
  const r4 = await validateManifestFile(join(d4, "manifest.json"));
  assert.equal(r4.ok, true);
  assert.match(r4.warnings.join("\n"), /no scale in the head.bin header nor in the manifest; read with 1\/sqrt\(headDim\)/);
  const m4 = JSON.parse(readFileSync(join(d4, "manifest.json"), "utf8"));
  for (const b of Object.values(m4.builds)) b.heads.choice.scale = 0.0625;
  writeFileSync(join(d4, "manifest.json"), JSON.stringify(m4));
  assert.equal((await validateManifestFile(join(d4, "manifest.json"))).ok, true);
});

test("package mode: missing files are warnings, errors with strict; manifest-only skips them", async () => {
  const { dir } = makePackage();
  const m = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
  for (const b of Object.values(m.builds)) b.files.data.path = "absent.data";
  writeFileSync(join(dir, "manifest.json"), JSON.stringify(m));
  const r = await validateManifestFile(join(dir, "manifest.json"));
  assert.equal(r.ok, true);
  assert.match(r.warnings.join("\n"), /absent.data not present/);
  const s = await validateManifestFile(join(dir, "manifest.json"), { strict: true });
  assert.equal(s.ok, false);
  const mo = await validateManifestFile(join(dir, "manifest.json"), { manifestOnly: true });
  assert.equal(mo.ok, true); assert.equal(mo.checked.length, 0); assert.equal(mo.warnings.length, 0);
});

test("findManifests: file, package dir, dir of package dirs", async () => {
  const parent = mkdtempSync(join(tmpdir(), "s1s2-pkgs-"));
  const { dir } = makePackage({ parent });
  assert.deepEqual(await findManifests(join(dir, "manifest.json")), [join(dir, "manifest.json")]);
  assert.deepEqual(await findManifests(dir), [join(dir, "manifest.json")]);
  assert.deepEqual(await findManifests(parent), [join(dir, "manifest.json")]);
});

test("s1s2 validate CLI: exit code 0 on valid, 1 on invalid", () => {
  const { dir } = makePackage();
  const out = execFileSync(process.execPath, [join(ROOT, "bin/s1s2.mjs"), "validate", dir], { encoding: "utf8" });
  assert.match(out, /1\/1 manifests valid/);
  const m = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
  m.hiddenSize = 0;
  writeFileSync(join(dir, "manifest.json"), JSON.stringify(m));
  assert.throws(() => execFileSync(process.execPath, [join(ROOT, "bin/s1s2.mjs"), "validate", dir], { encoding: "utf8", stdio: "pipe" }), (e) => e.status === 1);
});

test("s1s2 validate CLI: a * in the last path segment is expanded when the shell does not (cmd, PowerShell)", () => {
  const { dir } = makePackage();
  const src = readFileSync(join(dir, "manifest.json"));
  const glob = mkdtempSync(join(tmpdir(), "s1s2-glob-"));
  for (const f of ["m1.json", "m2.json", ".hidden.json"]) writeFileSync(join(glob, f), src);
  const run = (pattern) => execFileSync(process.execPath, [join(ROOT, "bin/s1s2.mjs"), "validate", "--manifest-only", join(glob, pattern)], { encoding: "utf8", stdio: "pipe" });
  assert.match(run("m*.json"), /2\/2 manifests valid/);
  assert.match(run("*.json"), /2\/2 manifests valid/); // like a shell, a leading * skips the dotfile
  assert.match(run("m*2.json"), /1\/1 manifests valid/);
  // prefix and suffix must not overlap: "m1*1.json" does not match "m1.json"
  assert.throws(() => run("m1*1.json"), (e) => e.status === 1);
});

test("head scale: header scale, scale_folded_into_q, the manifest's, else 1/sqrt(headDim) (alias3-web); a conflict is refused", async () => {
  const { loadHead, headScore } = await import("../dist/index.js");
  const mk = (meta) => {
    const j = Buffer.from(JSON.stringify({ headDim: 1, hiddenSize: 1, temperature: 1, ...meta }));
    const b = Buffer.alloc(4 + j.length + 8); b.writeUInt32LE(j.length, 0); j.copy(b, 4);
    const one = 0x3c00; // fp16 1.0: q = 1, qb = 0, k = 1, kb = 0
    b.writeUInt16LE(one, 4 + j.length); b.writeUInt16LE(one, 4 + j.length + 4);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  };
  const h = new Float32Array([2]), k = new Float32Array([3]);
  assert.equal(headScore(loadHead(mk({})), h, k), 6); // headDim 1: 1/sqrt(1) = 1
  assert.equal(headScore(loadHead(mk({}), 1 / 16), h, k), 6 / 16);
  assert.equal(headScore(loadHead(mk({ scale_folded_into_q: 0.0625 })), h, k), 6);
  assert.equal(headScore(loadHead(mk({ scale: 0.5 })), h, k), 3);
  assert.throws(() => loadHead(mk({ scale: 0.5 }), 1), /header scale 0.5 != manifest scale 1/);
  assert.throws(() => loadHead(mk({ scale: -1 })), /not a positive number/);
});

test("logits temperature: validated, applied as softmax(L / T)", async () => {
  const m = clone(CT0);
  m.logits.temperature = { choice: 2 };
  assert.deepEqual(await schemaProblems(m), []);
  validateManifest(m);
  m.logits.temperature = { choice: 0 };
  assert.throws(() => validateManifest(m), /logits.temperature.choice/);
  assert.ok((await schemaProblems(m)).length > 0);
});

test("spec 0.2 fields pass the schema and the validator: head kinds, build norm and attention, readout.score option, confidence hook", async () => {
  const m = clone(CT0);
  const b = Object.values(m.builds)[0];
  const f = { path: "heads/x.safetensors", sha256: "0".repeat(64) };
  b.heads = { lin: { file: f, types: ["choice"], kind: "linear", format: "safetensors", tensors: { weight: "proj.weight", bias: null }, rows: { choice: { start: 8, end: 24 } }, softcap: 30, temperature: { choice: 0.98 }, read: "decide", layer: "final", source: "jev-judge-config" } };
  b.norm = { file: { path: "heads/norm.bin", sha256: "1".repeat(64) }, kind: "rmsnorm", eps: 0, weight: "w" };
  b.attention = "causal";
  m.confidence = { definition: "tbd-007" };
  assert.deepEqual(await schemaProblems(m), []);
  validateManifest(m);
  const bad = clone(m); Object.values(bad.builds)[0].attention = "bidirectional";
  assert.ok((await schemaProblems(bad)).length > 0);
  assert.throws(() => validateManifest(bad), /attention/);
});
