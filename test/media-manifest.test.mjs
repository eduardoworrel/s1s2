// Media manifests (spec/MULTIMODAL.md): modalities, audio block, schema + cross-field rules, and that text
// manifests are untouched.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateManifest, validateMediaManifest, mediaProblems, ManifestError } from "./media-helpers.mjs";
import { ManifestError as MediaManifestError, pinnedProblems, loadMedia } from "../dist/media.js";
import { clone, CT0 } from "./helpers.mjs";
import { schemaProblems } from "../dist/package.js";

const load = (n) => JSON.parse(readFileSync(new URL(`../manifests/${n}`, import.meta.url), "utf8"));
const AUDIO = load("marimba-audio.json");
const problems = (m) => { try { validateMediaManifest(m); return []; } catch (e) { assert.equal(e.name, "ManifestError"); return e.problems; } };   // dist/media.js is its own bundle, like dist/manifest.js: its ManifestError is not the root's class, so compare the name

test("the media manifest validates (schema and rules)", async () => {
  assert.deepEqual(problems(AUDIO), []);
  assert.deepEqual(await schemaProblems(AUDIO), []);
});

test("text manifests: no modalities means text, and nothing changes", async () => {
  const ct0 = load("ct0-alias3-micro-1026.json");
  assert.equal(ct0.modalities, undefined);
  assert.doesNotThrow(() => validateManifest(ct0));
  assert.deepEqual(await schemaProblems(ct0), []);
  // text listed explicitly is the same manifest
  assert.doesNotThrow(() => validateManifest({ ...clone(ct0), modalities: ["text"] }));
  assert.deepEqual(await schemaProblems({ ...clone(ct0), modalities: ["text"] }), []);
  // a text manifest still needs its text sections for the schema
  const noEnc = clone(ct0); delete noEnc.encoding;
  assert.ok((await schemaProblems(noEnc)).some((x) => /encoding/.test(x)));
});

test("text + audio: a text manifest with an audio block is valid text and media; audio block without the modality is not", () => {
  const m = { ...clone(CT0), modalities: ["text", "audio"], audio: clone(AUDIO.audio) };
  assert.doesNotThrow(() => validateManifest(m));
  const bad = { ...clone(CT0), audio: clone(AUDIO.audio) };
  assert.throws(() => validateManifest(bad), (e) => e.problems.some((x) => /audio: present, but modalities/.test(x)));
});

test("text + audio: the blocks of the audio part are checked in full by mediaProblems / `s1s2 validate`, not by the text path", async () => {
  const m = { ...clone(CT0), modalities: ["text", "audio"], audio: clone(AUDIO.audio) };
  assert.deepEqual(mediaProblems(m, false), []);
  m.audio.readers.turn_end.tap = "h99";
  assert.ok(mediaProblems(m, false).some((x) => /tap/.test(x)));
  assert.throws(() => validateManifest({ ...clone(CT0), modalities: ["text", "video"] }), (e) => e.problems.some((x) => /modalities/.test(x)));
  assert.throws(() => validateManifest({ ...clone(CT0), modalities: ["text"], audio: clone(AUDIO.audio) }), (e) => e.problems.some((x) => /audio: present/.test(x)));
});

test("a media manifest is refused by the text path with a pointer to the media path", () => {
  assert.throws(() => validateManifest(AUDIO), (e) => e instanceof ManifestError && /loadMedia/.test(e.message));
});

test("media rules: modalities, repo/revision, blocks", () => {
  const bad = (f, re) => { const m = clone(AUDIO); f(m); assert.ok(problems(m).some((x) => re.test(x)), `${re} not in ${problems(m)}`); };
  bad((m) => { m.modalities = []; }, /modalities/);
  bad((m) => { m.modalities = ["audio", "audio"]; }, /distinct/);
  bad((m) => { m.modalities = ["video"]; }, /modalities/);
  bad((m) => { delete m.audio; }, /audio: required/);
  bad((m) => { m.revision = "main"; }, /revision/);
  bad((m) => { delete m.repo; }, /repo/);
  bad((m) => { m.audio.readers.turn_end.tap = "h99"; }, /tap/);
  bad((m) => { m.audio.preprocess.frames = 801; }, /frames/);
  bad((m) => { m.audio.tower.input.shape = [128, 801]; }, /shape/);
  bad((m) => { m.audio.readers.same_speaker.pooling.b.to = 9; }, /pooling.b/);
  bad((m) => { delete m.audio.readers.turn_end.tensors.coef; }, /tensors.coef/);
  bad((m) => { m.audio.readers.turn_end.calibration = { kind: "temperature", tensor: "t" }; }, /predicate reader uses platt/);
  bad((m) => { m.audio.readers.speaker_count.calibration = { kind: "platt", coefficient: "a", intercept: "b" }; }, /choice reader uses temperature/);
  bad((m) => { m.audio.readers.voice_activity.threshold = 1.5; }, /threshold/);
  bad((m) => { m.audio.readers.turn_end.weights.sha256 = "xyz"; }, /sha256/);
});

test("the schema rejects what the rules reject (typos, unknown readers)", async () => {
  const m = clone(AUDIO); m.audio.readers.turn_end.kind = "pooled-linearr";
  assert.ok((await schemaProblems(m)).length > 0);
  const m3 = clone(AUDIO); delete m3.repo;
  assert.ok((await schemaProblems(m3)).some((x) => /repo/.test(x)));
});

test("modalities: [] is invalid, not 'a media package'; text + audio is refused by loadMedia with the right reason", async () => {
  assert.throws(() => validateManifest({ ...clone(CT0), modalities: [] }), (e) => e instanceof ManifestError && e.problems.some((x) => /non-empty list/.test(x)) && !/media package/.test(e.message));
  const both = { ...clone(CT0), modalities: ["text", "audio"], audio: clone(AUDIO.audio), repo: "r", revision: "local" };
  await assert.rejects(() => loadMedia({ manifest: both }), (e) => /lists "text"/.test(e.message) && !/has none/.test(e.message));
});

test("@s1s2/web/media exports the ManifestError that validateMediaManifest throws", () => {
  assert.equal(typeof MediaManifestError, "function");
  assert.throws(() => validateMediaManifest({}), (e) => e instanceof MediaManifestError);
});

test("audio.preprocess numbers: integers where the code uses integers, whole samples in the window, frames as mel.ts counts them", () => {
  const bad = (f, re) => { const m = clone(AUDIO); f(m.audio.preprocess); assert.ok(problems(m).some((x) => re.test(x)), JSON.stringify(problems(m))); };
  bad((p) => { p.nFft = 400.5; }, /nFft: positive integer/);
  bad((p) => { p.nFft = 399; }, /nFft: even/);
  bad((p) => { p.hop = 160.5; }, /hop: positive integer/);
  bad((p) => { p.nMels = 128.5; }, /nMels: positive integer/);
  bad((p) => { p.frames = 800.5; }, /frames: positive integer/);
  bad((p) => { p.windowSeconds = 8.0004; }, /windowSeconds: a whole number of samples/);
  bad((p) => { p.windowSeconds = 8.01; }, /frames: 800 != floor/);   // 801 frames of 10 ms, but the tower is declared for 800
  assert.deepEqual(problems(AUDIO), []);
});

test("pinned paths: backslash separators cannot smuggle '..' or an absolute path", () => {
  const at = (path) => { const p = []; pinnedProblems({ path, sha256: "0".repeat(64) }, "f", p); return p; };
  for (const bad of ["a\\..\\..\\x", "..\\x", "\\abs\\x", "../../x", "a/../x", "/abs", "C:\\x", "file:x"]) assert.equal(at(bad).length, 1, bad);
  for (const ok of ["marimba-audio/a.onnx", "https://example.org/a.onnx", "a..b/c"]) assert.deepEqual(at(ok), [], ok);
  const m = clone(AUDIO); m.audio.tower.file.path = "a\\..\\..\\x";
  assert.ok(problems(m).some((x) => /tower.file.path/.test(x)));
});
