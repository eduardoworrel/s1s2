// The loadMedia snippets of the docs must run as written: a relative baseUrls value is an Invalid URL for fileUrl.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";
import { fileUrl } from "../dist/index.js";
import { ROOT } from "./media-helpers.mjs";

for (const doc of ["README.md", "README.pt-BR.md", "spec/MULTIMODAL.md"]) {
  test(`${doc}: the loadMedia baseUrls example resolves to a file URL`, () => {
    const text = readFileSync(path.join(ROOT, doc), "utf8");
    const m = /baseUrls:\s*\{\s*"([^"]+)":\s*(.+?)\s*\}\s*,\s*modules/.exec(text);
    assert.ok(m, "no loadMedia example with baseUrls found");
    const location = { href: "http://127.0.0.1:8788/audio/index.html" };
    const base = new Function("location", `return (${m[2]});`)(location);
    assert.equal(fileUrl(m[1], "local", "audio_tower.onnx", base), "http://127.0.0.1:8788/weights/marimba-audio/audio_tower.onnx");
  });
}

test("the README states the largest probability difference the manifest measured", () => {
  const m = JSON.parse(readFileSync(path.join(ROOT, "manifests/marimba-audio.json"), "utf8"));
  const max = Math.max(...Object.values(m.audio.fidelity?.chrome157WebGPU?.maxAbsDiffP ?? m.fidelity.chrome157WebGPU.maxAbsDiffP));
  const sci = (x, sep) => `${(x * 1e5).toFixed(2)}e-5`.replace(".", sep);
  assert.ok(readFileSync(path.join(ROOT, "README.md"), "utf8").includes(`within ${sci(max, ".")} of`), `README.md: ${sci(max, ".")}`);
  assert.ok(readFileSync(path.join(ROOT, "README.pt-BR.md"), "utf8").includes(`a até ${sci(max, ",")} do`), `README.pt-BR.md: ${sci(max, ",")}`);
});

test("every audio file shipped with the repo is attributed, and the NOTICE names the audio sources", () => {
  const read = (f) => readFileSync(path.join(ROOT, f), "utf8");
  const clips = readdirSync(path.join(ROOT, "examples/audio-turn/clips")).filter((f) => f.endsWith(".wav"));
  const attr = read("examples/audio-turn/clips/ATTRIBUTION.md");
  for (const c of clips) assert.ok(attr.includes(c), `${c} is not in clips/ATTRIBUTION.md`);
  const meta = JSON.parse(read("test/fixtures/media/audio/meta.json")), fa = read("test/fixtures/media/audio/ATTRIBUTION.md");
  for (const w of meta.windows) assert.ok(fa.includes(w.ref.fn) && fa.includes(`\`${w.key}\``), `fixture ${w.key} (${w.ref.fn}) is not in the fixtures' ATTRIBUTION.md`);
  assert.ok(fa.includes("CC BY 4.0"));
  const notice = read("NOTICE");
  for (const name of ["Marimba", "Qwen3-ASR", "Smart Turn v3.2", "FLEURS", "CC BY 4.0"]) assert.ok(notice.includes(name), `NOTICE does not mention ${name}`);
});

// ---- the audio package is on the Hub: the embedded manifest is the published one, and the README snippet reads from there ----
const readJson = (f) => JSON.parse(readFileSync(path.join(ROOT, f), "utf8"));
const PUBLISHED = "test/fixtures/hub-marimba-manifest.json"; // copy of manifest.json of eduardoworrel/Marimba-ONNX-S1S2 at the pinned commit (no weights)

test("manifests/marimba-audio.json equals the manifest published on the Hub (repo, revision, every hash)", () => {
  const lib = readJson("manifests/marimba-audio.json"), hub = readJson(PUBLISHED);
  assert.equal(lib.repo, "eduardoworrel/Marimba-ONNX-S1S2");
  assert.match(lib.revision, /^[0-9a-f]{40}$/);
  assert.deepEqual(lib, hub);
  assert.ok(!JSON.stringify(lib).includes('"local"') && !/local\//.test(lib.repo));
});

for (const doc of ["README.md", "README.pt-BR.md"]) {
  test(`${doc}: the Hub snippet loads from the Hub with the pinned revision and asks questions the manifest has`, async () => {
    const text = readFileSync(path.join(ROOT, doc), "utf8");
    const blocks = [...text.matchAll(/```js\n([\s\S]*?)```/g)].map((m) => m[1]);
    const code = blocks.find((b) => b.includes("huggingface.co/eduardoworrel/Marimba-ONNX-S1S2/resolve/main/manifest.json"));
    assert.ok(code, "no Hub snippet");
    const lib = readJson("manifests/marimba-audio.json");
    // 1. run the loadMedia call as written, with a fetch that serves the manifest and records the first file URL it is asked for
    const { loadMedia } = await import("../dist/media.js");
    const seen = [];
    const fetchMock = async (url) => {
      seen.push(String(url));
      if (String(url).endsWith("/resolve/main/manifest.json")) return new Response(JSON.stringify(lib), { status: 200 });
      return new Response("stop", { status: 404 });
    };
    const head = code.slice(code.indexOf("const audio = await loadMedia("), code.indexOf("// samples:"));
    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
    const run = new AsyncFunction("loadMedia", "ort", "console", `${head}; return audio;`);
    const wrapped = (o) => loadMedia({ ...o, fetch: fetchMock, cacheDir: path.join(tmpdir(), "s1s2-docs-test") });
    assert.ok(!/baseUrls|allowUnverified|localDirs/.test(head), "the Hub snippet needs none of the local options");
    await assert.rejects(run(wrapped, {}, { log() {} }));
    assert.equal(seen[0], "https://huggingface.co/eduardoworrel/Marimba-ONNX-S1S2/resolve/main/manifest.json");
    assert.ok(seen.length > 1 && seen[1].startsWith(`https://huggingface.co/eduardoworrel/Marimba-ONNX-S1S2/resolve/${lib.revision}/`), `first file URL: ${seen[1]}`);
    // 2. every question the snippet asks exists in the manifest, and `transcribe` returns what the snippet destructures
    const names = [...code.matchAll(/"([a-z_]+)"/g)].map((m) => m[1]).filter((n) => /^(voice_activity|turn_end|same_speaker|speaker_change|speaker_count|language)$/.test(n));
    assert.equal(names.length, 6);
    for (const n of [...names, "term"]) assert.ok(n in lib.audio.readers, `reader ${n}`);
    assert.match(code, /\{ task: "term", term: "[^"]+" \}/);
    assert.match(code, /const \{ text, language \} = await audio\.transcribe\(enc\)/);
    const { parseAsr } = await import("../dist/media.js");
    assert.ok(typeof parseAsr === "function");
  });
}
