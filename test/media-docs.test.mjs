// The loadMedia snippets of the docs must run as written: a relative baseUrls value is an Invalid URL for fileUrl.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
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
