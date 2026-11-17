# Changelog

## 0.1.2

- Docs, comments and messages describe what was measured and how, in plain words, instead of internal study names; a test (`test/plain-names.test.mjs`) keeps those names out of shipped files. `PROVISIONAL_NOTE` now reads `"provisional - no measured threshold"`. Converter: `--recipe kev` (default) and `--recipe short` replace `joint-k` and `preregistered`, which keep working as deprecated aliases; the builder patch is now `onnxruntime-genai-0.19.0-builder-fixes.patch` (same bytes).

## 0.1.1

- `S1S2.load` runs genai builds on the browser CPU (onnxruntime-web WASM) when the manifest lists device `"wasm"`, and falls back to it automatically when WebGPU is absent (`navigator.gpu` missing or no adapter); `cpuFallback: false` opts out, `model.device` / `model.fallbackReason` say what was used. Builds without `"wasm"` (S5-nano: no WASM kernel for GatedDeltaNet) are refused with the reason. Same route as the browser-CPU measurement (120 development items per task, Chrome on an Apple M2 Pro): 7.5–16.5× slower than WebGPU, same decisions in ≥116/120 items; long inputs on ≥1.7B models are slow.

## 0.1.0

- First release.
