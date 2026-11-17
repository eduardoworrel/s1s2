# Changelog

## 0.1.1

- `S1S2.load` runs genai builds on the browser CPU (onnxruntime-web WASM) when the manifest lists device `"wasm"`, and falls back to it automatically when WebGPU is absent (`navigator.gpu` missing or no adapter); `cpuFallback: false` opts out, `model.device` / `model.fallbackReason` say what was used. Builds without `"wasm"` (S5-nano: no WASM kernel for GatedDeltaNet) are refused with the reason. Same route as lab-s12 study 011 arm C: 7.5–16.5× slower than WebGPU, same decisions in ≥116/120 items; long inputs on ≥1.7B models are slow.

## 0.1.0

- First release.
