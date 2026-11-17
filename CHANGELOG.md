# Changelog

## 0.1.3

- Exact reuse on WebGPU. onnxruntime-web 1.31.1 (int4 genai graphs) computes every token fed in a prefill block shorter than 32 tokens on a different numeric path, so continuing from a kept state (`s.generate()` after `decide()`) and the prefix shared by several questions differed from a full prefill by up to ~0.5 logit. The genai backend now never feeds a block shorter than 32 tokens (`minBlock`, default 32; `genai: { minBlock: 0 }` restores the old chunking): a short final chunk is merged into the previous one, a kept state stops 32 tokens before the end of its prefill, and `decide()` places the shared prefix at least 32 tokens before each question's end (64 when the state is kept). Reuse is now bit-identical to a full prefill, checked in Chrome on an Apple M2 Pro with the published Qwen3-0.6B package: 32 of 32 decisions with a shared prefix equal to each question alone, 6 of 6 System-2 continuations with the same tokens, 18 of 18 cuts of a kept state with equal logits and hidden states (with the 0.1.2 chunking: 18 of 32, 5 of 6 and 4 of 18). Cost: up to 64 tokens prefilled again per question. A plain prefill whose last chunk was shorter than 32 tokens now gives slightly different numbers than 0.1.2 (the same as reuse). Prompts shorter than 64 tokens keep no state (`sharedPrefill: false`). `reusedTokens` counts the tokens actually served from the kept state.

## 0.1.2

- Docs, comments and messages describe what was measured and how, in plain words, instead of internal study names; a test (`test/plain-names.test.mjs`) keeps those names out of shipped files. `PROVISIONAL_NOTE` now reads `"provisional - no measured threshold"`. Converter: `--recipe kev` (default) and `--recipe short` replace `joint-k` and `preregistered`, which keep working as deprecated aliases; the builder patch is now `onnxruntime-genai-0.19.0-builder-fixes.patch` (same bytes).

## 0.1.1

- `S1S2.load` runs genai builds on the browser CPU (onnxruntime-web WASM) when the manifest lists device `"wasm"`, and falls back to it automatically when WebGPU is absent (`navigator.gpu` missing or no adapter); `cpuFallback: false` opts out, `model.device` / `model.fallbackReason` say what was used. Builds without `"wasm"` (S5-nano: no WASM kernel for GatedDeltaNet) are refused with the reason. Same route as the browser-CPU measurement (120 development items per task, Chrome on an Apple M2 Pro): 7.5–16.5× slower than WebGPU, same decisions in ≥116/120 items; long inputs on ≥1.7B models are slow.

## 0.1.0

- First release.
