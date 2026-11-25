# Changelog

## Unreleased

- New entry point `@s1s2/web/media` (`loadMedia`, `AudioModel`, `validateMediaManifest`, ...): audio in, same answers (`predicate`, `choice`) with probability and margin. The text bundle carries none of it.
- Manifest keys `modalities` (default `["text"]`) and `audio` (log-mel preprocess, ONNX tower, readers); `s1s2 validate` checks them. Text manifests are unchanged.
- The safetensors reader accepts `F64` tensors (the audio readers store float64 coefficients).

## 0.1.3

- `S1S2.load` picks a runtime for the device: ONNX Runtime Web (GenAI graphs and plain decoders), transformers.js, WebLLM or LiteRT.js in the browser, onnxruntime-node in Node. Builds that list device `"wasm"` run on the browser CPU when WebGPU is absent (`cpuFallback: false` opts out; `model.device` and `model.fallbackReason` say what was used); builds without it are refused with the reason.
- `decide()` answers `predicate`, `choice` and `score` questions in one forward pass and reports `probability`, `confidence`, `margin` and `rawMargin`. `s.generate()` continues from the same prefill. On WebGPU, reuse of a kept state and of the prefix shared by several questions is bit-identical to a full prefill: the genai backend never feeds a prefill block shorter than 32 tokens (`minBlock`).
- Choice by label logits with `logits.choice.map: "letters"` reads the option keys as the model writes them after the cue (" A", " B", ...).
- LiteRT frees every output tensor of a step that is not handed on.
- Packages: manifest `schema: "s1s2.manifest/1"` pins every file by sha256; `s1s2 validate` checks it and rejects any other schema id. The cache directory is `$S1S2_CACHE`.
- Docs, comments and messages describe what was measured in plain words; `test/plain-names.test.mjs` keeps internal names out of shipped files. `PROVISIONAL_NOTE` is `"provisional - no measured threshold"`.

## Converter 0.1.1

- `s1s2-convert` trains (`--recipe kev` by default, or `short`), fuses, exports and packs a specialist; the builder patch is `onnxruntime-genai-0.19.0-builder-fixes.patch`.
- `pack-mlc` and `pack-litert` (also `python -m s1s2_convert.pack_mlc|pack_litert`) repackage a published GenAI S1S2 repo as a WebLLM (MLC q4f16_1) or LiteRT.js (int4) repo. The LoRA is merged into the base in bf16 (`fuse_lora.py`), quantized by the runtime's own tool, and one manifest per source manifest is written. WebLLM builds ship a hidden-state recovery matrix (`recovery.rows.i32`, `recovery.P.f16`) so the trained heads work on logits-only WebLLM; `--readout logits` drops the heads. LiteRT builds read the label logits. `pin` writes the upload commit into the manifests and runs `s1s2 validate --strict`.
- `s1s2_convert/node/browser_decide.mjs` loads a packed folder in Chrome (WebGPU) with s1s2 and decides prompts.
- `converter/README.md` has the steps for other runtimes, the pinned tool versions and the architectures each converter handles.
