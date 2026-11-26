# Changelog

## Unreleased

## 0.2.1 (2026-11-26)

- `AudioModel.transcribe(input, options?)`: free transcription of one window with the decoder, from an `encode` result or from the audio. Returns `{ text, language, method, hypotheses?, tokens, finishReason, cost }`; the text has no "language" prefix and `language` is the code the model wrote (or `null`). Greedy by default, `{ beam: true }` for beam 4, `maxNewTokens`, `signal`. Same engine as `term`: the greedy text is its first hypothesis and the beams are the other four. Without the decoder it throws what `language` and `term` throw; `dispose()` waits for it. The text bundle and the manifest are unchanged; `parseAsr` is exported next to `parseAsrText`. Same limits as the decoder questions: int4 text differs from float32 torch, Portuguese only, one 8 s window, no timestamps.

## 0.2.0 (2026-11-25)

- New entry point `@s1s2/web/media` (`loadMedia`, `AudioModel`, `validateMediaManifest`, ...): audio in, same answers (`predicate`, `choice`) with probability and margin. The text bundle carries none of it.
- Manifest keys `modalities` (default `["text"]`) and `audio` (log-mel preprocess, ONNX tower, readers); `s1s2 validate` checks them. Text manifests are unchanged.
- The safetensors reader accepts `F64` tensors (the audio readers store float64 coefficients).
- Audio packages: `loadMedia` answers voice activity, end of turn, same speaker, speaker change and speaker count, and with the decoder block `language` (closed set, 30 languages) and `term` (a word said, as the fraction of five hypotheses). `s1s2-convert asr-decoder` builds the Qwen3-ASR decoder for it. `loadMedia({ decoder })` is `"auto"` by default: the decoder loads, and if it cannot the model goes on without it (`model.decoder` and `onStatus` say why; only `language` and `term` throw); `false` loads none, `true` fails. `dispose()` waits for questions in flight. Measured on Portuguese only; the 0.02 tolerance on the language probability is not met on a noise clip (0.09 on Node CPU, 0.035 on WebGPU) nor, by 0.0001, on one speech window; `spec/MULTIMODAL.md` section 9 has the tolerances, the numbers and the limits.
- `s1s2 validate` checks `audio.decoder` (model, prompt, language and term settings) and the decoder readers (`decoder-language`, `decoder-term`, one shared tap); text manifests are unchanged.
- Not validated: languages other than Portuguese, for `language` and `term`.

## 0.1.3

- `S1S2.load` picks a runtime for the device: ONNX Runtime Web (GenAI graphs and plain decoders), transformers.js, WebLLM or LiteRT.js in the browser, onnxruntime-node in Node. Builds that list device `"wasm"` run on the browser CPU when WebGPU is absent (`cpuFallback: false` opts out; `model.device` and `model.fallbackReason` say what was used); builds without it are refused with the reason.
- `decide()` answers `predicate`, `choice` and `score` questions in one forward pass and reports `probability`, `confidence`, `margin` and `rawMargin`. `s.generate()` continues from the same prefill. On WebGPU, reuse of a kept state and of the prefix shared by several questions is bit-identical to a full prefill: the genai backend never feeds a prefill block shorter than 32 tokens (`minBlock`).
- Choice by label logits with `logits.choice.map: "letters"` reads the option keys as the model writes them after the cue (" A", " B", ...).
- LiteRT frees every output tensor of a step that is not handed on.
- Packages: manifest `schema: "s1s2.manifest/1"` pins every file by sha256; `s1s2 validate` checks it and rejects any other schema id. The cache directory is `$S1S2_CACHE`.
- Docs, comments and messages describe what was measured in plain words; `test/plain-names.test.mjs` keeps internal names out of shipped files. `PROVISIONAL_NOTE` is `"provisional - no measured threshold"`.

## Converter 0.1.2

- `s1s2-convert asr-decoder` builds the Qwen3-ASR decoder package for `language` and `term` in `@s1s2/web/media`: it pins the checkpoint, writes the CPU build into the same layout as the WebGPU one, and records the term settings and tool versions in the manifest.

## Converter 0.1.1

- `s1s2-convert` trains (`--recipe kev` by default, or `short`), fuses, exports and packs a specialist; the builder patch is `onnxruntime-genai-0.19.0-builder-fixes.patch`.
- `pack-mlc` and `pack-litert` (also `python -m s1s2_convert.pack_mlc|pack_litert`) repackage a published GenAI S1S2 repo as a WebLLM (MLC q4f16_1) or LiteRT.js (int4) repo. The LoRA is merged into the base in bf16 (`fuse_lora.py`), quantized by the runtime's own tool, and one manifest per source manifest is written. WebLLM builds ship a hidden-state recovery matrix (`recovery.rows.i32`, `recovery.P.f16`) so the trained heads work on logits-only WebLLM; `--readout logits` drops the heads. LiteRT builds read the label logits. `pin` writes the upload commit into the manifests and runs `s1s2 validate --strict`.
- `s1s2_convert/node/browser_decide.mjs` loads a packed folder in Chrome (WebGPU) with s1s2 and decides prompts.
- `converter/README.md` has the steps for other runtimes, the pinned tool versions and the architectures each converter handles.
