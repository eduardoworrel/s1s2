# s1s2-convert

Turns a Hugging Face causal LM plus a labelled JSONL into an **S1S2 package** ([spec](../spec/s1s2-package.md))
that s1s2 loads: LoRA + pointer decision head (the recipe behind the GDPR/LGPD/CCPA specialists in the model cards), optional
evidence head (picks the supporting sentence), LoRA fused into the weights, an ONNX Runtime GenAI 0.19 int4 WebGPU graph with `hidden_states`, `head.bin`
files, and a manifest with provenance and license. The output passes `s1s2 validate --strict`.

```
cd converter && uv venv .venv && uv pip install --python .venv/bin/python -e .      # pins in pyproject.toml
(cd .. && npm ci && npm run build)                                                  # the converter calls s1s2
.venv/bin/s1s2-convert all --work /tmp/w --out /tmp/pkg --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --data rows.jsonl --question-template "Does this passage address {requirement}? Answer with one option only."
```

Training and export are memory-heavy; on a small machine run one stage at a time.

**Why this default.** On a three-law privacy benchmark (GDPR, LGPD, CCPA; held-out test, scored once), the joint
specialist trained with the Kev-faithful recipe (`--recipe kev`: 2 epochs, Kev's learning-rate rule, fp32 fusion;
`joint_k` in the model cards) was more accurate than the shorter recipe fixed before any result (`--recipe short`:
1 epoch, lr 5e-5, bf16 fusion) on every law for all five models that had both (e.g. GDPR: Jurema-7B 85.5 → 88.4%, S5-nano 87.1 → 89.4%), and it made
EuroLLM-1.7B (GDPR 53.2 → 83.0%) and Salamandra-2B (59.5 → 81.9%) learn the task, where one epoch left them near chance.
One seed, chosen on the development split; a descriptive comparison, not a hypothesis test fixed in advance; SmolLM2-135M stays near chance with
either recipe. `--recipe` is set at `prepare` (it fixes the epochs in the plan) and later stages refuse a different one. The old
names `joint-k` (= `kev`) and `preregistered` (= `short`) still work as deprecated aliases.

## Input

One JSON object per line: `text` (passage), `question` or `requirement` (filled into `--question-template`), `label`,
optional `evidence` (sentence index or list), `sentences` (default: s1s2 `splitSentences`), `split` (train/dev;
default: seeded `--dev-fraction` split), `id`. The label set (`--labels`, default order of first appearance) becomes the
options; each is shown and read as `" " + label`.

## Stages (resumable, `.done-<stage>` in `--work`)

| Stage | What | Source of the recipe |
|---|---|---|
| prepare | dataset split; prompt layout from the model's chat template (plain-text markers, cue `--cue`); marker/label tokens discovered on every prompt and on s1s2's load-time probe; **all prompts encoded by s1s2 itself** (`s1s2 encode`) and compared with the Python rendering (stops on any difference) | the prompt builder behind the model cards' specialists |
| train | `--recipe kev` (default): 2 epochs, lr 1e-4 if the checkpoint has ≤ 1B parameters else 5e-5 (Kev's rule, LoRA and head). `--recipe short`: 1 epoch, lr 5e-5. Both: LoRA r16/α32/dropout 0.05 on q/k/v/o/gate/up/down (+ DeltaNet projections on hybrids), pointer head dp=256 from scratch, CE over options, AdamW wd 0.01, OneCycle pct 0.1, clip 1.0, batch 4×2, bf16+checkpointing on CUDA (fp32 on CPU), option order re-drawn per epoch, dev-selected. `--epochs`/`--lr` override | reimplemented from the trainer behind the model cards' specialists (follows Kev @2074cc44); `kev` = its Kev-faithful variant (`joint_k`) |
| fuse | W' = W + s·BA on the same keys/files; kev: delta and sum in float64, rounded once to fp32, all tensors and `config.json` in fp32; short: rounded to the base dtype | same as that trainer's merge step (its fp32 path for `kev`) |
| evidence | head only on the fused model, init from the decision head, lr 1e-4, batch 32, ≤8 epochs, multi-target loss, epoch by dev loss | reimplemented from the trainer of the citation heads |
| build | `builder.py -p int4 -e webgpu --extra_options include_hidden_states=true exclude_mtp=true` from a copy of pip `onnxruntime-genai==0.19.0` patched with `s1s2_convert/patches/onnxruntime-genai-0.19.0-builder-fixes.patch` (or `--builder-src` a checkout). `--precision int8/fp16/fp32` for more builds | the builds in the model cards |
| package | files + manifest (`genai-<prec>-webgpu` and `-cpu` per graph, heads, `evidence`, `tasks` when the question is fixed, `license`, `provenance` with `source.format`); head.bin headers state `"scale": 1` + `s1s2 validate --strict` | `head.py` = port of the Kev PointerHead (Apache-2.0, see NOTICE) |
| parity | dev rows: PyTorch fp32 on the fused weights vs s1s2 `S1S2.load` on a build (onnxruntime-node CPU); writes the build's `fidelity` (spec 7) and `provenance.parity.<build>` | |

**Builder patch** (shipped in `s1s2_convert/patches/`): two commits on top of onnxruntime-genai 0.19.0 (`d01b2441`),
files builder.py + builders/: `38cbf9c3` SmolLM3 NoPE layers (0.19.0 raises KeyError on `position_ids`) and `b4bb8aec`
`linear_attn_op=linear_attention` for dense Qwen3.5 CUDA/WebGPU exports. Applies cleanly to the pip 0.19.0 builder.

Declared differences from the lab recipe: no kev data augmentation (none-of-the-above / distractors), no later kev
stages, no temperature calibration, no resumable mid-training checkpoints, one label set per package (the lab's joint
specialist mixes kev multi-option data and two languages in one adapter). `--head-lr` (kev's `head_lr`) is an addition;
the default is the lab's (same lr). No margin threshold is written: calibrate one on your build with
`thresholdFromQuantile` over its own dev margins (spec section 8).

## What is tested (2026-11-15, Mac, CPU)

End-to-end on `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f7` with the toy set `examples/toy.jsonl` (400 rows from
`examples/make_toy.py`: "does this passage address a refund?", Yes/No, evidence sentence on Yes rows; 360 train / 40
dev; 8 epochs, `--lr 1e-4 --head-lr 1e-3`; CPU training ~9 min). The package itself is not published; its manifest is in `spec/examples/`. Its head.bin files predate the `"scale"` header key and are accepted through `scale_folded_into_q` (spec 6.1); new runs write both.

* prepare: the manifest's encoding / markers / logits / generation / eos / warmup are **identical** to the lab's own
  SmolLM2 manifest (the one used on the privacy benchmark);
  s1s2 encoding == Python rendering on all 3120 prompts.
* `s1s2 validate --strict` on the package: ok (13 files sha256-checked, heads parsed).
* s1s2 loads it (`S1S2.load`, build `genai-*-cpu`, onnxruntime-node CPU, heads `choice` + `evidence`) and answers
  every dev row with `method: "both"` + `evidence`. Parity vs PyTorch fp32 (40 decision rows, 19 evidence rows):

| build | head agreement | head acc (PyTorch → lib) | logits agreement | evidence agreement | evidence acc |
|---|---|---|---|---|---|
| genai-fp32-cpu | 40/40, max \|Δp\| 0.000 | 1.00 → 1.00 | 40/40 | 19/19 | 0.79 → 0.79 |
| genai-int8-cpu | 40/40, max \|Δp\| 0.021, margin r 0.97 | 1.00 → 1.00 | 40/40 | 15/19 | 0.79 → 0.89 |
| genai-int4-cpu | 24/40, margin r −0.12 | 1.00 → **0.60** | 37/40 | 5/19 | 0.79 → **0.26** |

  The fp32 graph is exact, so the pipeline (layout, positions, head export, fusion, hidden-state output) is right; the
  int4 loss on this 135M model is consistent with quantization (the fused LoRA delta is ~0.15% of the weight norm); the cause was not measured further. For tiny models ship/escalate to int8; int4 fidelity for a given model must be measured, not assumed.
* The fork-checkout path (`--builder-src <fork>/src/python/py/models` at `38cbf9c3`) produced the
  same `model.onnx.data` (weights) bytes as the patched-pip path; `model.onnx` (graph proto, 214 KB) differs and the difference was not inspected.
* **Default recipe, one command** (2026-11-18, Mac, CPU; same model, same toy file, `--dev-fraction 0.1`, `--precision int4,int8,fp32`, no other option; 3 min):
  2 epochs at lr 1e-4 (90 steps) did not teach this 135M model the toy decision (dev accuracy 0.65 at step 0, 0.525 at the end); the evidence head
  reached 1.00 on its 19 dev rows. The pipeline itself held on the five builds (`genai-{int4,int8}-{webgpu,cpu}`, `genai-fp32-cpu`), each CPU build with a
  `fidelity` record: fp32 40/40 head decisions, max |Δp| 0.000; int8 40/40 (0.026); int4 40/40 (0.073), logits 34/40, evidence 17/19. These are agreements of a near-chance model.
  `--precision` takes a list and `parity` runs on every `-cpu` build by default. Python 3.14 refuses the 0.19.0 builder's help text ("N % 64"); the converter now escapes it in its copy.
* Unit tests: `python -m unittest discover -s tests` (fill/escape/render, data loading, head.bin vs lib formula).

## Transformers.js layout of a published package (`pack_tjs`)

`python -m s1s2_convert.pack_tjs pack|parity|card|publish|update` turns a published ONNX Runtime GenAI S1S2 repo into a
Transformers.js-loadable folder with the same weights (data file byte for byte): `onnx/model_q4.onnx` (KV head dim pinned,
external-data location renamed), `config.json` of the base + `transformers.js_config`, tokenizer, heads, and one manifest
per source manifest with a single `transformers.js` build. Builds made with `exclude_embeds` get the token table back as
`onnx/model_q4.onnx_data_1` (same file) and one Gather node. `parity` loads the source GenAI graph and the new one with
s1s2 on Node CPU (one at a time) and compares the hidden state at every read position and the head decisions; the result
goes into the build's `fidelity`. The prompts used for the published repos come from `examples/pack_tjs_items.py`
(hand-written texts in each manifest's own templates; no dataset rows). The published `*-ONNX-transformersjs-S1S2` repos were made with it.
`hydrate` rebuilds the card/manifest inputs from a published repo (no weights) so `update` can rewrite cards later. The
cards' "Without s1s2" code is `s1s2_convert/node/tjs_plain_decide.js`; `examples/tjs_checks/` holds the checks behind
them (that code vs s1s2 on Node CPU, transformers.js#1943 `pipeline("decision")` vs the head, s1s2 on the browser CPU)
and their results, `tjs_card_checks.json`, passed to `card`/`update` with `--checks`.

## Other runtimes (WebLLM, LiteRT)

`s1s2-convert pack-mlc` and `s1s2-convert pack-litert` (also `python -m s1s2_convert.pack_mlc|pack_litert`) turn a published
ONNX Runtime GenAI S1S2 repo into a WebLLM (MLC q4f16_1) or a LiteRT.js (int4) repo for the same specialist. Both start from the
base model plus the published LoRA (merged in bf16, `W' = W + alpha/r * B @ A`, the same weights the GenAI build was made from), quantize
them again to 4 bits with the runtime's own tool, and write one manifest per source manifest. The new build is a different
quantization of the same weights, not a copy of the GenAI graph: its decisions agree with the GenAI build's on most prompts, not all
(the published cards give the numbers they were measured with; `--readout` below says what each readout is).

**Readout.** WebLLM and LiteRT return logits, not the hidden state the trained heads read.
* WebLLM, `--readout head` (default): the heads work. For an output projection `W` (the tied embedding or `lm_head`, read from the quantized
  weights and dequantized), `logits = W h`, so `h = P . logits[rows]` with `P = (W_n^T W_n)^-1 W_n^T` over `--recovery-rows` (8192) fixed random
  vocabulary rows. The rows and `P` (f16, `[hidden, rows]` row-major) ship as `recovery.rows.i32` and `recovery.P.f16`; s1s2's WebLLM backend
  applies them (`src/backends/webllm.ts`, `recoverHidden`), costing one extra prefill per read position. `--readout logits`: no recovery files, no
  heads, every question reads the label logits.
* LiteRT: logits only (the graph has no hidden-state output), so the manifests carry no heads and read the label logits. Use manifest tasks (`type: "choice"`), which
  keep the option order; free-form letters are read as " A", " B", ... (s1s2 0.1.3 and later).

### WebLLM (MLC q4f16_1)

WebLLM runs a compiled WebGPU library, `model-lib.wasm`, that fixes the architecture and every tensor shape; the weights are data. A fine-tune of a base
that already has a prebuilt library therefore reuses it (`--prebuilt-lib`); any other shape needs `compile` (below).

```
uv venv -p 3.12 mlc && . mlc/bin/activate
uv pip install --pre -f https://mlc.ai/wheels "mlc-llm-nightly-cpu==0.26.dev6" "mlc-ai-nightly-cpu==0.26.dev246" \
    "apache-tvm-ffi==0.1.13" "numpy==2.2.6" torch==2.15.0 transformers==5.23.0 safetensors huggingface_hub \
    psutil pydantic fastapi uvicorn shortuuid requests tqdm ml_dtypes cloudpickle tornado prompt_toolkit openai tiktoken sentencepiece protobuf
# macOS only: the wheels' dylibs arrive with invalid signatures (dyld kills the process, "Code Signature Invalid"); sign the ones that fail
find mlc/lib -name '*.dylib' -o -name '*.so' | while read f; do codesign -v "$f" 2>/dev/null || codesign --force -s - "$f"; done
python -m s1s2_convert.pack_mlc pack --repo you/Qwen3-0.6B-Task-ONNX-genai-int4-S1S2 --out /tmp/mlc \
    --lora you/Qwen3-0.6B-Task-LoRA --lora-revision <40-hex> \
    --prebuilt-lib v0_2_86/base/Qwen3-0.6B-q4f16_1_cs1k-webgpu.wasm --conv-template qwen2
# upload /tmp/mlc as a new repo (everything in it), then pin the manifests to that commit and validate:
python -m s1s2_convert.pack_mlc pin --out /tmp/mlc --revision <40-hex of the upload commit>
```

`pack` writes `mlc-chat-config.json`, `params_shard_*.bin`, `tensor-cache.json` (+ `ndarray-cache.json`, the name web-llm fetches), tokenizer files,
`model-lib.wasm`, the recovery files, the heads, and the manifests; it needs the pinned versions above (wheel sha256:
`mlc_ai_nightly_cpu-0.26.dev246-py3-none-macosx_26_0_arm64.whl` 935d2a55…26cc, `mlc_llm_nightly_cpu-0.26.dev6-py3-none-macosx_26_0_arm64.whl` cab4ad76…8b66).
`--prebuilt-lib` paths are under `web-llm-models/` of mlc-ai/binary-mlc-llm-libs at the pinned commit `5b7251f3bdce` (Apache-2.0); use the
folder of your `@mlc-ai/web-llm` release (`v0_2_86` for 0.2.86, whose `base/` libraries ship `cs1k` = prefill chunk 1024; `--prefill-chunk-size 1024
--context-window-size 4096` match them). `compile` runs `mlc_llm compile --device webgpu` for a model with no prebuilt library; it needs emscripten (`emcc`)
and `MLC_LLM_SOURCE_DIR` pointing at an mlc-llm checkout prepared with `web/prep_emcc_deps.sh` (`web/dist/wasm/mlc_wasm_runtime.bc`,
`mlc_llm/support/auto_target.py`, `_build_webgpu`), and the library must come from the same MLC/TVM revision as the web-llm release that loads it.
**`compile` was not run here.**

Known problems with the pinned wheels (macOS 26, Apple silicon, Python 3.12.15, 2026-11-18): with `apache-tvm-ffi` 0.1.14-0.1.16 `relax.build` (so
`convert_weight`) crashes with a segmentation fault in `src/ffi/extra/module.cc`; 0.1.13 (yanked on PyPI, still installable) works. `numpy` 2.3 and later
makes `convert_weight` fail with "Could not convert T.float16 to a NumPy dtype". Both pins are in the command above.

### LiteRT.js (int4)

```
python3 -m venv litert && . litert/bin/activate
pip install torch==2.14.0 transformers==5.23.0 numpy==2.5.4 safetensors huggingface_hub \
    ai-edge-litert-nightly==2.4.0.dev20261114 litert-converter==0.6.0.dev20261113 litert-quantizer-nightly==0.12.0.dev20261115 \
    litert-lm-builder-nightly==0.20.0.dev20261114 "git+https://github.com/eduardoworrel/litert-torch@afb8755f1e362940d1e80546096624697279c563"
python -m s1s2_convert.pack_litert pack --repo you/Qwen3-0.6B-Task-ONNX-genai-int4-S1S2 --out /tmp/litert \
    --lora you/Qwen3-0.6B-Task-LoRA --lora-revision <40-hex>
python -m s1s2_convert.pack_litert pin --out /tmp/litert --revision <40-hex of the upload commit>
```

The export is `litert_torch.generative.export_hf` with the recipe in `s1s2_convert/data/litert_int4_block32_fc.json` (int4, block 32, fully
connected layers), external embedder, prefill 16, KV cache 2,048 (`--cache-length`; the longest prompt is 8 less), no GPU composites, no prefill logits.
`litert/model.graph.tflite` + `litert/model.weights.bin` come from LiteRT's `externalize_tflite_flatbuffer.py` (Apache-2.0, Google; kept with its header
in `s1s2_convert/tools/`), `litert/embed.f16.bin` holds the token rows as float16 and `litert/manifest.json` the signatures. The export is CPU-only and
memory-heavy: a 1.7B model needs about 64 GB of RAM (not run here); the 0.6B export ran on a 16 GB Mac.

### Browser check

`s1s2_convert/node/browser_decide.mjs` loads a packed folder in Chrome (WebGPU, through Playwright) with s1s2 and decides prompts from a JSONL (the same rows
as `pack_tjs parity`: `examples/pack_tjs_items.py`), with the head readout and the logits readout. Use it before uploading; the files are served locally:
`node browser_decide.mjs --lib <s1s2 checkout> --folder /tmp/mlc --rows rows.jsonl --out out.json --modules <node_modules with @mlc-ai/web-llm and @litertjs/core>`
(pin the manifests to any 40-hex revision first). Parity with the GenAI build is yours to measure with the same rows: run the GenAI folder through
`pack_tjs parity`'s Node side, or compare the head/logits decisions of both builds.

### Which architectures each converter handles

Rows are from the tools' own source or registries, read on 2026-11-18; "built" means a build was run through the converter and loaded in s1s2.

| Architecture (examples) | WebLLM / MLC | LiteRT |
|---|---|---|
| Qwen3 (0.6B, 1.7B) | model type `qwen3` (`mlc_llm/model/model.py`, wheel 0.26.dev6); prebuilt libraries `Qwen3-0.6B/1.7B-q4f16_1_cs1k` in `v0_2_86/base`. **Built** (0.6B: loaded in Chrome, head and logits decisions) | generic export path (`export_hf/export.py`: a checkpoint without a registered extension takes the generic path); **built** (0.6B: loaded in Chrome, label-logits decisions) |
| Qwen2 (Jurema-7B, Qwen2.5-based) | model type `qwen2`; prebuilt `Qwen2-7B-Instruct-q4f16_1_cs1k` (`v0_2_86/base`) has the same `hidden_size`, layers, heads, KV heads, intermediate size, vocabulary, `rope_theta`, `rms_norm_eps`, tying and sliding-window fields as Jurema-7B's `config.json`, so it applies. Not built here (about 4.5 GB q4f16, 7B) | generic path; not built |
| Llama family (EuroLLM-1.7B, Salamandra-2B) | model type `llama`; no prebuilt library has these shapes (2048 hidden, 24 layers, vocabularies of 128,000 and 256,000; the Llama-3.2 libraries are 1B and 3B), so `compile` is needed. Untied `lm_head`: the recovery reads it. Not built | generic path; not built |
| SmolLM3 (3B) | no `smollm3` model type in wheel 0.26.dev6; `python/mlc_llm/model/smollm3` is in the mlc-llm repository since commit bb208e51f6 (#3578, 2026-11-16) and a prebuilt library `SmolLM3-3B-q4f16_1_cs1k` exists in `v0_2_87/base`. Using the repository's model code needs a matching newer TVM: importing it against the wheel pair fails (`No module named 'tvm_tirx'`). Not built | generic path; not built |
| Qwen3.5 hybrid GatedDeltaNet (S5-nano) | model types `qwen3_5` / `qwen3_5_text` in the wheel; S5-nano's `text_config` equals Qwen3.5-4B's on every shape field (hidden 2560, 32 layers, 16/4 heads, 9216, vocabulary 248,320, linear-attention fields), so the prebuilt `Qwen3.5-4B-q4f16_1_cs1k` (`v0_2_86/base`) applies. The base checkpoint nests weights under `model.language_model.`, which `fuse_lora` handles. Not built | `qwen3_5` has its own exportable module (`export_hf/core/exportable_module.py`); not built |

## Not tested

* WebLLM and LiteRT builds of any model other than Qwen3-0.6B (end to end: convert, load in Chrome, decide a handful of prompts; no parity run against the GenAI build); `pack_mlc compile`; the Llama, Qwen2, SmolLM3 and Qwen3.5 rows of the table above are source reads, not builds; LiteRT on a 1.7B model.
* WebGPU (browser) execution of the produced package (only onnxruntime-node CPU); CUDA training path (`--device cuda`).
* Any model other than SmolLM2-135M; the effect of the builder fixes themselves (SmolLM3 NoPE, Qwen3.5 linear
  attention) — they are applied, not exercised (SmolLM2 is Llama).
* Full-size reproduction of the lab's SmolLM3-3B specialist (needs a GPU); not run.
* Multi-token labels (the logits readout then falls back to `first-token`), score/predicate tasks, kev encodings.
