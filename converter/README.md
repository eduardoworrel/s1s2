# s1s2-convert

Turns a Hugging Face causal LM plus a labelled JSONL into an **S1S2 package** ([spec](../spec/s1s2-package.md))
that s1s2 loads: LoRA + pointer decision head (lab-s12 study 001 recipe), optional evidence head (study 005
recipe), LoRA fused into the weights, an ONNX Runtime GenAI 0.19 int4 WebGPU graph with `hidden_states`, `head.bin`
files, and a manifest with provenance and license. The output passes `s1s2 validate --strict`.

```
cd converter && uv venv .venv && uv pip install --python .venv/bin/python -e .      # pins in pyproject.toml
(cd .. && npm ci && npm run build)                                                  # the converter calls s1s2
.venv/bin/s1s2-convert all --work /tmp/w --out /tmp/pkg --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --data rows.jsonl --question-template "Does this passage address {requirement}? Answer with one option only."
```

Training and export are memory-heavy; on a small machine run one stage at a time.

## Input

One JSON object per line: `text` (passage), `question` or `requirement` (filled into `--question-template`), `label`,
optional `evidence` (sentence index or list), `sentences` (default: s1s2 `splitSentences`), `split` (train/dev;
default: seeded `--dev-fraction` split), `id`. The label set (`--labels`, default order of first appearance) becomes the
options; each is shown and read as `" " + label`.

## Stages (resumable, `.done-<stage>` in `--work`)

| Stage | What | Source of the recipe |
|---|---|---|
| prepare | dataset split; prompt layout from the model's chat template (plain-text markers, cue `--cue`); marker/label tokens discovered on every prompt and on s1s2's load-time probe; **all prompts encoded by s1s2 itself** (`s1s2 encode`) and compared with the Python rendering (stops on any difference) | lab-s12 prompt builder |
| train | LoRA r16/α32/dropout 0.05 on q/k/v/o/gate/up/down (+ DeltaNet projections on hybrids), pointer head dp=256 from scratch, CE over options, AdamW wd 0.01, OneCycle pct 0.1, clip 1.0, batch 4×2, bf16+checkpointing on CUDA (fp32 on CPU), option order re-drawn per epoch, dev-selected | reimplemented from the lab-s12 study 001 trainer (follows Kev @2074cc44) |
| fuse | W' = W + s·BA on the same keys/files | same as the study 001 trainer's merge step |
| evidence | head only on the fused model, init from the decision head, lr 1e-4, batch 32, ≤8 epochs, multi-target loss, epoch by dev loss | reimplemented from the lab-s12 study 005 trainer |
| build | `builder.py -p int4 -e webgpu --extra_options include_hidden_states=true exclude_mtp=true` from a copy of pip `onnxruntime-genai==0.19.0` patched with `s1s2_convert/patches/onnxruntime-genai-0.19.0-lab-s12-fixes.patch` (or `--builder-src` a checkout). `--precision int8/fp16/fp32` for more builds | lab-s12 builds |
| package | files + manifest (`genai-<prec>-webgpu` and `-cpu` per graph, heads, `evidence`, `tasks` when the question is fixed, `license`, `provenance` with `source.format`); head.bin headers state `"scale": 1` + `s1s2 validate --strict` | `head.py` = port of the Kev PointerHead (Apache-2.0, see NOTICE) |
| parity | dev rows: PyTorch fp32 on the fused weights vs s1s2 `S12.load` on a build (onnxruntime-node CPU); writes the build's `fidelity` (spec 7) and `provenance.parity.<build>` | |

**Builder patch** = `git diff d01b2441..lab-s12/builder-0.19.0-fixes` of github.com/eduardoworrel/onnxruntime-genai
(builder.py + builders/): `38cbf9c3` SmolLM3 NoPE layers (0.19.0 raises KeyError on `position_ids`) and `b4bb8aec`
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
  SmolLM2 manifest (lab-s12 study 001);
  s1s2 encoding == Python rendering on all 3120 prompts.
* `s1s2 validate --strict` on the package: ok (13 files sha256-checked, heads parsed).
* s1s2 loads it (`S12.load`, build `genai-*-cpu`, onnxruntime-node CPU, heads `choice` + `evidence`) and answers
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
* Unit tests: `python -m unittest discover -s tests` (fill/escape/render, data loading, head.bin vs lib formula).

## Not tested

* WebGPU (browser) execution of the produced package (only onnxruntime-node CPU); CUDA training path (`--device cuda`).
* Any model other than SmolLM2-135M; the effect of the builder fixes themselves (SmolLM3 NoPE, Qwen3.5 linear
  attention) — they are applied, not exercised (SmolLM2 is Llama).
* Full-size reproduction of the lab's SmolLM3-3B specialist (needs a GPU); not run.
* Multi-token labels (the logits readout then falls back to `first-token`), score/predicate tasks, kev encodings.
