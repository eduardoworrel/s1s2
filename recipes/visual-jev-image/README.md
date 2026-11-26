# Visual Jev image reader: build and parity recipe (not run on the 4B yet)

Design and tolerances: `spec/MULTIMODAL.md` section 10. Everything below the line "needs the weights" has been written and tested only on synthetic graphs and on range reads of the public files.

Sources (pin the commits you use; sizes are the Hub's):

| what | where | size |
|---|---|---|
| base | `Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0), two bf16 shards | 8.88 GB |
| adapter | `guanxuyu/visual-jev-4b-answer-sft` (Apache-2.0), `adapter_model.safetensors` + `adapter_config.json` | 132 MB |
| public int4 build | `onnx-community/Qwen3-VL-4B-Instruct-ONNX`, `onnxruntime/cuda/cuda-int4-rtn-block-32` (fp16 io) or `onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-32` (fp32 io): `text.onnx(.data)`, `vision.onnx(.data)`, `embedding.onnx(.data)` | cuda 2.27 + 0.67 + 0.78 GB; cpu 2.52 + 0.18 + 0.21 GB |

```bash
cd converter && pip install -e .            # torch, transformers, peft, safetensors, onnx, onnxruntime (versions in pyproject.toml)

# 1. oracle: the quantizer against the public text graph, no LoRA (byte for byte; the report says oracle true/false and which cast chain)
s1s2-convert vl-swap --template PUBLIC/cuda-int4-rtn-block-32 --base BASE --verify --report verify.json

# 2. LoRA fused into the 252 matrices, int4 RTN block 32, written in place; external data split into pieces of at most 1.9 GiB
s1s2-convert vl-swap --template PUBLIC/cuda-int4-rtn-block-32 --base BASE --adapter ADAPTER/adapter_model.safetensors --out BUILD --split --report swap.json
cp PUBLIC/cuda-int4-rtn-block-32/{vision.onnx,vision.onnx.data} BUILD/

# 3. graph edits: DeepStack inputs + final-norm hidden output (text), DeepStack outputs (vision)
s1s2-convert vl-graph text   --in BUILD/text.onnx   --out BUILD/text.onnx   --deepstack --hidden-output
s1s2-convert vl-graph vision --in BUILD/vision.onnx --out BUILD/vision.onnx --base BASE
cp PUBLIC/cuda-int4-rtn-block-32/embedding.onnx.data BUILD/embedding.fp16       # float16 [151936, 2560], no header (offset 0)
```

Needs the weights, run elsewhere (about 18 GB for the float32 oracle):

```bash
python torch_reference.py --questions q.jsonl --out ref.jsonl                    # PyTorch float32 + adapter, one independent forward per question
python ort_reader.py --dir BUILD --questions q.jsonl --out ort.jsonl             # the built graphs, the pipeline of section 10.5
python torch_reference.py --questions q.jsonl --out ref_base.jsonl --no-adapter  # the same backbone without the adapter (what the adapter changes)
```

Compare `probabilities` per question with the tolerances of section 10.7 (median and 95th percentile of the absolute difference of the chosen option's probability, decision flips, accuracy). The questions must be items nobody
tuned anything on. Run the public graphs without the step 3 edits too (route B of section 10.2) and report that difference; do not describe that build as the paper's model.
