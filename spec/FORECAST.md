# Forecast reader (experimental)

Status: experimental, added after `s1s2.manifest/1`; the schema id does not change. Everything below is optional: a
manifest without `forecast` behaves exactly as before.

## What it is

A second kind of reader over the **same prefill** that `decide()` already runs. A small head (linear or one-hidden-layer
MLP, with temperature and Platt calibration) reads the hidden state of an **intermediate decoder layer** at the end of the
context and returns, per behavior, the probability that the *answer* the model has not written yet will show that
behavior (for example "refuses", "leaves the topic", "passes a verifier"). Nothing is generated.

```
decide(...)    System 1, as before (logits / head on the final hidden state)
forecast(...)  reads layer N of the same prefill -> P(behavior of the answer)
generate(...)  System 2, continues from the kept prefill (no second prefill)
```

Intended use (hypotheses, not measured here): a gate before System 2 (skip, reroute or escalate before spending
generation), decisions that never leave the device, and reuse of a prefill that was needed anyway.

## Reference and what this implements

Reference: He and Jiralerspong (Tufa Labs), *Context-Answer Metamodels: Forecasting LLM Activations Before the First
Answer Token*, arXiv 2611.06837 (4 Nov 2026, v2 19 Nov 2026). They train metamodels that predict the mean activation of
the answer from the activation at the last context token (a ridge regression on layer 18 of Qwen3.5-9B; an MLP and a flow
on a 32-bin trace of the answer), and read behaviors from the prediction with linear probes.

This library implements only the **reading**: a head on a mid-layer activation of the context, in the browser and Node.
Differences, stated so nothing is read into them:

- The paper forecasts the answer's activations and then probes them. Here the head maps the context activation straight
  to behavior probabilities (one step, no activation forecast, no flow model).
- No result of the paper (R^2, recovery of behaviors) is reproduced or claimed. No accuracy of this reader has been
  measured on a real task; the end-to-end check in this repository is a smoke test of the mechanism (see below).
- The paper's behavior labels come from an LLM judge, not validated by humans. This library ships no judge and no labels;
  labels are supplied by whoever trains a head.
- The authors' repository (Tufalabs/context-answer-metamodels) has no license; this is an independent implementation from
  the paper's description, no code copied.
- `pooling: "bins32"` is this library's own input variant (32 equal bins over the context tokens), not the paper's 32-bin
  answer trace. It is untested beyond unit tests of the arithmetic.

## Manifest

```jsonc
{
  "layer": { "read": "final", "norm": "in-graph" },     // "read": "final" | integer (see below)
  "forecast": {
    "id": "refusal-gate-v1",
    "layer": 14,                  // 0-indexed decoder block whose OUTPUT the head reads; 0 <= layer <= numLayers-2
    "pooling": "last",            // "last" | "mean" | "bins32"
    "head": {
      "type": "linear",           // "linear" | "mlp"
      "file": { "path": "forecast.safetensors", "sha256": "...", "size": 1234 },
      "repo": "...", "revision": "...",           // optional; default: the loaded build's repo and revision
      "dims": { "in": 576, "out": 2 }             // mlp: also "hidden"
    },
    "behaviors": [ { "name": "list", "description": "answer has a numbered list" }, ... ],
    "calibration": { "temperature": 1.0, "platt": { "a": 1.0, "b": 0.0 } },   // numbers or one number per behavior
    "trainedOn": { "baseId": "<manifest id>", "weightsSha256": "...", "loraSha256": "..." }
  },
  "builds": { "genai-fp32-cpu": { ..., "forecastOutput": "aux_hidden_states", "forecastLayer": 14 } }
}
```

- `layer.read` accepts `"final"` or an integer. The integer states which layer the S1 heads of the package read. This
  runtime reads S1 on the final hidden state only, so a manifest with an integer `layer.read` validates and is **refused at
  load** with a message saying so. The forecast layer is separate (`forecast.layer`) and does not change S1.
- `builds.<name>.forecastOutput` names the graph output carrying the residual stream after block `forecastLayer`, width
  `hiddenSize`; both keys appear together. Only `genai`, `decode-only` and `transformers.js` builds can declare them.
- Checks at load: the build's `forecastLayer` equals `forecast.layer`; the head file hash (sha256, as every pinned file);
  `trainedOn.baseId` equals the manifest `id`; `dims.in` equals `hiddenSize` (`last`, `mean`) or `32 * hiddenSize`
  (`bins32`); `dims.out` equals `behaviors.length`; the tensors in the file have those shapes. Each failure is an error naming
  both values. `trainedOn.weightsSha256` and `loraSha256` are provenance, not checked at load: one head must load on the
  fp32 and the int4 build of the same weights.

## Head file (safetensors, fp32)

| tensor | shape | |
|---|---|---|
| `mean`, `std` | `[in]` | training normalisation, `x = (pooled - mean) / std` (`std` > 0) |
| `weight`, `bias` | `[out, in]`, `[out]` | linear |
| `w1`, `b1`, `w2`, `b2` | `[hidden, in]`, `[hidden]`, `[out, hidden]`, `[out]` | mlp: `z = w2 relu(w1 x + b1) + b2` |

Order of operations per behavior `k`: `x = (pooled - mean) / std`; `z = head(x)`; `z = z / T_k`; `logit = a_k z + b_k`
(only when `platt` is given); `p = sigmoid(logit)`. Behaviors are independent (one sigmoid each, not a softmax).

Pooling over the context tokens `0..n-1` of the decide prompt (`n` = last index + 1): `last` takes token `n-1`; `mean` the
mean over all tokens; `bins32` splits the tokens into 32 bins `[floor(j n / 32), floor((j+1) n / 32))`, takes the mean of each
and concatenates them bin-major (needs `n >= 32`).

## API

```ts
const s = await decide(req, { forecast: true });       // same prefill as always, plus one extra graph output
const f = s.forecast();                                 // no backend call
// f = { behaviors: { list: { p, logit }, ... }, layer, pooling, question, cost: { prefillTokens, extraPrefillTokens: 0, headMs } }
if (f.behaviors.refuse.p < 0.5) await s.generate(prompt); // continues from the kept prefill

const { forecast, session } = await model.forecast(req);  // decide({forecast:true}) + session.forecast()
```

`forecast` is off unless asked (`decide(..., { forecast: true })`, `model.forecast`), so S1 costs and results are unchanged.
The read position of `last` pooling is the last token of the decide prompt (`enc.last`), so a head must be trained on the
prompts that `s1s2 encode` produces for the same manifest and question, and its labels must describe what the model does
after that prompt (the System 2 continuation this gate protects). `session.forecast({ question })` picks which question's
prompt; the default is the first. A build without `forecastOutput` (WebLLM, LiteRT, or a graph exported without the layer)
throws `forecast unsupported: ...` naming the reason.

## Converter

```
python -m s1s2_convert build ... --forecast-layer 14      # extra_options aux_hidden_state_layers=15 (see below)
python -m s1s2_convert train-forecast-head --model <fused dir> --prompts p.jsonl --labels l.jsonl --layer 14 --out dir
```

`train-forecast-head` runs offline: it generates the model's answers (greedy) for each prompt, joins behavior labels you
provide (`{"id", "labels": {"<behavior>": 0|1}}` per line; no judge, no labels are built in), standardises features,
fits a logistic regression or a small MLP, fits temperature (and optionally Platt) on a held-out split, and writes the
safetensors file and the `forecast` manifest block with sha256 values. With a fused LoRA the head must be trained on the
**fused** model; a head trained on the base does not transfer.

## Backend support (checked 2026-11-24)

Question: can onnxruntime-genai expose the hidden state of an intermediate layer?

- **Builder: yes.** `onnxruntime_genai` 0.19.0 (`__commit__` 331d6ebb6fab31e4a3bc04c2b873457181813408), `models/builder.py`
  lines 1088-1099 document `include_hidden_states` (final-norm `hidden_states` next to `logits`), `exclude_lm_head` and
  `aux_hidden_state_layers`: "Comma-separated decoder layer indices whose incoming residual streams are concatenated into
  an extra `aux_hidden_states` output ... Entry `i` is the residual stream entering layer i, so indices must lie in
  [1, num_hidden_layers)". `models/builders/base.py` lines 663-706 (option parsing, output shape `hidden_size * count`),
  3702-3736 (the tap is the skip-layernorm sum of layer `i`) and 7454-7474 (`Concat` into the output).
- **Sources.** Canonical locations of the files above at that commit:
  `https://github.com/microsoft/onnxruntime-genai/blob/331d6ebb9fab31e4a3bc04c2b873457181813408/src/python/py/models/builder.py` and
  `.../src/python/py/models/builders/base.py`. They were not reachable from the machine used (see below); the pip wheel
  files were read instead, so the line numbers are those of the wheel.
- **Mapping.** Block `L`'s output (manifest `forecast.layer = L`, 0-indexed) is the input of block `L+1`, so the builder gets
  `aux_hidden_state_layers=L+1` and the reference is Hugging Face `hidden_states[L+1]`; valid for `0 <= L <= numLayers-2`
  (the last block's output is only reachable after the final norm, as `hidden_states`). Pass one index: several are
  concatenated.
- **Checked on this machine:** SmolLM2-135M-Instruct, fp32 CPU build with `aux_hidden_state_layers=15`: output
  `aux_hidden_states` `[1, n, 576]`; against `transformers` `hidden_states[15]` max |diff| 0.041 where values reach 2.7e4
  (fp32 reorder noise). The state is the raw residual stream before any norm, in the graph's io dtype: on fp16 (WebGPU)
  builds large residual values can overflow, so the backend rejects non-finite output with a clear error. (On SmolLM2-135M at layer 14 the rows reached |x| of about 71, so this did not happen in the smoke test below.)
- **Runtime API: not verified.** `Generator::GetOutput` and the JavaScript API could not be read: every request to
  `api.github.com` and `raw.githubusercontent.com` for `microsoft/onnxruntime-genai` (and `microsoft/onnxruntime`) returned
  HTTP 404 from this machine, and the npm registry has no `onnxruntime-genai` package. What is visible in the installed
  wheel: `genai_config.json` lists only `logits` and `present_*` under `model.decoder.outputs` (not even `hidden_states`),
  the shared library names `aux_hidden_states` only in its speculative-decoding (`dflash2`) wiring, and contains the message
  "Unexpected call to ModelIO::GetOutput, this function is not implemented for ModelIO".
- **Conclusion.** s1s2 does not use the `Generator` API: `src/backends/genai.ts` runs the builder's graph with a plain
  ONNX Runtime `InferenceSession.run(feeds, fetches)` and fetches `hidden_states` that way today. The extra output is
  fetched the same way, so the `genai` backend supports forecast on graphs exported with the layer. `decode-only` and
  `transformers.js` get the same fetch-by-name. `webllm` and `litert` have no such output and throw
  `forecast unsupported`.

## Smoke test (2026-11-24, Mac CPU; a mechanism check, not a result)

`converter/examples/forecast_smoke.py` + `test/smoke/forecast_parity.mjs`, on SmolLM2-135M-Instruct (no LoRA), layer 14 (of 30),
300 prompts built from 25 topics x 12 templates, greedy answers of 48 tokens, two labels defined by the person running it
(`numbered_list`: the answer has a line starting `1.`-style; `long_answer`: more than 40 words), a logistic head with
temperature and Platt fitted on a held-out quarter of those same rows. What was checked:

- The graph exports the layer: fp32 CPU, int8 and int4 (WebGPU graph run on Node CPU) builds each carry `aux_hidden_states`.
- The library, on Node CPU, reads it in the decide prefill (0 extra prefill tokens) and the head output of the fp32 build
  equals the PyTorch reference on 100 prompts: max |dp| 2e-5, 0 flips of 200 prompt x behavior pairs. This confirms the layer
  index mapping, the pooling and the head formula end to end.
- Across precisions, against the fp32 build and the default tolerance of `forecastParity` (max |dp| 0.05, flips <= 5%, written
  before the run): int8 max |dp| 0.35, mean 0.033, 3 of 200 flips (fails the max |dp| bound); int4 max |dp| 1.0, mean 0.23, 49 of
  200 flips (fails both). The int4 layer row differs from fp32 by about 27% relative L2 (cosine >= 0.96). So on this model a
  head trained on fp32 activations does not carry over to the quantised builds; whether a head trained on the quantised
  activations does was not tried.

It says nothing about whether the forecasts are accurate: the labels are trivial text rules, the same rows trained and
calibrated the head, and no held-out accuracy is claimed.

## Limits and open questions

- No measured accuracy; the smoke test shows the mechanism runs and the head output survives int4 vs fp32, nothing else.
- Quantised builds shift the mid-layer state; the parity check (`forecastParity` in `fidelity.ts`) is for you to run on
  your build and data.
- Labels depend on the labeller; heads are specific to a base (or fused) checkpoint, a layer, a prompt layout and a label
  definition.
- Open: does the head add anything over S1 (the logits/pointer head on the same prefill), with and without a LoRA?
  Proposed test (not run): 2x2 of {no LoRA, LoRA} x {S1 only, S1 + forecast head}; measure accuracy of the gate decision,
  AUROC of error detection, and cost.
