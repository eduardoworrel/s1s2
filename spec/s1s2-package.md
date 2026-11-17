# S1S2 model package — specification 0.2-draft

Status: **draft 0.2** (2026-11-15). 0.1-draft described only what s1s2 already read. 0.2 describes **every
decision-head type found in the published ecosystem** (survey of published head formats, 2026-11) as peers, says for each
whether s1s2 0.1 executes it, and adds what converters need to translate a published head without silent loss
(explicit scale, logits temperature, linear and joint heads, out-of-graph norm, layer taps, attention mode, declared
source format, a fidelity record per conversion). **No head type is recommended here** (section 12): which type a
package should use is the question of lab-s12 study 008, and which confidence number an answer reports is study 007.

Machine-readable structure: [`s1s2-manifest.schema.json`](s1s2-manifest.schema.json) (JSON Schema 2020-12).
Cross-field rules a schema cannot express are listed in each section and enforced by `validateManifest`
(`src/manifest.ts`). The validator runs both:

```
node bin/s1s2.mjs validate <manifest.json | package dir | dir of package dirs> [--strict] [--manifest-only] [--root repo=dir]
```

Converters from the published formats: [`converters/heads/`](../converters/heads) (`s1s2-heads from-<format>`, section 11).

The key words MUST, SHOULD and MAY are used as in RFC 2119. "Executable" means s1s2 0.1 runs it in the browser and
Node; "described" means a manifest can state it (so a conversion does not lose it) but the runtime refuses to answer
with it.

## 1. What a package is

An S1S2 package is one language model plus everything needed to ask it **typed questions in one forward pass**
(System 1: predicate / choice / score, read from next-token logits or from a head over hidden states) and to
**generate** on the same loaded weights (System 2). It is a directory (or a Hub repo) holding:

| Item | Required | Notes |
|---|---|---|
| `manifest*.json` | yes | One manifest per prompt language/layout. |
| tokenizer files | yes | `tokenizer.json` + `tokenizer_config.json` (Hugging Face format), pinned by sha256. |
| one or more builds | yes | Graph files of a runtime (ONNX Runtime GenAI, WebLLM, LiteRT, …), pinned by sha256. |
| heads | no | Head files of any type of section 6 (`head.bin`, `*.safetensors`), pinned inside a build. |
| norm | no | `norm.bin` when the build exports the hidden state before the final norm (section 6.5). |

Every file is named by `{path, sha256, size?}` inside a repo id pinned at a 40-hex revision. A package on disk uses
`repo: "local/<package name>"` with the base model's revision as the placeholder revision.

**Validation modes.** *Manifest-only*: schema + `validateManifest`. *Package* (default): also every pinned file
present has the pinned sha256 and size, every head parses with its declared kind/format and its hidden size equals the
manifest's, `norm.bin` parses, and `genai_config.json` agrees with `hiddenSize` / `kvHeadDim` / `files.model`. A
`head.bin` whose scale is stated nowhere is a warning (section 6.1). Missing pinned files are warnings (errors with
`--strict`).

## 2. Top level

| Key | Req. | Meaning |
|---|---|---|
| `schema` | yes | `"s1s2.manifest/1"`; the deprecated `"s12.manifest/1"` is still accepted (section 10). |
| `id`, `base`, `hiddenSize` | yes | Package id; `{repo, revision}` of the checkpoint; width of the hidden state heads read. |
| `layer` | yes | `{read: "final", norm: "in-graph"}`: the default read for heads (per-head `layer` in 6.6; out-of-graph norm per build in 6.5). |
| `tokenizer`, `eos`, `warmup` | yes (warmup no) | Section 4. |
| `encoding`, `markers`, `readout` | yes | Prompt layout and read positions (section 3). |
| `logits` | yes | Label-token logits readout (section 4). |
| `defaults` | yes | `method` per question type (`"logits"` or `"head"`), optional `instructions`, `levels`, `maxSafeTokens`. |
| `generation` | yes | System 2 templates. |
| `builds` | yes | Name → build (section 7). |
| `tasks`, `escalation`, `evidence` | no | Sections 4.3, 8, 6.7. |
| `confidence` | no | Section 5.1 (hook for study 007). |
| `license`, `provenance` | no | Section 9. |
| `lab`, `x-*` | no | Free notes / vendor extensions, ignored by the runtime. Converters keep source-format facts that have no field yet under `x-<format>`. |

## 3. Prompt layout and read positions

Most published formats keep the prompt **in the reader's code**, not in the checkpoint (`formatos-head.md` §0.4).
A package MUST carry the layout its head was trained with, because the runtime has nothing else to go by:

| Source format | Where the source keeps the prompt | S1S2 encoding |
|---|---|---|
| Kev `head.pt`, GGUF `kev` | `kev.py` / `kev/model.py` (special-token delimiters), GGUF `tokenizer.chat_template.systemone` | `kev` (3.2) |
| alias3-web, ct0-edge | `aliasweb.mjs` (Kev delimiters) / lab manifests | `kev` / `tagged` |
| lab-s12 | manifest | `tagged` |
| JEV (`judge_config.json`) | `JEV_TEMPLATE` "bare-v1" in transformers `configuration_jev.py` | `tagged` when reducible; else `decision-template` (3.5) |
| pplx-decider / d3 (`decision_config.json`) | `DECIDER_PROMPTS[prompt]` + `DECIDER_TEMPLATE` in `configuration_jev.py` | `decision-template` (3.5) |
| transformers `DecisionConfig` | `decision_template.jinja` | `decision-template` (3.5) |
| Clef (`joint_head_config.json`) | `systemone/joint_schema.py` (pieces tokenized one by one) | `joint-schema` (3.4) |

### 3.1 `tagged` encoding (text markers) — executable

Unchanged from 0.1: `encoding.template` (placeholders `{state}` then `{question}`), `encoding.question` per type,
`encoding.option` (`{label}`, `{key}`, `{index}`), `forbiddenInLabels`, `escapeSpecial`, `escapeLiterals`; markers
`decide` / `option` / `last` with `at: "last-token" | "token"` and the accepted token ids, asserted on every prompt and
on the load-time probe. Normative: `{state}` before `{question}`; `markers.decide` after `{question}`;
`markers.option` after `{label}`. Recommended plain-text layout for new heads: the plain-text marker layout of 0.1.

### 3.2 `kev` encoding (special-token delimiters) — executable

`delimiters.{state, question, optionStart, optionEnd, decide}` are single special tokens. Layout:
`[prefix] state <state…> question <instructions> (optionStart <option> optionEnd)* decide [suffix]`, one prompt per
question after the shared state. Caller text has `<|x|>` rewritten to `<¦x¦>`. `predicateOptions` = [no, yes] when
`readout.predicate` is `"options"`; score levels are options too when `readout.score` is `"option"` (Kev renders each
level as an option). Checked against the reference (`sglang@da614ba systemone/kev.py encode_kev`) on the Kev tokenizer: 30/30 prompts
with equal ids and read positions (`converters/heads/s1s2_heads/e2e_kev.py`, section 11).

### 3.3 `readout`

`{predicate: "last" | "options", choice: "option", score: "last" | "option"}`: where a **pointer** head reads its keys.
Query = hidden state at the decide marker; keys = hidden state at each option marker (choice; predicate `"options"`;
score `"option"`, kev only) or at the last prompt token (score / predicate `"last"`). Linear heads read one position
(6.3); joint heads read every position (6.4).

### 3.4 `joint-schema` encoding (Clef) — described; JS encoder available

One prompt holds the state and **every** question: prefix, state, `"\n\nSCHEMA FIELDS:\n"`, then per question
`FIELD n / ID / TYPE / INSTRUCTION: <span> / ALLOWED OPTIONS:` and `OPTION i: <span>` per option (the option span is
`{"option_id": name, "description"?: …}` as compact sorted JSON), `END FIELD`, then the suffix. Each piece is tokenized
on its own and concatenated; spans are token ranges. Option order: noul `[true, false]`, choice sorted by option id,
score by level. s1s2 exports `encodeJointSchema` / `CLEF_PROMPT` (`src/heads/joint.ts`); `decide()` does not route
questions through it yet (6.4).

### 3.5 `decision-template` encoding (Jinja) — described only

`{kind: "decision-template", template: <verbatim Jinja>, mark: "<|mark|>", split?: "<|split|>", labels?, noulOptions?}`:
the transformers convention — the template renders `state` with `question` (one prompt per question) or `questions`
(one prompt), the head reads the token before each `mark`, and the sentinels are dropped from the ids. s1s2 0.1 has
no Jinja renderer: a converter MUST either reduce the template to `tagged`/`kev` and prove the ids equal on the real
tokenizer, or keep it verbatim here (the package is then "described, not executable").

## 4. Logits readout and label tokens — executable

Token refs `{id, text}` are checked against the tokenizer at load. `logits.predicate = {yes, no}`;
`logits.choice.map` ∈ `first-token | letters | tokens` (`tokens`: pinned `{label, id, text}` per label);
`logits.score = {map: "tokens", tokens} | {map: "first-token"}`.

**`logits.temperature`** (0.2): `{predicate?, choice?, score?}` positive numbers (default 1). p = softmax(L / T) over
the read tokens; predicate p = sigmoid((L[yes] − L[no]) / T); `rawMargin` is the gap of L / T. This is the
transformers `DecisionConfig.temperature` and the GGUF `decision.temperature.{type}`.

Described only: `logits.x-logitScale`, `logits.x-softcap` (Standard One `lm_rows` applies both in the server); a
package that needs them is not executable in 0.2.

`tasks` (4.3): unchanged from 0.1.

## 5. Margin and confidence

Every System-1 answer carries `margin` = p(1st) − p(2nd) and `rawMargin` = the same gap in the readout's raw score space
(0.1, study 002). Linear and joint heads produce scores like logits, so both fields are defined for them; the pointer
**score** head on `"last"` (sigmoid) has `null`.

### 5.1 `confidence` (hook, pending study 007)

Engines call different numbers "confidence" (seven definitions, C1–C7, in the lab-s12 study 007 preregistration). Every definition except C2 is a
function of the K option scores and the temperature, so it can be computed for **every head type** at answer time; C2
(p_top × label_mass) exists only for label-token logits. Manifest key (optional, reserved):

```
confidence: { definition: "C1" | "C3" | "C4" | "C5" | "C6" | "C7" | "tbd-007", source?: string, temperature?: {...} }
```

s1s2 0.1 ignores it and keeps reporting `confidence` = p(1st) (C1) plus `margin` (C4) and `rawMargin` (C5). When
study 007 reports, this section names the definition(s) per question type and head type and the runtime computes them;
no definition is preferred before that.

## 6. Heads

### 6.0 Taxonomy (all peers)

| `kind` | What it computes | Reads | Published as | s1s2 0.1 |
|---|---|---|---|---|
| `pointer` | z_k = scale · (Wq h_q + bq)·(Wk h_k + bk) / T + offset; p = softmax_k | h at decide marker vs h at each option marker | alias3-web `head.bin` (+`norm.bin`), Kev `head.pt` (fp32), GGUF `kev` (`cls.output` = q‖k), lab-s12 / ct0-edge `head.bin` | executable (`head.bin`, safetensors) |
| `linear` | z_i = softcap(W[r0+i]·h + b) / T(type), i < n options | h at one position (decide marker or last token) | pplx-decider / d3 `readout.safetensors` (rows = option codes), JEV `head.safetensors` (`proj`, slot ranges, softcap), Standard One `classifier` | executable |
| `joint` | all options of all questions of one prompt together (span means, evidence routing over the prompt, attention between questions, lexical prior from LM-head rows) | h at every prompt token + LM-head rows of option tokens | Clef `joint_head.safetensors` | reader executable (`jointScores`), not routed by `decide()` |
| label logits (no head) | z = L[label token] / T(type) | next-token logits at the last position | transformers `DecisionConfig`, Standard One `lm_rows`, vLLM `classifier_from_token`, lab-s12 `logits` | executable (temperature; not logit_scale/softcap) |
| layer-tap heads | any of the above on an intermediate layer | h at layer n | Standard One SH (`taps: ["final", 19]`, a transformer head) | described (6.6) |

### 6.1 `head.bin` (pointer, fp16) — the alias3-web layout

```
u32 LE  n = byte length of the JSON header
n bytes UTF-8 JSON {"headDim", "hiddenSize", "temperature", "score_offset"?, "scale"?, ...}
fp16 LE q [headDim × hiddenSize] (row-major), qb [headDim], k [headDim × hiddenSize], kb [headDim]; no trailing bytes
```

The published reference for this file is **`aliasrobotics/alias3-web`** (Alias Robotics, Apache-2.0, revision
`2ab2e60`): its `heads/head.bin` and its reader `aliasweb.mjs` (`scale: meta.scale ?? 1 / Math.sqrt(d)`), which s1s2
follows. (The same byte layout was used before by ct0-edge heads, which fold 1/sqrt(headDim) into q and qb and state no
key; those must declare `scale: 1`, below.)

```
z = scale · ((Wq h_q + bq) · (Wk h_k + bk)) / temperature + score_offset
```

**Scale (normative).** The reader's scale is, in order: the header's `scale`; else `1` when the header has the lab-s12
marker `scale_folded_into_q` (the factor was multiplied into q/qb); else the build head's `scale` in the manifest; else
**1/sqrt(headDim)** (alias3-web). Header and manifest MUST agree when both state it. Writers MUST write `scale`
explicitly (`1.0` when folded, together with `scale_folded_into_q: 1/sqrt(headDim)` for the lab-s12 convention), so readers that know only one convention agree. `s1s2 validate` warns on a head.bin
whose scale is stated nowhere. Compatibility proof: s1s2 `test/alias3web.test.mjs` runs the vendored alias3-web
reader and s1s2 on the real `head.bin` + `norm.bin`: identical probabilities (max |Δp| = 0).

Predicate on `"last"`: p = sigmoid(z); score head on `"last"`: score = sigmoid(z) × (levels − 1).

### 6.2 Declaring heads

Inside a build: `heads.<name> = {file, types, kind?, format?, scale?, temperature?, scoreOffset?, tensors?, rows?,
softcap?, read?, layer?, config?, source?}`, `types` ⊆ {predicate, choice, score, evidence}.

| Field | Kinds | Meaning |
|---|---|---|
| `kind` | all | `pointer` (default) / `linear` / `joint`. |
| `format` | all | `head.bin` (pointer only; default for pointer) / `safetensors`. |
| `scale` | pointer | 6.1. For safetensors pointer heads: field, else file metadata `scale`, else 1/sqrt(headDim). |
| `temperature`, `scoreOffset` | pointer, linear | Number, or per type for linear. Default: the file's (head.bin header, safetensors metadata), else 1 / 0. |
| `tensors` | linear | `{weight?, bias?}` tensor names (default `weight` / `bias`; JEV `proj.weight` / `proj.bias`; `bias: null` = none). |
| `rows` | linear | `{predicate?, choice?, score?, evidence?}: {start, end?}`; a question with n options reads rows start..start+n−1. Absent = every type from row 0 (pplx / d3). Predicate rows are ordered [no, yes]. |
| `softcap` | linear | c > 0: z → c·tanh(z / c) before the rows are taken (JEV); `null` = none. |
| `read` | linear | `"decide"` (hidden at the decide marker; JEV reads the token before `<|mark|>`) or `"last"` (last prompt token; pplx/d3 `pooling: "last"`). |
| `layer` | all | 6.6. |
| `config` | joint | The head's own config (Clef `joint_head_config.json`: hidden_size, width, routing_layers, layers, heads, feedforward). |
| `source` | all | Source format id of the conversion (section 9.1). |

Normative: at most one head per type per build; a build with heads MUST have a non-null `hiddenOutput`; each head's
hidden size MUST equal the manifest's. Heads are per build (a head is trained on one set of weights); reusing one across
graphs of the same weights is allowed and its fidelity is a build property.

### 6.3 Linear readout

z = W h + b over all K rows, optional softcap, then the rows of the question's type, then / T(type); p = softmax.
Choice: one row per option in order. Predicate: two rows [no, yes] (JEV `noul_options` order [false, true]).
Score: one row per level; score = expected level index (as transformers `Question.answer`). Reference:
transformers `31d7392` `JevForDecision.forward` (both the `decision_config` branch, offsets 0, and the `judge_config`
branch, slot offsets + softcap).

### 6.4 Joint schema head (Clef)

Port of `sglang@da614ba layers/joint_schema_head.py` (`JointSchemaHead._forward_group`), fp64 JavaScript,
`jointScores(head, hidden[T], lexicalRows(pos), questions)`. Needs the final hidden state at **every** prompt token and
the LM-head (output-embedding) row of every token inside an option span. A build that serves a joint head MUST expose
both (all positions from the hidden output; the rows from an `embeddings` file when the model ties its embeddings, or
a pinned `lmHeadRows` file). s1s2 0.1: reader + prompt encoder executable and tested; `decide()` refuses joint heads
with a pointer to `jointScores` (routing all questions of a request through one joint prompt is pending).

### 6.5 Out-of-graph final norm (`builds.<n>.norm`) — executable

`norm = {file, kind?: "rmsnorm", eps?: number (default 0), weight?: "w" | "1+w" (default "w")}`. When present,
`hiddenOutput` carries the state **before** the decoder's final norm and the runtime applies
h / sqrt(mean(h²) + eps) · (w | 1 + w) to every hidden state before any head reads it. `norm.bin` = u32 LE JSON length,
JSON `{kind?, hiddenSize}`, fp32 LE weight. alias3-web: eps 0, weight `w` (its reader applies no eps).

### 6.6 Layer taps — described

`heads.<n>.layer`: `"final"` (default) or a layer index; a build MAY name the graph output of each tapped layer
(`hiddenOutputs: {<layer>: <output name>}`). s1s2 0.1 builds export the final state only and refuse a head with an
integer `layer`.

### 6.7 Evidence head

Unchanged from 0.1 (pointer head over the passage's sentences; `evidence = {instructions, sentencePrefix?, source?}`).

### 6.8 GGUF decision metadata

llama.cpp PR #29818 stores `{arch}.decision.type`, `decision.temperature.{type}`, `decision.block_count`,
`decision.max_head_tokens` and the prompt as `tokenizer.chat_template.systemone`. Mapping: type `kev` → `pointer`
(q = rows 0..D−1 and k = rows D..2D−1 of `cls.output.weight` [2D × hidden], biases likewise from `cls.output.bias`;
scale 1/sqrt(D); temperature = `decision.temperature.choice`); the temperatures → head `temperature`; the template →
`kev` encoding when it reduces to the delimiters. Other `decision.type` values map as their source head type does
(`clef` → `joint`, `pplx-decider` → `linear`, …). The GGUF backbone weights serve no s1s2 runtime: the build comes
from another export.

## 7. Builds

```
builds.<name> = { runtime, repo, revision, files, hiddenOutput, heads?, norm?, attention?, devices, dtype?,
                  modelId?, kvHeadDim?, maxSafeTokens?, fidelity, escalation? }
```

Runtimes and roles unchanged from 0.1 (`genai`, `decode-only`, `transformers.js`, `webllm`, `litert`).

* `norm`: 6.5.
* `attention`: `"causal"` (default) or `"noncausal"` — the attention the decision pass was trained with
  (pplx-decider v1.1 and d3 `attention_mode: "noncausal_full_attention"`). Described: s1s2 0.1 builds are causal and
  share the state prefix across questions, so a package whose head needs non-causal attention is "head converted,
  package not runnable" until a full-attention build exists.
* `fidelity`: `null` until measured, else `{reference, n, decisionChanged?, orderInversion?, measuredAt?, notes?}`.
  A converter MUST write `fidelity` for every build it measured against a reference reader of the source (decision
  changed per readout, on stated items) and SHOULD measure every build it emits; a conversion counts as lossless only
  when decisions and margins agree. Head conversions also write the record of section 9.2.

## 8. Margin policy (`escalation`)

Unchanged from 0.1. No default threshold exists.

## 9. Provenance, license, conversion fidelity

`license`: SPDX expression of the package as distributed; MUST NOT be more permissive than the base's. A package
converted from a non-commercial source (e.g. `cc-by-nc-4.0`) keeps that license.

`provenance`: `generator`, `createdAt`, `base`, `dataset`, `training`, `builds`, `notes` (0.1), plus:

### 9.1 `provenance.source` (0.2)

`{format, repo?, revision?, files?, ...}`: the **declared** source format of a conversion. A converter MUST take the
format from its user (`s1s2-heads from-<format>`), never infer it from file names: `head.bin`, `head.pt` and
`decision.json` each name several incompatible layouts on the Hub. Format ids: `kev-head.pt`, `gguf-kev`,
`alias3-web-head.bin`, `decision-config-readout`, `jev-judge-config`, `transformers-decision-config`,
`standard-one-readout-v2`, `clef-joint-head`, `lab-s12-head.bin`, `ct0-edge-head.bin`.

### 9.2 Conversion fidelity record (0.2)

Every conversion writes one record (`fidelity.json` next to the converted head; schema id
`s1s2.conversion-fidelity/1`; records written before the rename carry `s12.conversion-fidelity/1`, same layout):

```
{ sourceFormat, reference: {impl, repo, commit, file}, ours: {impl: "s1s2", commit, reader},
  inputs: {kind: "synthetic-hidden" | "real-hidden" | "synthetic-logits" | ..., n, seed?},
  n, decisionAgreement, decisionChanged, maxAbsDp, maxRelDz, tolerance: {maxAbsDp, maxRelDz?}, pass, notes? }
```

The reference is the source format's own reader (cited by commit), run on the same inputs as s1s2's reader.
A conversion is "lossless" only when `pass` is true with decisions equal; tolerances are declared per record
(1e-5 for fp32 arithmetic, 2e-3 when the target stores fp16 weights). `inputs.kind: "synthetic-hidden"` tests the head
arithmetic and its declaration on the real weights; "real-hidden" also tests the prompt and read positions on the real
backbone.

## 10. Versioning and migration from 0.1

**Schema id rename.** `s12.manifest/1` became `s1s2.manifest/1` with the rename of the library; the content is the same, readers accept both and report the old id as deprecated, writers write the new one.

Additive changes keep `schema: "s1s2.manifest/1"`. **One meaning changed in 0.2**: a `head.bin` with neither `scale` nor
`scale_folded_into_q` in its header, and no `scale` in the manifest, was read with scale 1 in 0.1 and is read with
1/sqrt(headDim) in 0.2 (alias3-web compatibility). By section 10 of 0.1 this would bump the schema id; it does not,
because no S1S2 manifest has been published and every local head.bin was inventoried (2026-11-15):

| Heads | Header | Read in 0.2 | Action |
|---|---|---|---|
| 306 lab-s12 heads (unpublished) | `scale_folded_into_q: 0.0625` | scale 1 (unchanged) | none (the lab's head writer is frozen while its studies run; writing `scale: 1.0` there is pending) |
| ct0 `head.bin` (sha `1138a774…`, T 2.0, offset 16.27) and `head_score.bin` (`82640a99…`), pinned 14× in `manifests/ct0-alias3-micro-1026.json`, published in `eduardoworrel/alias3-micro-1026-*` | no scale key, folded | 1/16 without the manifest | manifest declares `scale: 1` on every head (the Hub files cannot change) |
| other unpublished ct0 heads | no scale key, folded | — | not in any S1S2 manifest; declare `scale: 1` if ever packaged |
| alias3-web `f586cb80…` | no scale key, not folded | 1/16 (= its own reader) | none |
| s1s2 test fixtures | built in memory | scale given | none |
| converter (`converter/`) heads | `scale: 1.0` + `scale_folded_into_q` | 1 | none |

A change that makes an existing valid manifest invalid, or changes another key's meaning, bumps the schema id.

## 11. Coverage (formats × conversion × fidelity)

Converters: `s1s2-heads from-<format>` (`converters/heads`). Proofs 2026-11-15 on the smallest license-compatible
public repo of each format; the fidelity records are not published yet.
n = questions; Δp = max |p_ref − p_lib|; synthetic = seeded hidden states on the real published weights.

| Source format | Head kind | Converter | Proof repo (revision) | Fidelity vs reference | Package runnable in s1s2 0.1 |
|---|---|---|---|---|---|
| Kev `head.pt` | pointer | `from-kev-headpt` | jaredpalmer/kev-0.8b (`bf75a6a`), Apache-2.0 | synthetic 64/64, Δp 2.9e-7 (fp16 head.bin 1.1e-4); **real backbone** (Qwen3.5-0.8B-Base + LoRA, fp32): ids/positions 30/30, decisions 30/30, Δp 8.0e-8 (fp16 2.6e-6) | yes (kev encoding) |
| GGUF `decision.type = kev` | pointer | `from-gguf-kev` | ggml-org/Kev-4B-GGUF (`f4624b6`, BF16 file, 2.6 MB range-fetched), Apache-2.0 | 64/64, Δp 3.9e-7; tensors vs kev-4b head.pt: max 6.1e-5 (bf16), biases exact | yes (build from another export) |
| alias3-web `head.bin` + `norm.bin` | pointer + norm | `from-alias3-web` | aliasrobotics/alias3-web (`2ab2e60`), Apache-2.0 | 32/32, Δp 0 (exact) | yes |
| `decision_config.json` + `readout.safetensors` | linear | `from-decision-config` | vllm-sr/d3-mini (`df99bf4`), Apache-2.0 | 64/64, Δp 7.6e-8 | head yes; package no while `attention: noncausal` |
| JEV `judge_config.json` + `head.safetensors` | linear | `from-jev` | autotrust/AutoTrust-Decider-Mini-9B (`7ac669b`), Apache-2.0 | 64/64, Δp 1.5e-7 | head yes; prompt (bare-v1) not yet reduced to `tagged` |
| transformers `decision.json` | label logits | `from-transformers-decision` | ArthurZ/pplx-decider-v1.1-27b (`df876c9`), Apache-2.0 | 64/64, Δp 0 (synthetic logits) | arithmetic yes; Jinja template described only |
| Clef `joint_head.safetensors` | joint | `from-clef` | Cloudflare/clef-flash (`22b7550`), Apache-2.0 | 12 questions / 6 prompts, 12/12, Δp 5.4e-7 (reference fp32 on CPU, ours fp64) | reader yes; `decide()` routing pending |
| Standard One `jev-gpu-readout-v2` (.npz) | label logits / linear | `from-lm-rows` | none public | structure only | lm_rows with logit_scale/softcap: no |
| Standard One SH (`schema_head`, taps final + 19) | joint on two layers | none | — | — | no (layer taps, no reader) |
| ONNX GenAI `component_manifest.json` | pointer as ONNX graph | none | none public | — | no artifact to convert |
| CT0 `head.pt` (aliasrobotics/alias3-micro-1026) | pointer | none | — | — | no: the file is truncated (unreadable) |

## 12. Recommended head type and confidence: TBD

**Recommended head type: to be decided by lab-s12 study 008** (the same backbone(s) and
training data with each head type, compared on accuracy, calibration and margin, int4 robustness, latency and size in
the browser). **Recommended confidence definition: to be decided by study 007.** Until then every head type above is a
peer; the order of the tables is not a preference.

**Study 008 (2026-11-16) does not change this.** Its report (unpublished) is descriptive: dev-B, one
3B backbone, two tasks, two seeds.
- No head type beat the label-token logits on balanced accuracy.
- The pointer head tied the logits on ContractNLI and led the other hidden-state heads there.
- The heads changed fewer decisions than the logits under simulated int4 on a binary task: 3–7% vs 14.5%.
- Recommended head type: still **TBD**. A recommendation needs a fresh split, more backbones and browser latency.
