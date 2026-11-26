# Media packages: audio (experimental)

Status: draft, written before the code in `src/media/`. Package spec 0.2-draft stays as it is; this file adds two optional
top-level keys to the manifest (`modalities`, `audio`; a package without a base model also carries `repo` and `revision`) and one new entry point (`loadMedia`). The numbers below were
measured in the reference runs on the builds named here; nothing else is claimed.

## 1. What changes and what does not

| | Text (unchanged) | Media (new) |
|---|---|---|
| Manifest | `s1s2.manifest/1` with tokenizer, encoding, markers, logits, heads, builds | the same schema id; a manifest whose `modalities` lacks `"text"` has none of those sections and carries `audio` |
| Load | `S1S2.load` | `loadMedia` (returns an `AudioModel`); `S1S2.load` and `validateManifest` refuse a media manifest with a message that says so |
| Request | `{ input, questions }` | `{ audio, questions }` |
| Answer | `PredicateAnswer` / `ChoiceAnswer` | the same objects (same `probability`, `confidence`, `margin`, `rawMargin`, `path: "s1"`) plus `modality`, `task`, `decision`, `threshold?` |
| Margin | `margin = p(1st) - p(2nd)` (predicate: `|2p - 1|`); `rawMargin` = gap in score space | identical definitions. The decision of a predicate uses the reader's own threshold (voice activity: 0.225), `margin` is still relative to 0.5 |
| Shared work | prefix shared by questions on one input (snapshot + prefillFrom) | ONE tower pass for any number of questions (`encode` once, `read` many times); language and term on a short clip add one more tower pass, on the left-aligned window (section 9) |
| Escalation | margin policy (`escalation`) | the manifest key is accepted and ignored; no media threshold is measured |
| Evidence / pointer heads, `generate()` | yes | no (the `term` question generates text internally, section 9; there is no `generate()` for callers) |

`modalities` is the list of inputs `decide` takes: absent = `["text"]`. `["text", "audio"]` is a regular text
manifest that also has an `audio` block (free-text questions over media: not built, see limits). The package here
is media-only: `["audio"]` (questions are fixed tasks).
A media-only manifest needs `repo` and `revision` (a 40-hex sha, or `"local"` for an unpublished package whose files `baseUrls` /
`localDirs` map) in place of the text sections.

## 2. Types

Full, commented version: [`src/media/types.ts`](../src/media/types.ts). Reduced here to what a reader of the spec needs.

```ts
type Modality = "text" | "audio";

interface MediaManifest {
  schema: "s1s2.manifest/1"; id: string;
  modalities: Modality[]; repo: string; revision: string;      // pinned files live at <repo>@<revision>
  audio?: AudioSpec; license?: string; provenance?: Provenance;
}

// ---- audio ----
interface LogMelSpec {                       // pre-processor, in JavaScript
  kind: "log-mel"; sampleRate: 16000; windowSeconds: number; nFft: number; hop: number; nMels: number;
  window: "hann-periodic"; padding: "reflect"; filters: "slaney";
  log: { floor: number; dynamicRange: number; shift: number; divide: number }; frames: number;
}
interface AudioTowerSpec {                   // encoder: one ONNX graph, weights inside, frozen
  runtime: "onnx"; file: PinnedFile; input: { name: string; shape: [number, number] };
  outputs: Record<string, { layer: number | "final"; frames: number; dim: number }>;   // taps
  devices: ("webgpu" | "wasm" | "cpu")[];
}
type Pooling =                               // frames of one tap -> one vector (after a float16 round)
  | { kind: "tail-mean"; seconds: number }                       // last seconds that hold real audio
  | { kind: "pair"; a: Span; b: Span }                           // [x*y, |x-y|] of two L2-normalised span means
  | { kind: "pair-split"; split: number }                        // [pair(pre, last), pre, last]
  | { kind: "mean-std" };
type Calibration =
  | { kind: "platt"; coefficient: string; intercept: string }    // p = sigmoid(logit * coefficient + intercept)
  | { kind: "temperature"; tensor: string }                      // p = softmax(logits / T)
  | { kind: "none" };
interface PooledLinearReader { kind: "pooled-linear"; type: "predicate" | "choice"; tap: string; pooling: Pooling;
  weights: PinnedFile; tensors: { coef; intercept; mean?; scale? }; calibration: Calibration; labels?: string[] }
interface PooledMlpReader    { kind: "pooled-mlp"; type: "predicate"; tap: string; pooling: Pooling;
  weights: PinnedFile; tensors: { layer0Weight; layer0Bias; layer1Weight; layer1Bias }; calibration: Calibration }
interface SpeechHeadReader   { kind: "speech-head"; type: "predicate"; tap: string; graph: PinnedFile;   // small ONNX graph
  question: PinnedFile & { dim: number; prompt: string }; inputs: { tap; mask; question; output };
  frames: number; temperature: number; threshold: number }
interface AudioSpec { preprocess: LogMelSpec; tower: AudioTowerSpec; readers: Record<string, AudioReader>; notes?: string; fidelity?: object }

// ---- request and answer ----
interface MediaDecisionRequest { audio?: { samples: Float32Array; sampleRate: number };
                                 questions: (string | { task: string; term?: string; options?: string[] })[] }
type MediaAnswer = (PredicateAnswer | ChoiceAnswer) & { modality; task; decision: string; threshold?: number; available: true }
                 | { name; task; modality; available: false; reason: string };       // e.g. a speaker question on a short window
interface MediaDecision { answers: MediaAnswer[]; cost: { ms: number; parts: Record<string, number>; passes: number;
                          backend: string; device: string } }
```

## 3. Audio

Pre-processor (`audio.preprocess`, `src/media/mel.ts`): Whisper-style log-mel, 16 kHz mono, window 8 s (128000 samples, right
aligned: shorter audio is zero-padded on the left, longer keeps its last 8 s), n_fft 400, hop 160, 128 Slaney mel bands (the filter
bank is computed, identical to the 25728 values of the reference run's `mel_filters.json`), periodic Hann, reflect padding 200, power,
`log10(max(x, 1e-10))`, floor at max - 8, `(x + 4) / 4`, last frame dropped: [128, 800]. Float64 inside, float32 out; about 0.15-0.22 s per
window in JS (median 118-128 ms in Node and Chrome).

Tower (`audio.tower`): the frozen Qwen3-ASR-0.6B audio encoder exported for the fixed 800-frame window (one ONNX file, 746 MB,
float32), outputs `h9`, `h12`, `h18` (104 frames x 896, after layers 9, 12, 18) and the final projection `audio_features` (104 x 1024), which the readers do not use and the decoder (section 9) reads.

Readers (`audio.readers`, task name -> reader):

| task | reader | tap | pooling | calibration |
|---|---|---|---|---|
| `voice_activity` | `speech-head` (ONNX, 9.7 MB + a pinned 256-float question vector of ONE canonical question) | `h9` | inside the graph (100 frames, frame mask) | `p = sigmoid(score / 0.654)`, decide at 0.225 |
| `turn_end` | `pooled-linear` | `h18` | `tail-mean` 2 s of real audio | standardise (float64) -> linear -> Platt |
| `same_speaker` | `pooled-linear` | `h12` | `pair` 0-4 s / 4-8 s | standardise -> linear -> Platt |
| `speaker_change` | `pooled-mlp` (3584 -> 256 -> 1) | `h12` | `pair-split` at 6 s | sigmoid -> logit -> Platt |
| `speaker_count` | `pooled-linear`, 4 rows, labels 0 / 1 / 2 / 3+ | `h12` | `mean-std` | softmax with a fitted temperature |
| `language` | `decoder-language` (section 9) | `audio_features` | none (decoder reads the 104 vectors) | softmax of summed log-probabilities |
| `term` | `decoder-term` (section 9) | `audio_features` | none | none (fraction of five hypotheses) |

Readers run in float64 over float16-rounded features (the readers were fit on float16-stored features). Speaker tasks need a full
window (8 s of real audio) and answer `available: false` otherwise; `turn_end` needs at least one frame of real audio in the last 2 s.
The readout files are Marimba's own safetensors, pinned unchanged; the tower and the speech head are ONNX exports written by
`recipes/marimba-audio/`.

## 4. API

```js
import { loadMedia } from "@s1s2/web/media";

const audio = await loadMedia({ manifest: "manifests/marimba-audio.json", baseUrls: { "local/marimba-audio-web": new URL("/weights/marimba-audio/", location.href).href }, modules: { ort }, allowUnverified: true });
const d = await audio.decide({ audio: { samples, sampleRate: 16000 }, questions: ["voice_activity", "turn_end", "speaker_change"] });
// d.answers[i]: { task, probability, margin, rawMargin, decision, threshold, available } ; d.cost: { ms, parts: { mel, tower, readers }, passes: 1 }
const enc = await audio.encode({ samples, sampleRate: 16000 });          // one tower pass ...
const more = await audio.read(enc, ["same_speaker", "speaker_count"]);   // ... any number of readers (the five together: 8-11 ms, no tower pass)
// with audio.decoder in the manifest (section 9); ask language and term in one call to share the decoder's prompt pass
const w = await audio.read(enc, ["language", { task: "term", term: "obrigado" }]);   // w[0].choice, w[1].probability (n/5), w[1].hypotheses
```

`@s1s2/web/media` exports its own `ManifestError` (thrown by `validateMediaManifest`). Each entry point is a separate bundle, so it is not `instanceof` the `ManifestError` of `@s1s2/web` or `@s1s2/web/manifest`: compare `e.name === "ManifestError"` across entry points.

## 5. Tolerances (fixed before the tests that use them)

| quantity | reference | tolerance | reference measurement |
|---|---|---|---|
| log-mel, JS vs torch feature extractor | `melA.f32` / `melB.f32` | max abs <= 1e-4 | 3.2e-5 (56 + 56 windows) |
| voice_activity, turn_end, same_speaker, speaker_change, speaker_count: probability | torch Marimba (`ref.json`) | abs <= 1e-4 | <= 3.8e-5 (Chrome WebGPU, 56 clips) |
| decisions at their thresholds | torch | identical on every window | 56 / 56 |
| float16 rounding, JS fallback vs `Math.f16round` | itself | exactly equal | n/a |

The comparison of the readers and of the speech head with the torch probabilities needs the weights, which are not published: on a clone
without `.weights/marimba-audio/` (or `S1S2_MM_WEIGHTS`) those tests are reported as skipped, with the message saying what is missing, and
`npm test` does not prove them. The log-mel against torch, the float16 fallback and the reader arithmetic run from the stored fixtures on every clone.

A reader tested on fixtures (tower outputs stored with the tests) is held to the same 1e-4 against the torch reference: the tower
itself differs from torch by <= 1.8e-4 on any hidden value, which moves a probability by < 4e-5.

## 6. Limits

- **Language and spoken-word questions need the decoder block.** They are answered by Marimba's own route, the Qwen3-ASR decoder fed the
  tower's 104 final vectors (section 9). A package without `audio.decoder` has voice activity, turn end, same speaker, speaker change and
  speaker count only.
- **`voice_activity` answers one question.** Its question vector is the pre-computed encoding of one Portuguese prompt; another
  wording needs Marimba's text encoder, which is not exported. Speech head temperature and threshold are those of that prompt.
- **Windows are 8 s.** The tower graph is fixed at 800 mel frames. Speaker tasks need 8 s of real audio; nothing longer than 8 s is
  scored (the last 8 s are). Marimba itself calls context above 8 s unvalidated.
- **Sample rate is 16 kHz mono, checked, not converted.** The reference parity used soxr-resampled windows; the demos resample with the
  browser (Web Audio), which is another resampler. Parity of that path is not measured.
- **Readers are calibrated on their own data.** The Marimba card reports that turn-end calibration shifts across domains (ICSI: ECE 0.200 before per-domain calibration, 0.011 after), and that speaker questions compare two four-second halves or the last two seconds with the previous six; they identify no one. A `turn_end` or `speaker_*` answer on something that is not speech is not defined: the demo's synthetic noise clip gets `voice_activity` no and `turn_end` yes (90 %). Read `voice_activity` first.
- **The demo's microphone button is not verified.** It is written (MediaRecorder on click, decoded with Web Audio at 16 kHz); in headless Chrome with a fake device `getUserMedia` did not resolve, so that path was never run. File and clip inputs were.
- **Margins are not escalated.** No margin threshold is measured for audio; `escalation` is ignored.
- **Weights are not published.** The manifest is `revision: "local"`; their files are produced by the recipes from third-party
  checkpoints and mapped with `baseUrls` / `localDirs`. Publication is a decision for the package owner; section 8 lists the licences read on each source.
- Large files in the browser are not re-hashed by the page (see `Files.large`); the demo server hashes what it serves against the
  manifest before it starts.

## 7. Open points

- **Not verified:** the demo's microphone button; browser resampling to 16 kHz against the reference resampler; weights downloaded
  from a hub (every run used local files, none are published).
- **Not measured:** a margin policy for audio.

## 8. Licences read on the sources

Read on the model and dataset cards (and the `LICENSE` file where the repository has one), November 2026. This is a record of what the
sources state, not legal advice, and nothing here distributes weights.

- The decoder build (section 9.3) is a derivative of `Qwen/Qwen3-ASR-0.6B` (Apache-2.0): a published copy needs that licence text, a notice that the weights were converted (renamed, int4-quantised, embedding table written as float16), and the source revision. The language and term logic is a port of `marimba-audio` 0.1.2 (Apache-2.0; no code is copied, behaviour is) and should be credited as such.
- Apache-2.0 is declared by `Qwen/Qwen3-ASR-0.6B`, `charcot-ai/marimba` (head and readout weights; its code repository has an Apache-2.0
  `LICENSE`). Some of
  these Hugging Face repositories have no `LICENSE` file, only the card field.
- Training data have their own terms: the Smart Turn v3.2 data (turn-end readout) is CC BY 4.0. Whether those terms reach the trained
  weights is not stated by the sources.
- FLEURS (the example clips and test fixtures) is CC BY 4.0; attribution is in `examples/audio-turn/clips/ATTRIBUTION.md`.

## 9. Language and spoken term (decoder)

Status: written before the code that implements it and before any number below existed; the tolerances in 9.4 were fixed first.

Marimba answers "which language" and "was this word said" with no extra model: the Qwen3-ASR decoder (a Qwen3 text model: 28 layers,
hidden 1024, 16 / 8 attention heads, vocabulary 151936) reads the tower's final 104 x 1024 vectors in place of the `<|audio_pad|>`
tokens of a fixed prompt. The package has two more readers over that decoder and one more block, `audio.decoder`. Source of the behaviour:
`marimba-audio` 0.1.2 (Apache-2.0), `language_scores`, `hypotheses`, `exact_fraction`; nothing below is the authors' claim about this port.

### 9.1 Behaviour that is ported

- **Window.** The decoder lane takes the window with the audio at index 0 and zeros after it (left-aligned; the tower reads the right-aligned
  one). With less than 8 s of real audio that is a second tower pass (mel + tower, once per `encode`, reused by both questions); with 8 s it is
  the same pass. Always 104 audio vectors.
- **Prompt (119 positions).** 9 ids, 104 audio positions, 6 ids: `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>`, audio,
  `<|audio_end|><|im_end|>\n<|im_start|>assistant\n`. The ids are in the manifest; no tokenizer runs for the prompt. The audio rows enter as float16.
- **`language`** (choice). The prompt plus the token `language` (120 positions) is read; each language's score is the sum of the log-probabilities
  (full-vocabulary log-softmax) of the tokens of its name written after `language ` (30 names, 28 of one token; Cantonese and Macedonian have two).
  One prefill, then one extra step for each two-token name. Probabilities are the softmax of the scores over the languages asked for (default all 30).
  Decision = the most probable code.
- **`term`** (predicate; the request carries the term). Free generation from the 119-position prompt, with no forced language: one greedy
  hypothesis and four beam hypotheses (beam 4, `max_new_tokens` 256, length penalty 1, `early_stopping` false, EOS 151645 and 151643), five in all.
  Each is decoded without special tokens, repetitions are collapsed (`detect_and_fix_repetitions`, threshold 20), text before `<asr_text>` is dropped.
  The term is normalised (case-fold, NFKD, combining marks removed, runs of non-letters and digits and underscores become one space) and counts as said when
  its words occur as whole words in the normalised hypothesis. `probability` = fraction of the five hypotheses that contain it (0, 0.2, ..., 1), **uncalibrated**; `decision`
  "yes" at 0.5 or more. The answer carries the five hypotheses.

### 9.2 Manifest (`audio.decoder`) and readers

```ts
interface DecoderSpec {
  runtime: "onnx-genai";
  graph: PinnedFile; data: PinnedFile;          // model.onnx + model.onnx.data (ONNX Runtime GenAI int4 build, built with inputs_embeds): the WebGPU build
  cpuBuild?: { graph: PinnedFile; data: PinnedFile };   // the same weights built for CPU kernels (Node CPU, browser WASM); without it the decoder is WebGPU only
  embeddings: PinnedFile;                       // float16 little-endian [vocab, hidden]: the token-embedding rows of the prompt
  tokens: PinnedFile;                           // JSON {tokens: string[vocab] (byte-level BPE pieces), added: number[], special: number[]}, to decode hypotheses
  model: { layers: 28; kvHeads: 8; headDim: 128; hidden: 1024; vocab: 151936 };   // layers and kvHeads are checked against the graph's cache inputs
  prompt: { before: number[]; after: number[]; audioFrames: 104; languageToken: number };
  eos: number[];
  languages: Record<string, { name: string; ids: number[] }>;   // code -> the token ids of " Name" (one or two tokens)
  term: { beams: 4; maxNewTokens: 256; lengthPenalty: 1; repetitionThreshold: 20 };   // all four are read by the library
  devices: ("webgpu" | "wasm" | "cpu")[];
}
// readers: { language: { kind: "decoder-language", type: "choice", tap: "audio_features" },
//            term:     { kind: "decoder-term",     type: "predicate", tap: "audio_features" } }
```

Request: `questions: ["language", { task: "term", term: "obrigado" }]`, and `{ task: "language", options: ["pt", "en", "es"] }` for a closed set.
Answers: `language` is a `ChoiceAnswer` (labels = codes, `probabilities`), `term` a `PredicateAnswer` with `threshold` 0.5 and `hypotheses`.
Both are answered from one prefill of one encoding (`read` many times); `cost.parts` names `decoderOpen` (the prompt rows built from the tower vectors), `language`, `term` and, when the window was short, `leftTower` (the second tower pass).
`options` of `language` must be at least two distinct codes of `audio.decoder.languages`; anything else is refused before the tower runs.

**Loading.** The decoder is 0.66 GB on top of the 0.75 GB package, so loading it can fail where the rest works (WebGPU out of memory, a server without Range requests, a manifest with no build for the device). `loadMedia({ decoder })` says what then happens:

| `decoder` | loads the decoder | if loading fails |
|---|---|---|
| `"auto"` (default) | yes, when the manifest has one | the model works **without** it: the five other questions answer as before, `onStatus` gets "decoder not loaded, language and term are unavailable: <reason>", `model.decoder` is `{ loaded: false, reason }` and `model.tasks` leaves out `language` and `term`; asking for either throws "needs the decoder, which failed to load: <reason>" |
| `true` | yes | `loadMedia` throws and releases what it opened |
| `false` | no, nothing of it is fetched | not applicable; `model.decoder` is `{ loaded: false, reason: "off" }`; asking for `language` or `term` throws "needs the decoder, which was not loaded" |

A manifest of the audio package without `audio.decoder` is unchanged by all of this (`model.decoder` is `undefined`). `model.dispose()` waits for the language and term questions in flight before it releases the decoder session.

### 9.3 Build of the decoder

Exported from the `Qwen/Qwen3-ASR-0.6B` checkpoint (revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, safetensors) by `s1s2-convert asr-decoder`:
the `thinker.model.*` tensors become a plain Qwen3 causal LM (`lm_head` is byte-identical to the embedding, so the head is tied), the audio tower is dropped,
and the ONNX Runtime GenAI 0.19.0 builder writes an int4 graph that takes `inputs_embeds` (no embedding in the graph) plus the float16 table. For audio-only
input the three rotary positions of the original are equal, so the plain rotary embedding is the same computation. That is checked by a separate script,
`recipes/marimba-audio/check_decoder_rename.py`, against the original thinker on real audio embeddings (largest logit difference in 9.5); the converter does not run it
and `decoder.json` does not record it.

Reproducing the build: `s1s2-convert asr-decoder` takes the checkpoint directory, its revision and the sha256 of `model.safetensors` (both required: the revision is checked
against a Hugging Face snapshot directory name, the hash against the file), and is run twice into the same `--out`, `--ep webgpu` first and then `--ep cpu`. It writes the
layout the manifest lists (`decoder/model.onnx`, `decoder/cpu/model.onnx`, `decoder/embed_tokens.fp16.bin`, `decoder/decoder_tokens.json`, ...) and `decoder/decoder.json` with
every file's sha256 and size, the build arguments of each run, the versions of torch, transformers and the builder tools, and the settings of the term question
that `recipes/make-manifests.mjs` copies into the manifest. The second run checks that the shared files are byte-identical. The builder is the patched one of `s1s2-convert build`;
the int4 files are reproducible only with the same builder and onnxruntime versions, which is why the manifest pins the hashes of the files and not of the recipe.

### 9.4 Tolerances (fixed before the tests that use them)

References: `marimba-audio` 0.1.2, float32, CPU, on the 56 pt-BR clips, the three demo clips and 12 full windows made of three joined clips
(`recipes/marimba-audio/decoder_refs.py`). A "pair" is a (window, term) with terms taken from the reference transcript (present), from other
clips (absent) and from the hypotheses.

| quantity | reference | tolerance |
|---|---|---|
| `audio_features` of the left-aligned window, tower ONNX (Node CPU) | torch | max abs <= 2e-3 |
| prompt ids, name token ids, decoded text of given token ids, `normalize`, `detect_and_fix_repetitions`, `parse_asr_output` | Python | exactly equal on every fixture |
| language, decoder fed torch `audio_features` (int4, Node CPU): most probable code | torch | identical on every window |
| same: probability of the most probable code | torch | abs <= 0.02 |
| term fractions, same input | torch | within 0.2 on >= 95 % of pairs; identical on >= 85 % |
| term decision at 0.5 | torch | identical on >= 95 % of pairs |
| greedy hypothesis text, beam hypotheses text | torch | **reported**, not required |
| end to end (JS mel -> tower -> decoder), same quantities | torch | the same as the three rows above |
| Chrome WebGPU against Node CPU (two builds of the same weights, see below) | Node CPU | language code identical on every window; probability abs <= 0.02; term decision identical on >= 95 % of pairs |

The last row first said "same graph". That could not be: the WebGPU build fuses the per-head query/key norm into its attention node and only the
WebGPU kernel runs it, so Node CPU and browser WASM run a second int4 build of the same weights (`cpuBuild`), quantised by the same builder for another
provider. The numbers in the row were not changed; its wording was, after the first runs. Every run below says which build it used.

Not a gate: time and memory (reported in 9.5). Marimba's own acceptance rule (change of AUROC at most 0.005, unchanged language accuracy against float32)
cannot be tested at this sample size and is not claimed.

### 9.5 Measured

Reference windows: 72 (56 FLEURS pt-BR clips, 3 demo clips including a synthetic noise clip, 12 full windows of three joined clips); 952 (window, term) pairs
(terms from the window's own transcript, from other clips, from the hypotheses, and "marimba"). Hardware: Apple M2 Pro, 16 GB, Chrome 158, onnxruntime-web 1.31.1, onnxruntime-node 1.30.0.

Builds (int4, ONNX Runtime GenAI 0.19.0): WebGPU graph 0.35 GB (`model.onnx.data` 352,149,504 B) and CPU graph 0.41 GB (406,306,816 B), each with the shared
float16 embedding table 311,164,928 B and 1.8 MB of token pieces. The package grows from 0.75 GB to 1.41 GB on WebGPU (+ 0.41 GB if the CPU build is shipped too).

Equivalence of the rewrite (float32): the plain Qwen3 checkpoint and the original thinker give logits that differ by at most 5.2e-4 (logit magnitude up to 34), same arg max
at every position (3 windows). The float32 ONNX build, read by the library on Node CPU, against torch (12 windows): language probability within 7.5e-8, greedy and
beam texts identical on every window, term fractions identical on every pair. The JavaScript beam search gave the Hugging Face results on those 12 windows (and its logic was read against `_beam_search` of transformers 4.57.6); every difference below comes from int4.

| run (int4) | windows | language code | top probability, largest abs diff | term fraction identical / within 0.2 | term decision | greedy / beam text equal |
|---|---|---|---|---|---|---|
| Node CPU, CPU build, torch `audio_features` in | 72 | 72 / 72 | 0.090 (noise clip; speech only: 0.020) | 94.3 % / 98.1 % | 98.6 % | 51 % / 26 % |
| Node CPU, CPU build, end to end (JS mel, ONNX tower) | 72 | 72 / 72 | 0.092 (noise clip; speech only: 0.018) | 94.3 % / 97.7 % | 98.3 % | 54 % / 25 % |
| Chrome WebGPU, WebGPU build, torch `audio_features` in | 72 | 72 / 72 | 0.035 (noise clip; speech only: 0.015) | 94.3 % / 98.2 % | 98.5 % | 47 % / 25 % |
| Chrome WebGPU against Node CPU (two builds) | 72 | 72 / 72 | 0.055 | 98.1 % / 99.7 % | 99.7 % | 72 % / 50 % |

The `audio_features` of the left-aligned window from the JavaScript mel and the ONNX tower differ from torch's by at most 3.2e-5 (tolerance 2e-3).
Language and term in Chrome WASM (CPU build; 6 windows): language code 6 / 6, all term gates met on those windows.

Against the tolerances of 9.4: every row is met **except** "probability of the most probable code <= 0.02": it holds on every speech window of the WebGPU build (largest 0.015)
and of the end-to-end CPU run (0.018), fails by 0.0001 on one speech window of the CPU run with torch features (0.0201), and fails on the noise clip in every int4 run
(torch gives English 0.76 there, a spread-out distribution; int4 gives 0.85: 0.090 on Node CPU, 0.092 end to end, 0.035 on WebGPU). **The 0.02 tolerance is therefore not met** on the noise clip in any int4 run
and by 0.0001 on one speech window of one run; the language code is right on all 72 windows, and the probability of the language is not within 0.02 of torch's on non-speech audio. The tolerance was not changed. Equality of text is low for int4 (greedy about half, beams a quarter):
the differences are spelling and a word here and there, which is why the term gates are on the fraction and the decision.

Cost (Chrome, WebGPU, median of 8 windows each; the tower is warm):

| step | joined 8 s window | short clip (4-8 s of audio) |
|---|---|---|
| log-mel (JavaScript) | 156 ms | 149 ms |
| tower | 72 ms | 72 ms (+ 214 ms for the left-aligned pass) |
| decoder prompt (open) | 1 ms | 215 ms (first prefix) |
| language (prefill + two extra steps) | 126 ms | 126 ms |
| term: greedy (25-29 tokens) | 355 ms | 322 ms |
| term: four beams | 1.36 s | 1.10 s |
| term (greedy + beams) | 1.66 s | 1.41 s |
| mel + tower + language + term | 2.03 s | 1.97 s |

Node CPU (CPU build): language 0.17 s, term 1.2-1.3 s per window. Chrome WASM (CPU build, 2 short clips): tower 0.51 s (+ 0.58 s left-aligned), language 1.9 s,
term 36.7 s (greedy 8.2 s, beams 28.7 s). Memory: Chrome with the tower and the decoder loaded, 3.3 GB resident in all processes (4.7 GB right after loading; GPU process 0.4-0.5 GB);
the page took 4.3 s to load 1.4 GB from local files.

### 9.6 Limits of the decoder questions

- Only Portuguese audio was run (the clips available), plus one synthetic noise clip. The other 29 languages are the reference's behaviour, not checked here, and the two-token
  names (Cantonese, Macedonian) go through one extra decoding step that was not checked on their own audio.
- Rare words, names and numbers: the hypotheses come from a 0.6B model; a term the model spells another way is "no". Not validated beyond the
  probe terms of 9.4.
- Beam search was re-written (the reference uses the Hugging Face generation code); the hypotheses can differ in ties and spelling without a quality
  difference. Scores are summed in float64 where Hugging Face uses float32, which can turn a near tie in rare cases. The int4 graph and the float16 audio rows can reorder the beams. Equality of text is reported, not promised.
- A term in Chinese, Japanese or Thai (scripts written without spaces) never matches inside a sentence: whole words are found by spaces, as in the reference (`termFraction("你好", ["你好世界"])` is 0).
- Normalisation uses the Unicode version of the JavaScript engine (the fixtures come from Python 3.11, Unicode 14): letters added since may be treated differently from one browser to another. The fixture lists the code points it covers.
- 8 s windows; Marimba itself calls longer context unvalidated.
- int4 changes the text: about half of the greedy transcripts and a quarter of the beam transcripts differ from float32 torch (a word or a spelling), and the probability of the most probable language moved by up to 0.09 on a non-speech clip. A float32 build (2.4 GB) reproduces torch exactly (12 windows) and is the fallback if that matters.
- The term takes 1.4-1.7 s per window on WebGPU and about 37 s on browser WASM: ask it on demand, not on every window.
- `probability` of `term` is a fraction of five hypotheses, not a calibrated probability.

### 9.7 Free transcription (`transcribe`)

`AudioModel.transcribe(input, options?)` returns what the decoder writes for one window, with no term to look for. It is the same engine as `term` (same window, same prompt,
same greedy and beam search, same repetition clean-up and `parse_asr_output`), so it adds nothing to the manifest: it needs `audio.decoder` and a decoder reader (the tap is the one
`language` and `term` read), and it throws the same message they throw when the decoder is not loaded (`needs the decoder, which failed to load: ...` or `... was not loaded (loadMedia option decoder: false)`).

```ts
transcribe(input: AudioEncoding | { samples: Float32Array; sampleRate: number }, options?: { beam?: boolean; maxNewTokens?: number; signal?: AbortSignal }): Promise<Transcription>
interface Transcription {
  text: string;                       // the transcript, without the "language X" prefix; "" for silence
  language: string | null;            // code of the language the model wrote before the transcript; null when it wrote none ("language None" too) or a name with no code
  method: "greedy" | "beam";
  hypotheses?: string[];              // beam only: the four beams, best first; text === hypotheses[0]
  tokens: number;                     // generated tokens of text, the end token not counted
  finishReason: "stop" | "length";    // length: maxNewTokens cut it
  cost: MediaCost;                    // ms, parts { mel, tower (only when given audio), leftTower, decoderOpen, decode }, passes, backend, device
}
```

- **Input.** An `encode` result (reuse it for `read` and `transcribe`) or the audio itself, encoded first. Less than 8 s of audio: the left-aligned window, the second tower pass of 9.1, done once per encoding and shared with `language` and `term`. Only the last 8 s are read; there is no chunking.
- **Method.** Greedy by default (the `term` greedy hypothesis, the cheapest). `beam: true` runs beam 4 (the manifest's `term.beams`) with the length penalty of the manifest and returns the best beam as `text`; the four beams are `hypotheses`, the same four that `term` returns after its greedy one. `maxNewTokens` defaults to the manifest's (256).
- **Same engine as `term`.** On the same encoding, `transcribe(enc).text` is `term`'s `hypotheses[0]` and `transcribe(enc, { beam: true }).hypotheses` are its `hypotheses.slice(1)`. `recipes/marimba-audio/transcribe-parity.mjs` checks this on the real package.
- **Language.** The name after `language ` in the generated prefix, mapped to a code by `audio.decoder.languages[code].name` (case-insensitive).
- **Abort and dispose.** An aborted `signal` rejects with an `AbortError` before the next step (no partial text) and frees the decoder state. `dispose()` waits for transcriptions in flight, like for questions; after it, `transcribe` rejects with "disposed".
- **Cost.** Node CPU, CPU build, 12 full windows (median): greedy 0.37 s, beam 1.21 s, the whole `term` question 1.62 s (greedy + beams), on top of the tower and the prompt rows. Other runtimes: 9.5; the browser WASM beam is slow, ask for it on demand.
- **Real-package check.** `recipes/marimba-audio/transcribe-parity.mjs`, 12 pt-BR windows: the greedy text equals `term`'s first hypothesis on 12 of 12 and the beams equal its other four on 12 of 12; language `pt` on all. Against float32 torch, int4 gave the same greedy text on 5 of 12 windows and the same four beams on 0 of 12 (reported, not required).
- **Limits.** All of 9.6 applies. int4 changes the text: about half of the greedy transcripts and three quarters of the beam transcripts differ from float32 torch (a word or a spelling); only Portuguese was validated; one 8 s window, no timestamps, no streaming, no punctuation or number guarantees beyond what the 0.6B model writes. Equality with the torch text is asserted only for the float32 build and the text fixtures (the parse and the decoding of given token ids); the int4 text is checked for format and stability, and for equality with `term`'s hypotheses.
