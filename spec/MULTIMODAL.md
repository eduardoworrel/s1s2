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
A media-only manifest needs `repo` and `revision` (a 40-hex sha, or `"local"` for a package that is not on the Hub, whose files `baseUrls` /
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

const audio = await loadMedia({ manifest: "manifests/marimba-audio.json", baseUrls: { "eduardoworrel/Marimba-ONNX-S1S2": new URL("/weights/marimba-audio/", location.href).href }, modules: { ort }, allowUnverified: true });
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

The comparison of the readers and of the speech head with the torch probabilities needs the weights, which are not in this repository (the package is on the Hub, but the tests read local files): on a clone
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
- **Weights are on the Hub, not in this repository.** `manifests/marimba-audio.json` pins `eduardoworrel/Marimba-ONNX-S1S2` at the commit that
  holds the weight files, so `loadMedia` with no `baseUrls` downloads them from there (the browser checks each large file against the sha256 the
  Hub reports, so `allowUnverified` is not needed). The files are produced by the recipes from third-party checkpoints; `baseUrls` / `localDirs`
  still map them to a local copy (the `revision` is then not used). Section 8 lists the licences read on each source.
- Large files in the browser are not re-hashed by the page (see `Files.large`); the demo server hashes what it serves against the
  manifest before it starts.

## 7. Open points

- **Not verified:** the demo's microphone button; browser resampling to 16 kHz against the reference resampler; weights downloaded
  from the Hub by this repository's tests or in a browser (the tests and the measurements here read local copies; the URLs it would fetch are checked by a test with a mocked `fetch`).
- **Not measured:** a margin policy for audio.

## 8. Licences read on the sources

Read on the model and dataset cards (and the `LICENSE` file where the repository has one), November 2026. This is a record of what the
sources state, not legal advice, and nothing in this repository distributes weights; the weights package on the Hub carries its own `LICENSE` (Apache-2.0) and `NOTICE`.

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

## 10. Image: the Visual Jev decision reader (draft, partly built)

Status: specified before the runtime. Built and tested (no model weights needed): the manifest block `image` (types, schema, rules), the pure parts in `src/media/imagePrep.ts`
(smart resize, Pillow bicubic resize, patch layout, prompt text, M-RoPE positions, option readout) and the converter commands `vl-swap` and `vl-graph`
(`converter/s1s2_convert/`). Not built: the browser runtime (`loadMedia` refuses an image manifest with that message), the weights package, every number that needs the 4B model.
Numbers called "reference" are the authors' (arXiv 2609.25845); none was reproduced here.

### 10.1 What the reader is

- **Model.** `guanxuyu/visual-jev-4b-answer-sft`: a LoRA adapter (r = 16, alpha = 32, dropout 0.05 in training; targets q, k, v, o, gate, up, down) on `Qwen/Qwen3-VL-4B-Instruct`, both Apache-2.0.
  Read from the adapter itself: 504 tensors = 252 modules = 36 layers x 7 projections of the **language tower**; no key mentions the vision tower
  (`exclude_modules: ".*visual.*"`), the embedding or the LM head. So `vision.onnx` and the token embedding of the public build are used unchanged, and only the 252 int4 matrices of the text decoder change.
- **Readout.** No extra head. One prompt per question ends at the colon of `Answer:`; the score of option i is the LM-head logit of the token `" A"`, `" B"`, ... at that position; the answer is a
  softmax over the K options supplied (2 to 16 in the reference; the reference evaluates K up to 8 and trained on K in {2, 4}). The LM head is tied to the embedding, so the score is `hidden . embedding_row(token)`.
- **Reference numbers (the authors').** Macro accuracy on four held-out benchmarks 0.885 for the 4B adapter against 0.834 for the same backbone with the LM-head readout and no SFT (GQA 0.916, SNLI-VE 0.808,
  TextVQA 0.975, TallyQA 0.841); the gain sits on the two task families in training (GQA, SNLI-VE); counting stays weak; English only; probabilities are overconfident (ECE 0.025 after temperature scaling on the task).
  A fixed confidence threshold exceeded its target error rate, and reordering the options changed 17% of accept-or-review decisions on SNLI-VE: option order is part of the question and is never permuted by the reader.
- **Branches.** One visual prefix (system turn, the image, the shared context) is encoded once; each question is an independent branch over the prefix cache. Reference: 5.6 ms against 20.5 ms per question at 32 questions per image on one RTX 5090. Not measured here.

### 10.2 What the public ONNX builds are, and the DeepStack finding

Public graphs (`onnx-community/Qwen3-VL-4B-Instruct-ONNX`, ONNX Runtime GenAI layout: `embedding.onnx`, `vision.onnx`, `text.onnx`; cpu_and_mobile and cuda variants, int4 RTN block 32; there is **no WebGPU build and no
transformers.js layout** for the 4B; transformers.js 4.x maps `qwen3_vl` to its Qwen2.5-VL class, which reads `embed_tokens`, `vision_encoder` and `decoder_model_merged` files and has no DeepStack path):

| file (cpu / cuda) | size | content |
|---|---|---|
| `text.onnx` + `.data` | 2.52 GB / 2.27 GB | 36 layers, 253 `MatMulNBits` (q, k, v, o, gate, up, down separate; lm_head), contrib ops `GroupQueryAttention`, `RotaryEmbedding`, `SkipSimplifiedLayerNormalization`; inputs `inputs_embeds`, `attention_mask`, `position_ids [3, B, S]`, cache; output `logits [B, S, 151936]` + cache. Scales float32 / float16, norm weights float32 |
| `vision.onnx` + `.data` | 0.19 GB (int4) / 0.67 GB (float16) | 24 blocks + one merger; 24 `Loop` nodes; output `image_features` only |
| `embedding.onnx` + `.data` | 0.21 GB (int4, block 128) / 0.78 GB (float16 table) | token lookup + scatter of the image rows into the sequence |

**DeepStack is not in these graphs.** Qwen3-VL adds visual features taken after vision blocks 5, 11 and 17 (`deepstack_visual_indexes`), through three extra mergers, to the hidden state after text layers 0, 1 and 2 at the
image positions; the reference model code of the paper (`vdm_model.py`) relies on it. Checked in the public graphs (cpu and cuda, and the 2B transformers.js layout): the vision graph has 49 LayerNorm (24 x 2 + the merger) and 98
`MatMulNBits` (24 x 4 + 2), i.e. one merger, and `image_features` is the output of that merger (a bias Add after its second linear), not a sum; the text graph reads no input besides `inputs_embeds`, the mask, positions and the cache, and
layers 0 to 2 add nothing but layer outputs. The ONNX Runtime GenAI builder in a local checkout (commit of 2026-10-25) has no DeepStack either; whether a newer upstream has was not checked (the file URL did not resolve).
Consequence: the public builds compute the backbone **without DeepStack**, a different function from the one the adapter was trained and evaluated on. How large the difference is on decisions has not been measured.

Two routes, both written down, neither claimed:

- **A. Faithful build (recommended).** `vl-graph vision` adds three outputs `deepstack_0..2` (the hidden input of the LayerNorm that opens blocks 6, 12 and 18, through `reshape(-1, 4096)`, LayerNorm, Linear, GELU, Linear with the weights of
  `visual.deepstack_merger_list.N`, 165 MB read from the base shards); `vl-graph text --deepstack` adds inputs `deepstack_0..2 [B, S, hidden]` added into the `down_proj` output that feeds layers 1, 2 and 3 (the fused residual stream
  has no other place); zero except at image positions, so the suffix passes feed zeros. Each edit is tested on synthetic graphs against a numpy model (`converter/tests/test_vl.py`); the real graphs are only checked structurally.
- **B. Public graphs as they are**, reported as "Qwen3-VL-4B + LoRA without DeepStack", never as the paper's model, with the decision difference against route A or float32 PyTorch measured first (section 10.7).

### 10.3 Request and answer

```ts
interface ImageQuestion { name?: string; instruction: string; options: string[]; context?: string }
decide({ image: { rgb: Uint8Array | Uint8ClampedArray; width: number; height: number; channels: 3 | 4 }, questions: ImageQuestion[], context?: string })
```

- `context` is the shared public text after the image (the reference's `state`); a question's own `context` is not used by the reference and is refused. Options are strings (no per-option description in v1); 2 to 16 of them; more is rejected,
  not truncated. Options and instructions are English (the reference is English only). Alpha is dropped, not composited (PIL `convert("RGB")`).
- The answer is a `ChoiceAnswer` of the text path (`probability` of the chosen option, `margin = p1 - p2`, `rawMargin` = logit gap) plus `modality: "image"`, `task: "choice"`, `decision` (the chosen option string), `probabilities` per option, `path: "s1"`.
  Probabilities are the **raw** softmax of the reference (`readout.temperature` = 1); they are overconfident, there is no abstention outside the options supplied (add an option such as `not enough evidence`), and calibration is the caller's.

### 10.4 Prompt and readout

Text (`prefixText`, `suffixText` in `imagePrep.ts`, checked against the reference tokenizer on three prompts): `<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<|vision_start|>` + one `<|image_pad|>` per merged image
token + `<|vision_end|>` + (`\n{context}` if any) is the prefix; each question adds `\n\nQuestion: {instruction}\nOptions:\nA. {option}\nB. ...\nReply with the single letter of the correct option.<|im_end|>\n<|im_start|>assistant\nAnswer:`.
`SYSTEM` is "You are a visual decision model. You inspect the image and answer each question by choosing exactly one of the given options. You never explain."

- **Tokenisation.** Prefix and suffix are tokenised separately and concatenated, as the published inference does. Training tokenised the joint text; the two differ when a token spans the boundary: a context ending in "." followed by
  "\n\n" merges into one token in the joint text (fixture, first prompt). The reader follows the inference path and the fixtures record the difference.
- **Option tokens.** `" A"` to `" P"` are single tokens (ids 362, 425, 356, 422, 468, 434, 479, 472, 358, 619, 730, 444, 386, 451, 506, 393 in the Qwen3-VL tokenizer).
- **Readout position.** The last token of the suffix (the colon). The graph is edited to output the final-norm hidden state (`vl-graph text --hidden-output`), not logits: the score is read in JavaScript as `hidden . row`,
  with the 16 rows in `readout.rows` (float16, 80 KB, copied from the embedding table), accumulated in float64. This removes the int4 `lm_head` (200 MB) and the `[S, 151936]` logits of a prefill, and scores with the exact (tied)
  head instead of its int4 copy. The public graph's own `logits` is the fallback and is measured as a second reading.
- **Token embeddings** are looked up in JavaScript (`embeddings`: float16 table of 778 MB, exact, or the public int4 block-128 table of 207 MB, rows dequantised on demand; the choice is made on the parity measurement, section 10.7).

### 10.5 Branches, positions, cache

1. Preprocess the image (10.6), run the vision graph once: `features [N, hidden]` and the DeepStack features.
2. Build the prefix embeddings (token rows, image rows scattered into the `<|image_pad|>` slots), the three DeepStack tensors (zero outside the image slots) and the M-RoPE positions of the **full** sequence of the first question;
   the prefix positions are those of the full sequence sliced (`mropePositions`, equal to `get_rope_index` on four grids). Run the text graph once on the prefix: cache snapshot.
3. Each question: copy the prefix cache (a real cost, 147 KB per token in float16), run the suffix with positions `nextPosition(prefix) + i` on all three axes, zero DeepStack inputs, read `hidden[last]`.
4. v1 runs the branches one after the other (the graph is single-sequence); a padded batch of suffixes is not built. Prefix sharing is exact by construction (the reference's parity is the same: only arithmetic differences).

### 10.6 Pre-processing and the pixel budget

Reference processor: `smart_resize` to multiples of 32 within [65,536, 16,777,216] pixels, PIL bicubic, `(x/255 - 0.5)/0.5`, 16 x 16 patches in 2 x 2 merge groups, temporal patch 2 (the image repeated). The reference ran at **native resolution**
(the authors report that their `max_pixels` setting was ignored: a 500 x 500 image gave 256 visual tokens, 25 x 25 patches to 32 x 32 after resize). The browser cannot take 16 M pixels: the manifest carries `minPixels` / `maxPixels`
(proposal 65,536 / 786,432 = at most 768 merged tokens) and `referenceMaxPixels`. A package whose `maxPixels` is below what the image needs **resizes down**, which is a deviation from the reference; accuracy at the budget against native
resolution is a measurement (10.7), not an assumption. Implemented in JavaScript: `smartResize` (30 cases identical), `resizeBicubicRGB` (Pillow's algorithm: byte-identical on 5 images, Pillow 13.0.0), `patchify` (values within 1e-6). The
reference's fast processor (torchvision, antialiased bicubic) can differ from Pillow by a few levels: not measured.

### 10.7 Build and parity (tolerances fixed before the runs)

Pipeline (`converter/`): `vl-swap --verify` (no LoRA: every initializer the swap would write, the norms and `lm_head` recomputed from the base shards and compared byte for byte with the public file; the cast chain is probed on `lm_head`),
`vl-swap` (LoRA fused in float32 into the 252 matrices, RTN block 32 symmetric with the quantizer of ONNX Runtime, written in place; `--split` rewrites the external data into pieces of at most 1.9 GiB), `vl-graph text`, `vl-graph vision`.
The lm_head is the oracle because the adapter never touches it. Measured (range reads of the public files, no download of the weights): the quantizer of `vl-swap` (`rtn_q4`: ties up, the negative value when +x and -x tie for the
largest magnitude) reproduces the public int4 bytes and scales exactly, 0 differing bytes, on 4096 rows of `lm_head` and on the whole of layer 0 q_proj, layer 17 v_proj and layer 35 down_proj, in both the cpu and the cuda build; the
ONNX Runtime 1.31 quantizer function differs from them on 0.03% of the bytes (exact ties). The layer norms are byte-equal to bf16 widened to float32. If the full `--verify` run finds a difference, the swap still runs and the report says `oracle: false`:
fidelity then rests on the checks below, not on byte equality. The cast chain matters only for the fused weights (the base weights are bf16, exactly representable in float16): the default is float32 into the quantizer (no intermediate rounding);
a float16-scale build stores the scale rounded to float16 and computes q with that stored scale.

| check | oracle | tolerance |
|---|---|---|
| quantizer | independent numpy reading of the same RTN | packed bytes and scales identical (tested) |
| LoRA fusion, one layer | PEFT forward, float32 | max abs difference 1e-5 (tested) |
| swapped graph, synthetic | ONNX Runtime CPU vs `x @ dequant(requantised fused W)` | 1e-4; vs unquantised fused W, relative Frobenius 0.20 (tested) |
| untouched initializers | the public file | byte-identical (tested; on the real files: the heavy run) |
| pure parts | the reference processor, tokenizer, `get_rope_index` | exact, or 1e-6 for float32 values (tested) |
| graph edits | numpy model; on the real graph, PyTorch hidden states after blocks 5/11/17 and layers 0 to 2 | 1e-5 text / 1e-4 vision synthetic; real: float32 PyTorch within 1e-3 relative on the three DeepStack features, fp16 build 1e-2 |
| end to end (the 4B) | PyTorch float32, `run_independent`, LoRA applied, same pixels | on at least 200 questions of GQA-style and SNLI-VE-style items not used for any choice here: median abs difference of the chosen option's probability <= 0.02, 95th percentile <= 0.10, decision flips <= 3%, accuracy within 1 point |
| budget | the same questions at native resolution | accuracy drop at `maxPixels` reported; no claim below that |

Reported, not required: the public graphs without DeepStack (route B) against the same oracle; the int4 `lm_head` reading against the exact rows; the int4 embedding table against the float16 table.

### 10.8 Limits and risks

- Everything about the 4B at run time is unmeasured: parity, accuracy, speed, memory. Memory arithmetic only: text int4 2.3 to 2.5 GB, vision 0.2 to 0.7 GB, float16 embedding table 0.78 GB, cache 147 KB per token and per branch
  (a 700-token prefix with 8 branches is about 0.9 GB): about 4 GB, on a browser that measured 3.9 GB for a different 4B text model. It may not fit a 6 GB WebGPU device.
- WebGPU: the text graph uses the contrib operators the audio decoder of this library already runs (`MatMulNBits`, `GroupQueryAttention`, `RotaryEmbedding`, `SkipSimplifiedLayerNormalization`) but with 3-axis `position_ids`, fp16 io in
  the cuda layout and an `accuracy_level` 4 attribute (cpu layout); the vision graph has 24 `Loop` nodes, 3-D convolution and (cpu layout) `GatherBlockQuantized`, which may fall back to the CPU provider or fail. None of this was run.
- onnxruntime-web reads at most 2 GB per external-data file and a graph's single buffers are limited by the adapter's `maxBufferSize`: the 2.3 to 2.5 GB data is split (`--split`); the largest single tensor is the embedding-derived `lm_head`
  (194 MB, removed by `--hidden-output`).
- The adapter helps only on task families seen in training; counting is weak; English only; K trained in {2, 4}; raw probabilities are overconfident; option order changes decisions.
- Licence: base model and adapter Apache-2.0; training data (GQA, SNLI-VE over Flickr30k) have their own terms; a published build needs both licence texts and a notice that the weights were fused and requantised. Nothing is published from here.

### 10.9 Measured (26 November 2026; one Apple-silicon Mac; Chrome headless, onnxruntime-web 1.31.1; n is small and the oracle is bf16)

Build: public text graph (cuda layout, fp16 io, or cpu layout, fp32 io) with the LoRA fused and swapped (`vl-swap`: `--verify` 651 of 651 compared initializers byte-equal, both layouts swapped 504 initializers = 252 modules), DeepStack edits, hidden output, declared intermediate shapes dropped (onnxruntime-web refuses the rank-0 shapes the public graph declares for rank-1 tensors), pixel budget 65,536 to 200,704.
Questions: ScienceQA test split, image questions with 2 to 8 options (neither the reader's training families, GQA and SNLI-VE, nor any choice made here); the first 60 for the oracle comparison, 240 for accuracy.

**Deviations from 10.7, stated plainly.** (a) The oracle is PyTorch **bfloat16** on CPU (float32 needs ~18 GB; the Mac has 16), so the bf16 error of the oracle is mixed with the int4 error of the reader and cannot be separated. (b) n = 60 for the oracle (about 25 minutes on this CPU), n = 240 for accuracy and for the provider comparison. (c) The accuracy gate "within 1 point of the oracle" was only checked at n = 60 (0.90 and 0.90).

| comparison | n | median abs dp | p95 | max | decision flips | accuracy (first / second) |
|---|---|---|---|---|---|---|
| ONNX Runtime CPU, faithful build (cuda layout) vs torch bf16 + adapter | 60 | 0.0014 | 0.198 | 0.458 | 4 (6.7%) | 0.90 / 0.90 |
| ONNX Runtime CPU, fp32-activation matmul (`accuracy_level` 0) vs the same oracle | 60 | 0.0031 | 0.263 | 0.642 | 4 (6.7%) | 0.90 / 0.90 |
| ONNX Runtime CPU, public graphs without DeepStack (route B) vs the same oracle | 60 | 0.0029 | 0.436 | 0.685 | 6 (10%) | 0.90 / 0.85 |
| Chrome (text on WebGPU, vision on WASM, cpu layout fp32) vs ONNX Runtime CPU, same graphs | 240 | 0.0011 | 0.136 | 0.284 | 7 (2.9%) | 0.917 / 0.925 |
| Chrome (same) vs ONNX Runtime CPU | 60 | 0.0002 | 0.022 | 0.150 | 0 | 0.90 / 0.90 |

Accuracy against the labels, n = 240 (Wilson 95%): ONNX Runtime CPU 0.917 (0.875 to 0.945); Chrome 0.925. Chance for these questions is 0.36. This is far from the reference's four benchmarks and says only that the build reads images and options sensibly.

**The tolerances of 10.7 (p95 <= 0.10, flips <= 3%) against the oracle were not met** (p95 0.198 and 6.7% flips at n = 60). The int4 build is not the oracle's arithmetic, the oracle is bf16, and at n = 60 four flips are four questions; what can be said is that the faithful build is closer than route B on every number, which supports the DeepStack finding (10.2), and that accuracy did not move. The reader as built meets the browser-against-CPU gate on the same graphs (p95 0.136 and 2.9% flips at n = 240, 0.022 and 0 at n = 60: the n = 240 flips are near-ties of the same graph under different kernels, which is the size of the provider noise).

**Browser.** Text on WebGPU is correct: with the vision graph on the WASM provider the browser reproduces ONNX Runtime CPU (table above). `loadMedia({ visionDevice: "wasm" })` selects that split. Time per question (one question per image, text on WebGPU, vision on single-thread WASM, 240 questions): median 15.6 s, p95 28 s, of which vision 10.8 s and the branch 4.3 s; with the vision graph on WebGPU the same questions took 3.7 s but gave wrong answers (below). Memory: the sampled resident set of all Chrome processes peaked near 3 GB (the sampler also counts any other Chrome open; it stopped at 30 minutes, so this is a lower bound). All-WASM (both graphs) is correct and takes 65 to 85 s per question.

**Vision graph on WebGPU is wrong and the cause is not isolated.** Same inputs, same graph: the output of the interpolated position-embedding path differs from CPU (`Gather` nodes `embedding`, `embedding_1` return the rows of `embedding_3`; the indices they read, captured as graph outputs, equal CPU's; casting the indices to int32, optimisation level `basic`, and one concatenated Gather in place of four do not fix it; the patch convolution before it matches). A stand-alone model with the same four Gathers over a random fp16 [2304, 1024] table and the same indices is correct on WebGPU and WASM (`recipes/visual-jev-image/browser-repro-gather.html`), so the bug needs the rest of the graph (suspect: buffer reuse around the `Reshape` aliases that feed the Gathers). The full-WebGPU run gave 0.75 accuracy against 0.90 at n = 60. Not reduced further; a report for the onnxruntime project (not filed) would say: onnxruntime-web 1.31.1, WebGPU EP, Chrome headless on Apple silicon, `vision.onnx` of onnx-community/Qwen3-VL-4B-Instruct-ONNX (cuda layout), outputs `embedding` and `embedding_1` of the position-embedding Gathers equal `embedding_3` on WebGPU and are correct on WASM and CPU.

**What was ruled out on the text graph.** Chunked prefill is not needed: a 200-token prefill in blocks of 16, 32 and 40 tokens gives the same numbers as one block, and the text graph on WebGPU equals the same graph on CPU with `accuracy_level` 0 to the last digit. An earlier comparison of WebGPU against CPU with `accuracy_level` 4 (int8-quantised activations on CPU) looked like a divergence from layer 17 on; it is the CPU's activation quantisation, not a WebGPU error, and the CPU with `accuracy_level` 0 gives the same decisions as the default (table).
