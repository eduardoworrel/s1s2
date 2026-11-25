# Media packages: audio (experimental)

Status: draft, written before the code in `src/media/`. Package spec 0.2-draft stays as it is; this file adds three optional
top-level keys to the manifest (`modalities`, `audio`) and one new entry point (`loadMedia`). The numbers below were
measured in the reference runs on the builds named here; nothing else is claimed.

## 1. What changes and what does not

| | Text (unchanged) | Media (new) |
|---|---|---|
| Manifest | `s1s2.manifest/1` with tokenizer, encoding, markers, logits, heads, builds | the same schema id; a manifest whose `modalities` lacks `"text"` has none of those sections and carries `audio` |
| Load | `S1S2.load` | `loadMedia` (returns an `AudioModel`); `S1S2.load` and `validateManifest` refuse a media manifest with a message that says so |
| Request | `{ input, questions }` | `{ audio, questions }` |
| Answer | `PredicateAnswer` / `ChoiceAnswer` | the same objects (same `probability`, `confidence`, `margin`, `rawMargin`, `path: "s1"`) plus `modality`, `task`, `decision`, `threshold?` |
| Margin | `margin = p(1st) - p(2nd)` (predicate: `|2p - 1|`); `rawMargin` = gap in score space | identical definitions. The decision of a predicate uses the reader's own threshold (voice activity: 0.225), `margin` is still relative to 0.5 |
| Shared work | prefix shared by questions on one input (snapshot + prefillFrom) | ONE tower pass for any number of questions (`encode` once, `read` many times) |
| Escalation | margin policy (`escalation`) | the manifest key is accepted and ignored; no media threshold is measured |
| Evidence / pointer heads, `generate()` | yes | no |

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
                                 questions: string[] }
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
float32), outputs `h9`, `h12`, `h18` (104 frames x 896, after layers 9, 12, 18) and the final projection (104 x 1024, unused here).

Readers (`audio.readers`, task name -> reader):

| task | reader | tap | pooling | calibration |
|---|---|---|---|---|
| `voice_activity` | `speech-head` (ONNX, 9.7 MB + a pinned 256-float question vector of ONE canonical question) | `h9` | inside the graph (100 frames, frame mask) | `p = sigmoid(score / 0.654)`, decide at 0.225 |
| `turn_end` | `pooled-linear` | `h18` | `tail-mean` 2 s of real audio | standardise (float64) -> linear -> Platt |
| `same_speaker` | `pooled-linear` | `h12` | `pair` 0-4 s / 4-8 s | standardise -> linear -> Platt |
| `speaker_change` | `pooled-mlp` (3584 -> 256 -> 1) | `h12` | `pair-split` at 6 s | sigmoid -> logit -> Platt |
| `speaker_count` | `pooled-linear`, 4 rows, labels 0 / 1 / 2 / 3+ | `h12` | `mean-std` | softmax with a fitted temperature |

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
```

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

- **Audio language and spoken-word questions are not here.** In Marimba they need the Qwen3-ASR decoder (closed-set language scoring,
  beam hypotheses for a term). The package has voice activity, turn end, same speaker, speaker change and speaker count only.
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

- Apache-2.0 is declared by `Qwen/Qwen3-ASR-0.6B`, `charcot-ai/marimba` (head and readout weights; its code repository has an Apache-2.0
  `LICENSE`). Some of
  these Hugging Face repositories have no `LICENSE` file, only the card field.
- Training data have their own terms: the Smart Turn v3.2 data (turn-end readout) is CC BY 4.0. Whether those terms reach the trained
  weights is not stated by the sources.
- FLEURS (the example clips and test fixtures) is CC BY 4.0; attribution is in `examples/audio-turn/clips/ATTRIBUTION.md`.
