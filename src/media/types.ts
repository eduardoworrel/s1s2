// Media packages (audio): manifest blocks, request and answer types. Design: spec/MULTIMODAL.md.
// The text types (manifest.ts, readout.ts, questions.ts) are reused, not copied: a media answer IS a PredicateAnswer
// or a ChoiceAnswer with the same margin / rawMargin definitions, plus what the media path adds.

import type { PinnedFile, EscalationSpec, Provenance } from "../manifest.js";
import type { PredicateAnswer, ChoiceAnswer } from "../readout.js";

/** What `decide` accepts besides the question. "text" alone is the existing manifest; a media package lists its media. */
export type Modality = "text" | "audio" | "image";
export const MODALITIES: readonly Modality[] = ["text", "audio", "image"];

// ---- audio: log-mel pre-processor, tower, readers ------------------------------------------------------------------

/** Whisper-style log-mel, computed in JS (src/media/mel.ts). Every number is part of the contract. */
export interface LogMelSpec {
  kind: "log-mel";
  sampleRate: 16000;
  /** Length of the window the tower was exported for (right-aligned, left zero-padded; longer audio keeps its last seconds). */
  windowSeconds: number;
  nFft: number; hop: number; nMels: number;
  /** Periodic Hann window, reflect padding of nFft/2 on both ends, power spectrum, Slaney mel filters (norm and scale). */
  window: "hann-periodic"; padding: "reflect"; filters: "slaney";
  /** log10(max(x, floor)), floor at (max - dynamicRange), then (x + shift) / divide; the last STFT frame is dropped. */
  log: { floor: number; dynamicRange: number; shift: number; divide: number };
  /** Mel frames the tower takes: windowSeconds * sampleRate / hop. */
  frames: number;
}

/** One tap of the tower graph: hidden states after `layer` (1-based), or the final projection. */
export interface TowerTap { layer: number | "final"; frames: number; dim: number }

export interface AudioTowerSpec {
  runtime: "onnx";
  /** Graph with its weights inside (one file; onnxruntime-web reads up to 2 GB). */
  file: PinnedFile;
  input: { name: string; shape: [number, number] };
  /** Output name -> tap. Readers name the tap they read. */
  outputs: Record<string, TowerTap>;
  devices: ("webgpu" | "wasm" | "cpu")[];
}

/** A span of the window in seconds, [from, to). */
export interface Span { from: number; to: number }

/**
 * How frames of one tap become one feature vector. Features are rounded to float16 first (the readers were fit on
 * float16-stored features), then pooled in float64. A frame belongs to a span when its centre sample does:
 * floor((i + 0.5) * windowSamples / frames).
 */
export type Pooling =
  /** Mean of the frames in the last `seconds` of the window that hold real audio. Needs at least one. */
  | { kind: "tail-mean"; seconds: number }
  /** [x*y, |x-y|] of the L2-normalised means of two spans. Needs a full window. */
  | { kind: "pair"; a: Span; b: Span }
  /** [pair(pre, last), pre, last] with pre = [0, split), last = [split, window). Needs a full window. */
  | { kind: "pair-split"; split: number }
  /** [mean, std] over all frames. Needs a full window. */
  | { kind: "mean-std" };

/** Calibration of the raw logit (or logits). Tensor names refer to the reader's safetensors file. */
export type Calibration =
  /** p = sigmoid(logit * coefficient + intercept). */
  | { kind: "platt"; coefficient: string; intercept: string }
  /** p = softmax(logits / T). */
  | { kind: "temperature"; tensor: string }
  | { kind: "none" };

interface ReaderBase {
  /** Tower output the reader reads (a key of audio.tower.outputs). */
  tap: string;
  pooling: Pooling;
  weights: PinnedFile;
  /** Tensor names inside `weights`; absent optional ones mean "not used". */
  tensors: { coef?: string; intercept?: string; mean?: string; scale?: string; layer0Weight?: string; layer0Bias?: string; layer1Weight?: string; layer1Bias?: string };
  calibration: Calibration;
}

/** standardise (mean, scale) -> W x + b -> calibration. Binary (one row, predicate) or K rows (choice). */
export interface PooledLinearReader extends ReaderBase {
  kind: "pooled-linear";
  type: "predicate" | "choice";
  /** choice: one label per row. */
  labels?: string[];
}

/** pooled feature -> Linear(in, h) -> ReLU -> Linear(h, 1) -> sigmoid -> logit -> Platt. Layout of layer weights: [in, out]. */
export interface PooledMlpReader extends ReaderBase { kind: "pooled-mlp"; type: "predicate" }

/**
 * Speech question answered by a small graph over the tower's tap: inputs {tap, frame mask, question vector}; output one
 * score; p = sigmoid(score / temperature). The question vector is the pre-computed encoding of ONE canonical question.
 */
export interface SpeechHeadReader {
  kind: "speech-head";
  type: "predicate";
  tap: string;
  graph: PinnedFile;
  /** Little-endian float32 [dim]. */
  question: PinnedFile & { dim: number; prompt: string };
  inputs: { tap: string; mask: string; question: string; output: string };
  /** Frames the graph pools to (the mask has this length). */
  frames: number;
  temperature: number;
  /** Decision threshold on p, from the head's validation (not 0.5). */
  threshold: number;
}

/** "Which language": closed-set scoring with the Qwen3-ASR decoder over the tower's final vectors (spec section 9). Needs `audio.decoder`. */
export interface DecoderLanguageReader { kind: "decoder-language"; type: "choice"; tap: string }
/** "Was this word said": the term is looked for in five decoder hypotheses (greedy + beam 4). Needs `audio.decoder`. */
export interface DecoderTermReader { kind: "decoder-term"; type: "predicate"; tap: string }

export type AudioReader = PooledLinearReader | PooledMlpReader | SpeechHeadReader | DecoderLanguageReader | DecoderTermReader;

/** The Qwen3-ASR text decoder as an ONNX Runtime GenAI build that reads embeddings (converter: `asr-decoder`). */
export interface DecoderSpec {
  runtime: "onnx-genai";
  /** model.onnx and model.onnx.data (the graph names its data file): the WebGPU build. */
  graph: PinnedFile; data: PinnedFile;
  /**
   * The same decoder built for CPU kernels (Node CPU and browser WASM): the WebGPU build fuses the per-head query/key norm into its
   * attention node, which only the WebGPU kernel implements. Absent: the decoder runs on WebGPU only.
   */
  cpuBuild?: { graph: PinnedFile; data: PinnedFile };
  /** Float16 little-endian [vocab, hidden]: rows of the prompt's text tokens. */
  embeddings: PinnedFile;
  /** JSON { tokens: string[vocab], added: number[], special: number[] }: byte-level pieces, to decode generated ids. */
  tokens: PinnedFile;
  model: { layers: number; kvHeads: number; headDim: number; hidden: number; vocab: number };
  /** Token ids before and after the audio rows (their number is audioFrames) and the id of "language". */
  prompt: { before: number[]; after: number[]; audioFrames: number; languageToken: number };
  eos: number[];
  /** Language code -> name and the token ids of " Name" after "language". */
  languages: Record<string, { name: string; ids: number[] }>;
  term: { beams: number; maxNewTokens: number; lengthPenalty: number; repetitionThreshold: number };
  devices: ("webgpu" | "wasm" | "cpu")[];
}

export interface AudioSpec {
  preprocess: LogMelSpec;
  tower: AudioTowerSpec;
  /** Task name -> reader. The request names the task. */
  readers: Record<string, AudioReader>;
  /** Present when readers of kind decoder-language / decoder-term are listed. */
  decoder?: DecoderSpec;
  /** Free text: what each task means and was validated on. */
  notes?: string;
  /** Measured fidelity against the reference implementation, per task (free-form record). */
  fidelity?: Record<string, unknown>;
}

// ---- image: Qwen3-VL decision reader (spec section 10) ------------------------------------------------------------------

/** Pre-processor in JavaScript (src/media/imagePrep.ts): smart_resize, Pillow bicubic, normalise, flat patch layout of the Qwen processors. */
export interface ImagePreprocessSpec {
  kind: "qwen-vl-patches";
  patchSize: number; mergeSize: number; temporalPatchSize: number;
  mean: number[]; std: number[];
  resample: "pillow-bicubic";
  /** Pixel budget of THIS package. The reference ran at native resolution (longest_edge 16,777,216): a budget below that is a documented deviation. */
  minPixels: number; maxPixels: number;
  /** The reference processor's own maximum, for the record (16777216). */
  referenceMaxPixels?: number;
}

/** The vision tower as an ONNX graph. It must output the merged features AND the DeepStack features (the public build outputs only the first: `vl-graph vision`). */
export interface ImageVisionSpec {
  runtime: "onnx";
  graph: PinnedFile; data?: PinnedFile[];
  inputs: { pixelValues: string; grid: string };
  outputs: { features: string; deepstack: string[] };
  /**
   * The position embedding as a graph input computed by the reader (`vl-graph vision --external-posembed`): the learned table (float16 [side * side, hidden]) is resampled to
   * the image grid in JavaScript (`positionEmbeddings`). Needed on the WebGPU provider of onnxruntime-web 1.31.1, whose own Gather of that path is wrong.
   */
  posEmbeds?: { input: string; table: PinnedFile; side: number };
  devices: ("webgpu" | "wasm" | "cpu")[];
}

/** The text decoder as an ONNX graph with a KV cache (public `text.onnx` + LoRA swap + `vl-graph text`): takes embeddings and DeepStack rows, outputs the final-norm hidden state. */
export interface ImageTextSpec {
  runtime: "onnx";
  graph: PinnedFile;
  /** External-data pieces, at most 2 GB each (the graph names them by file name). */
  data: PinnedFile[];
  inputs: { embeds: string; attentionMask: string; positionIds: string; deepstack: string[] };
  /** Cache names with %d for the layer: past_key_values.%d.key / .value in, present.%d.key / .value out. */
  cache: { pastKey: string; pastValue: string; presentKey: string; presentValue: string };
  output: { hidden: string };
  model: { layers: number; kvHeads: number; headDim: number; hidden: number; vocab: number };
  /** Element type of embeds / deepstack / hidden / cache. */
  ioDtype: "float32" | "float16";
  devices: ("webgpu" | "wasm" | "cpu")[];
}

/** Token embedding rows, read in JavaScript (no graph): the prompt's tokens are looked up here, the image rows come from the vision tower. */
export interface ImageEmbeddingSpec {
  file: PinnedFile;
  /** "f16": little-endian float16 [vocab, hidden] (exact for bf16 weights). "q4-block128": the public int4 gather table (symmetric, block 128 along hidden): rows are dequantised on demand. */
  format: "f16" | "q4-block128";
  /** Required by "q4-block128": the float scales file ([vocab, hidden / 128] float32). */
  scales?: PinnedFile;
}

export interface ImageTokenizerSpec { tokenizer: PinnedFile; config: PinnedFile }

/** Ids of the markers of the prompt; the text is fixed by the reader (prefixText / suffixText). */
export interface ImagePromptSpec { imagePad: number; visionStart: number; visionEnd: number; imStart: number; imEnd: number }

/** "Which option": the LM-head logit of the token of each option letter at the readout position, softmax over the K options supplied. */
export interface ImageReadoutSpec {
  kind: "option-letters";
  /** Token ids of " A", " B", ... (single tokens), one per possible option. */
  tokenIds: number[];
  /** Float16 little-endian [tokenIds.length, hidden]: the embedding rows of those tokens (the LM head is tied to the embedding): score = hidden . row. */
  rows: PinnedFile;
  minOptions: number; maxOptions: number;
  /** softmax(logits / temperature); 1 = the raw probabilities of the reference (overconfident: calibrate on your task). */
  temperature: number;
}

export interface ImageSpec {
  preprocess: ImagePreprocessSpec;
  vision: ImageVisionSpec;
  text: ImageTextSpec;
  embeddings: ImageEmbeddingSpec;
  tokenizer: ImageTokenizerSpec;
  prompt: ImagePromptSpec;
  readout: ImageReadoutSpec;
  notes?: string;
  /** Measured fidelity against the reference (free-form record). */
  fidelity?: Record<string, unknown>;
}

// ---- manifest --------------------------------------------------------------------------------------------------------

/** A media package manifest. Text packages (Manifest) are unchanged; a manifest with text AND media is a Manifest with these keys. */
export interface MediaManifest {
  schema: "s1s2.manifest/1";
  id: string;
  /** Inputs `decide` takes. Default ["text"] (absent = a text manifest). A media package lists its media: ["audio"] or ["image"]. */
  modalities: Modality[];
  /** Where the pinned paths live: a Hub repo id, or any name that `baseUrls` / `localDirs` map (unpublished packages). */
  repo: string;
  /** 40-hex commit sha of `repo`, or "local" for an unpublished package (then nothing is fetched from the Hub). */
  revision: string;
  audio?: AudioSpec;
  image?: ImageSpec;
  escalation?: EscalationSpec;
  license?: string;
  provenance?: Provenance;
}

// ---- requests and answers --------------------------------------------------------------------------------------------

export interface AudioInput {
  /** Mono float32 in [-1, 1] at 16 kHz. Other rates are refused: resample first (spec limits). */
  samples: Float32Array;
  sampleRate: number;
}

export interface MediaDecisionRequest {
  audio?: AudioInput;
  /** Task names, or `{ task }` objects; `{ task: "term", term }` carries the word; `{ task: "language", options }` a closed set of codes. */
  questions: (string | { name?: string; task?: string; term?: string; options?: string[] })[];
}

/** Options of `AudioModel.transcribe`. */
export interface TranscribeOptions {
  /** Beam search (the manifest's `term.beams`, 4) instead of greedy decoding: about four times slower. Default false. */
  beam?: boolean;
  /** Most tokens to generate (default: the manifest's `term.maxNewTokens`, 256). */
  maxNewTokens?: number;
  /** Rejects with an `AbortError` before the next step when aborted. */
  signal?: AbortSignal;
}

/** Free transcription of one window: what the decoder writes, without a term to look for. */
export interface Transcription {
  /** The transcript, without the "language X" prefix the model writes before it; "" for silence. */
  text: string;
  /** Code (a key of `audio.decoder.languages`) of the language the model wrote before the transcript; null when it wrote none or a name with no code. */
  language: string | null;
  method: "greedy" | "beam";
  /** Beam: the beams' transcripts, best first (`text` is the first). Absent for greedy. */
  hypotheses?: string[];
  /** Generated tokens of `text`, the end token not counted. */
  tokens: number;
  /** "stop": the model ended; "length": `maxNewTokens` cut it. */
  finishReason: "stop" | "length";
  cost: MediaCost;
}

export interface MediaCost {
  /** Wall time of the whole decide() call, and its parts: mel, tower, readers. */
  ms: number;
  parts: Record<string, number>;
  /** Forward passes of the big graph (audio tower: 1 for any number of questions). */
  passes: number;
  backend: string;
  device: string;
}

/** Fields every media answer adds to the text answer. */
export interface MediaFields {
  modality: "audio";
  task: string;
  /** Decision the reader's own threshold gives (VAD uses 0.225, not 0.5); predicate: "yes" | "no"; choice: the label. */
  decision: string;
  /** Predicate: the threshold p was compared with. */
  threshold?: number;
  /** Predicate: probability before calibration (pooled readers: sigmoid of the raw logit; the speech head: absent). */
  uncalibrated?: number;
  /** term: the five hypotheses (greedy first, then four beams) the fraction was counted on. */
  hypotheses?: string[];
}
export type MediaPredicateAnswer = PredicateAnswer & MediaFields & { available: true };
export type MediaChoiceAnswer = ChoiceAnswer & MediaFields & { available: true };
/** A question the input cannot answer (e.g. a speaker question on less than a full window). No number is made up. */
export interface UnavailableAnswer { name: string; task: string; modality: "audio"; available: false; reason: string }
export type MediaAnswer = MediaPredicateAnswer | MediaChoiceAnswer | UnavailableAnswer;

export interface MediaDecision {
  answers: MediaAnswer[];
  cost: MediaCost;
}
