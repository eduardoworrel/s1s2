// Media packages (audio): manifest blocks, request and answer types. Design: spec/MULTIMODAL.md.
// The text types (manifest.ts, readout.ts, questions.ts) are reused, not copied: a media answer IS a PredicateAnswer
// or a ChoiceAnswer with the same margin / rawMargin definitions, plus what the media path adds.

import type { PinnedFile, EscalationSpec, Provenance } from "../manifest.js";
import type { PredicateAnswer, ChoiceAnswer } from "../readout.js";

/** What `decide` accepts besides the question. "text" alone is the existing manifest; a media package lists its media. */
export type Modality = "text" | "audio";
export const MODALITIES: readonly Modality[] = ["text", "audio"];

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

export type AudioReader = PooledLinearReader | PooledMlpReader | SpeechHeadReader;

export interface AudioSpec {
  preprocess: LogMelSpec;
  tower: AudioTowerSpec;
  /** Task name -> reader. The request names the task. */
  readers: Record<string, AudioReader>;
  /** Free text: what each task means and was validated on. */
  notes?: string;
  /** Measured fidelity against the reference implementation, per task (free-form record). */
  fidelity?: Record<string, unknown>;
}

// ---- manifest --------------------------------------------------------------------------------------------------------

/** A media package manifest. Text packages (Manifest) are unchanged; a manifest with text AND media is a Manifest with these keys. */
export interface MediaManifest {
  schema: "s1s2.manifest/1";
  id: string;
  /** Inputs `decide` takes. Default ["text"] (absent = a text manifest). A media package lists its media: ["audio"]. */
  modalities: Modality[];
  /** Where the pinned paths live: a Hub repo id, or any name that `baseUrls` / `localDirs` map (unpublished packages). */
  repo: string;
  /** 40-hex commit sha of `repo`, or "local" for an unpublished package (then nothing is fetched from the Hub). */
  revision: string;
  audio?: AudioSpec;
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
  /** Task names (or `{ task }` objects). */
  questions: (string | { name?: string; task?: string })[];
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
