#!/usr/bin/env node
// Writes manifests/marimba-audio.json from a weights directory: every pinned file gets its
// sha256 and size from the file as found. Usage: node recipes/make-manifests.mjs <weights dir> (default .weights)
// The weights dir holds marimba-audio/ (layout in recipes/marimba-audio/README.md). Nothing is downloaded.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// The package lives on the Hub; `revision` is the commit that holds the weight files.
const HUB_REPO = "eduardoworrel/Marimba-ONNX-S1S2", HUB_REVISION = "635924eb45bde6197a9be2e7e0cabc69f2930248";
const W = path.resolve(process.argv[2] ?? path.join(ROOT, ".weights"));

async function pin(dir, rel) {
  const f = path.join(dir, rel), size = fs.statSync(f).size, h = crypto.createHash("sha256");
  for await (const c of fs.createReadStream(f, { highWaterMark: 8 << 20 })) h.update(c);
  return { path: rel, sha256: h.digest("hex"), size };
}

// ---- audio ----
{
  const d = path.join(W, "marimba-audio");
  const P = async (r) => pin(d, r);
  const same = { type: "predicate", tap: "h12", weights: await P("diarization/same_speaker.safetensors"), kind: "pooled-linear" };
  const audio = {
    preprocess: { kind: "log-mel", sampleRate: 16000, windowSeconds: 8, nFft: 400, hop: 160, nMels: 128, window: "hann-periodic", padding: "reflect", filters: "slaney", log: { floor: 1e-10, dynamicRange: 8, shift: 4, divide: 4 }, frames: 800 },
    tower: {
      runtime: "onnx", file: await P("audio_tower.onnx"), input: { name: "mel", shape: [128, 800] },
      outputs: { h9: { layer: 9, frames: 104, dim: 896 }, h12: { layer: 12, frames: 104, dim: 896 }, h18: { layer: 18, frames: 104, dim: 896 }, audio_features: { layer: "final", frames: 104, dim: 1024 } },
      devices: ["webgpu", "wasm", "cpu"],
    },
    readers: {
      voice_activity: {
        kind: "speech-head", type: "predicate", tap: "h9", graph: await P("vad_head.onnx"),
        question: { ...(await P("vad_question.f32")), dim: 256, prompt: "Há fala humana neste áudio?\nA. sim\nB. não\nResponda só com a letra da opção." },
        inputs: { tap: "h9", mask: "fmask", question: "qpool", output: "score" }, frames: 100, temperature: 0.6540604692061299, threshold: 0.225,
      },
      turn_end: {
        kind: "pooled-linear", type: "predicate", tap: "h18", pooling: { kind: "tail-mean", seconds: 2 },
        weights: await P("turn_end/readout.safetensors"), tensors: { coef: "coef", intercept: "intercept", mean: "mean", scale: "scale" },
        calibration: { kind: "platt", coefficient: "platt_coef", intercept: "platt_intercept" },
      },
      same_speaker: {
        ...same, pooling: { kind: "pair", a: { from: 0, to: 4 }, b: { from: 4, to: 8 } },
        tensors: { coef: "coef", intercept: "intercept", mean: "scaler_mean", scale: "scaler_scale" },
        calibration: { kind: "platt", coefficient: "platt_coefficient", intercept: "platt_intercept" },
      },
      speaker_change: {
        kind: "pooled-mlp", type: "predicate", tap: "h12", pooling: { kind: "pair-split", split: 6 },
        weights: await P("diarization/speaker_change.safetensors"),
        tensors: { layer0Weight: "layer_0_weight", layer0Bias: "layer_0_bias", layer1Weight: "layer_1_weight", layer1Bias: "layer_1_bias" },
        calibration: { kind: "platt", coefficient: "platt_coefficient", intercept: "platt_intercept" },
      },
      speaker_count: {
        kind: "pooled-linear", type: "choice", labels: ["0", "1", "2", "3+"], tap: "h12", pooling: { kind: "mean-std" },
        weights: await P("diarization/speaker_count.safetensors"),
        tensors: { coef: "coef", intercept: "intercept", mean: "scaler_mean", scale: "scaler_scale" },
        calibration: { kind: "temperature", tensor: "temperature" },
      },
    },
    notes: "voice_activity answers ONE canonical Portuguese question (its encoding is pinned in question). Speaker tasks need a full 8 s window; turn_end reads the last 2 s of real audio. Language and spoken-word questions of Marimba need the Qwen3-ASR decoder and are not part of this package.",
    fidelity: {
      reference: "marimba-audio 0.1.2 (torch float32 CPU), 56 FLEURS pt-BR clips, windows right-aligned to 8 s (set A) and 3 clips concatenated (set B)",
      chrome157WebGPU: { n: 56, maxAbsDiffP: { voice_activity: 2.04e-5, turn_end: 3.82e-5, same_speaker: 3.17e-5, speaker_change: 8.2e-6, speaker_count: 3.35e-5 }, decisionAgreement: "56/56 (vad at 0.225, turn_end at 0.5)" },
      source: "test/browser/media-chrome.mjs audio",
    },
  };
  // Decoder (language, term): present when <weights>/marimba-audio/decoder/ holds the `s1s2-convert asr-decoder` output.
  const dd = path.join(d, "decoder");
  if (fs.existsSync(path.join(dd, "model.onnx"))) {
    const pr = JSON.parse(fs.readFileSync(path.join(dd, "decoder_prompt.json"), "utf8"));
    const rec = JSON.parse(fs.readFileSync(path.join(dd, "decoder.json"), "utf8")), tc = rec.text_config;
    audio.decoder = {
      runtime: "onnx-genai", graph: await P("decoder/model.onnx"), data: await P("decoder/model.onnx.data"),
      embeddings: await P("decoder/embed_tokens.fp16.bin"), tokens: await P("decoder/decoder_tokens.json"),
      model: { layers: tc.num_hidden_layers, kvHeads: tc.num_key_value_heads, headDim: tc.head_dim, hidden: tc.hidden_size, vocab: tc.vocab_size },
      prompt: { before: pr.before, after: pr.after, audioFrames: rec.audioFrames, languageToken: pr.languageToken },
      eos: pr.eos, languages: pr.languages,
      term: rec.term,
      devices: ["webgpu"],
    };
    // The same decoder built for CPU kernels (`s1s2-convert asr-decoder --ep cpu`): Node CPU and browser WASM.
    if (fs.existsSync(path.join(dd, "cpu", "model.onnx"))) {
      audio.decoder.cpuBuild = { graph: await P("decoder/cpu/model.onnx"), data: await P("decoder/cpu/model.onnx.data") };
      audio.decoder.devices = ["webgpu", "wasm", "cpu"];
    }
    audio.fidelity = {
      ...audio.fidelity,
      decoder: {
        reference: "marimba-audio 0.1.2 (torch float32 CPU); 72 windows with the audio at index 0: 56 FLEURS pt-BR clips, 3 demo clips, 12 joined 8 s windows; 952 (window, term) pairs",
        nodeCpuInt4: { build: "cpuBuild", languageArgmax: "72/72", maxAbsDiffTopProbability: 0.090, speechOnlyMaxAbsDiffTopProbability: 0.0201, termFractionIdentical: 0.943, termFractionWithin02: 0.981, termDecisionEqual: 0.986, greedyTextEqual: 0.51, beamTextEqual: 0.26 },
        chromeWebGPUInt4: { build: "graph", languageArgmax: "72/72", maxAbsDiffTopProbability: 0.035, speechOnlyMaxAbsDiffTopProbability: 0.015, termFractionIdentical: 0.943, termFractionWithin02: 0.982, termDecisionEqual: 0.985, greedyTextEqual: 0.47, beamTextEqual: 0.25 },
        nodeCpuFloat32: { windows: 12, languageArgmax: "12/12", maxAbsDiffProbability: 7.5e-8, termFractionIdentical: 1, greedyTextEqual: 1, beamTextEqual: 1 },
        note: "The tolerance of spec section 9.4 for the probability of the most probable language (0.02) is not met: on the synthetic noise clip (largest difference 0.090 on Node CPU, 0.035 on WebGPU) and by 0.0001 on one speech window of the Node CPU run with torch features (0.0201). Float32 and torch agree to 1e-7, so the differences are int4. Measured on Portuguese audio only."
      },
    };
    audio.readers.language = { kind: "decoder-language", type: "choice", tap: "audio_features" };
    audio.readers.term = { kind: "decoder-term", type: "predicate", tap: "audio_features" };
    audio.notes = audio.notes.replace(" Language and spoken-word questions of Marimba need the Qwen3-ASR decoder and are not part of this package.", " language and term are answered by the Qwen3-ASR decoder over the tower's final vectors (spec section 9): language is the closed-set score over 30 languages, term the fraction of five hypotheses (greedy + beam 4) that contain the word, uncalibrated.");
    audio.notes = audio.notes.replace(/ language and term are/, " Language and term are");
  }
  const man = {
    schema: "s1s2.manifest/1", id: "marimba-qwen3-asr-0.6b-web", modalities: ["audio"], repo: HUB_REPO, revision: HUB_REVISION,
    license: "Apache-2.0",
    provenance: {
      generator: { name: "recipes/marimba-audio", version: "0.1.0" },
      base: { repo: "Qwen/Qwen3-ASR-0.6B", revision: "5eb144179a02acc5e5ba31e748d22b0cf3e303b0", license: "Apache-2.0" },
      source: { format: "marimba-readouts", repo: "charcot-ai/marimba", revision: "ad374b4e61202c8425d1b3b2766bef985b01df45", files: ["turn_end/readout.safetensors", "diarization/same_speaker.safetensors", "diarization/speaker_change.safetensors", "diarization/speaker_count.safetensors", "speech_head/model.safetensors"] },
      notes: "Published package. audio_tower.onnx and vad_head.onnx are ONNX exports of the frozen Qwen3-ASR-0.6B audio encoder and of the Marimba speech head (recipes/marimba-audio/export_*.py in eduardoworrel/s1s2.js); the readout safetensors are the Marimba files unchanged (their sha256 are in Marimba's config.json); decoder/ is the Qwen3-ASR-0.6B text decoder converted by `s1s2-convert asr-decoder` (renamed tensors, tied lm_head, int4, float16 embedding table), WebGPU build and CPU build. Licenses as the cards state them: Apache-2.0 for the encoder, the decoder and the head/readout weights; the turn-end readout was trained on pipecat-ai/smart-turn-data-v3.2-train (CC BY 4.0). `revision` is the commit of this repository that holds the weight files. This is not an official package of the Qwen team or of Charcot.",
    },
    audio,
  };
  fs.writeFileSync(path.join(ROOT, "manifests/marimba-audio.json"), JSON.stringify(man, null, 2) + "\n");
}

console.log("wrote manifests/marimba-audio.json");
