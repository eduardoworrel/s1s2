# Marimba audio package (Qwen3-ASR-0.6B encoder + readouts)

How `manifests/marimba-audio.json` is produced. Nothing here is downloaded by the library; the files below were copied from the reference run (then renamed and their comments translated; hashes of the committed files in `../SOURCES.sha256`) and are the record of how the ONNX files and the references were made.

Sources (all read from the pinned revisions in the manifest):

| piece | source | what was done |
|---|---|---|
| audio encoder | `Qwen/Qwen3-ASR-0.6B` @ `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` (Apache-2.0 on its card) | `export_tower.py`: fixed 8 s window (800 mel frames), outputs h9, h12, h18 and the final projection, float32 ONNX (745.7 MB) |
| speech head | `charcot-ai/marimba` @ `ad374b4e61202c8425d1b3b2766bef985b01df45`, `speech_head/model.safetensors` | `export_head.py`: ONNX graph (9.7 MB) plus the question vector of the canonical prompt (1024 bytes, `vad_question.f32`) |
| turn end, same speaker, speaker change, speaker count | the same Marimba repo, `turn_end/readout.safetensors`, `diarization/*.safetensors` | used unchanged (read in JavaScript); their sha256 equal those in Marimba's `config.json` |
| references | `marimba-audio` 0.1.2 on torch float32 CPU | `refs.py` writes `ref.json` and the windows / mel features of 56 FLEURS pt-BR clips; `e2e_py.py` is the Python end-to-end check |

Layout of the weights directory the demos and tests read (`<weights>/marimba-audio/`):

```
audio_tower.onnx   vad_head.onnx   vad_question.f32
turn_end/readout.safetensors
diarization/same_speaker.safetensors   diarization/speaker_change.safetensors   diarization/speaker_count.safetensors
```

`vad_question.f32` is `base64-decode(weights.json.vad.qpool.b64)` of the `export_head.py` output (256 float32, little-endian).
`node recipes/make-manifests.mjs <weights dir>` rewrites the manifest with the sha256 and size of each file as found.

Files here: `browser-index.html`, `browser-readers.js` and `browser-mel.js` are the reference run's browser page, readers and mel pre-processor, and `browser-run.mjs`
its Chrome runner (the library re-implements them in `src/media/`); `node-parity.mjs` reproduces the measurements through the library on Node CPU;
`../../test/browser/media-chrome.mjs audio` does it in Chrome. These scripts are a record, not a build; paths come from `AUDIO_REF_DIR` and `CHROME`.

Layout of the reference directory (`<reference dir>`; `S1S2_AUDIO_REF` for the opt-in test, the first argument of `node-parity.mjs`):

```
port/audio_tower.onnx   port/vad_head.onnx            exported by export_tower.py / export_head.py
port/ref/ref.json       port/ref/{winA,winB,melA,melB}.f32   written by refs.py
result.json                                           written by run.py (torch decisions per clip)
web/weights.json        web/mel_filters.json          written by export_head.py (browser page only)
```
Licences: the cards state Apache-2.0 for the encoder and for the head and readout weights; the training data of the turn-end readout is CC BY 4.0. See `spec/MULTIMODAL.md`.
