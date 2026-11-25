# Examples

| Demo | What it does |
|---|---|
| [`policy-checker/`](policy-checker) | Privacy Policy Checker (GDPR / LGPD / CCPA) with the published S5-nano package; own `serve.mjs`. |
| [`audio-turn/`](audio-turn) | **Experimental.** Voice activity, end of turn, same speaker, speaker change and speaker count from an audio file, the microphone (on click; **not yet verified**: see [`spec/MULTIMODAL.md`](../spec/MULTIMODAL.md)) or example clips, each with probability and margin. Frozen Qwen3-ASR-0.6B encoder + Marimba readouts; one pass for all five. With the decoder files: the language of the audio and a typed word (yes / no, from five transcripts). |

The media demo runs 100 % in the browser and need **local weights**: they are not published and the pages never fetch them from
anywhere else. Produce them as described in [`../recipes/marimba-audio`](../recipes/marimba-audio/README.md), put them in one directory (`marimba-audio/`), then:

```sh
npm ci && npm run build
node examples/serve-media.mjs --weights <dir>        # http://127.0.0.1:8788/audio-turn/
```

The server binds to 127.0.0.1, checks the sha256 of every pinned file against `manifests/*.json` before it starts (the page cannot
hash gigabytes), and serves them under `/weights/<package>/`. Where a page looks for them is configurable: `?weights=<url>` or the
`data-weights` attribute of `<html>` (default `../weights/`), and `?manifest=` / `data-manifest` for the manifest. `?lang=en|pt` sets the
language. Needs a recent Chrome or Edge with WebGPU (the audio demo falls back to WASM). Opening the
HTML from `file://` cannot load the models (the browser blocks the fetches); use the server.

Sizes: audio 746 MB (encoder) + 10 MB; language and word add the decoder (0.66 GB on WebGPU: 0.35 GB graph + 0.31 GB embeddings; the CPU build adds 0.41 GB). The page has a checkbox to leave it out.
Example clips of the audio demo: FLEURS (CC BY 4.0), attribution in [`audio-turn/clips/ATTRIBUTION.md`](audio-turn/clips/ATTRIBUTION.md).

Design, tolerances and limits: [`../spec/MULTIMODAL.md`](../spec/MULTIMODAL.md). Measurements, open points and licences: the same file, sections 5 to 8.

---

# Exemplos

| Demo | O que faz |
|---|---|
| [`policy-checker/`](policy-checker) | Verificador de política de privacidade (GDPR / LGPD / CCPA) com o pacote S5-nano publicado; `serve.mjs` próprio. |
| [`audio-turn/`](audio-turn) | **Experimental.** Voz ativa, fim de turno, mesmo locutor, troca de locutor e contagem de locutores a partir de um arquivo, do microfone (ao clicar; **ainda não verificado**, ver [`spec/MULTIMODAL.md`](../spec/MULTIMODAL.md)) ou de clipes de exemplo, com probabilidade e margem. Codificador Qwen3-ASR-0.6B congelado + leituras do Marimba; uma passada para as cinco. Com os arquivos do decodificador: o idioma do áudio e uma palavra digitada (sim / não, a partir de cinco transcrições). |

A demo de mídia roda 100 % no navegador e precisam de **pesos locais**: não estão publicados e as páginas nunca os buscam em
outro lugar. Gere-os como em [`../recipes/marimba-audio`](../recipes/marimba-audio/README.md), coloque em uma pasta (`marimba-audio/`) e rode
`node examples/serve-media.mjs --weights <pasta>`. O servidor confere o sha256 de cada arquivo contra `manifests/*.json` antes de subir.
O caminho é configurável: `?weights=<url>` ou o atributo `data-weights` do `<html>`; `?lang=en|pt` troca o idioma. Abrir o HTML por
`file://` não carrega os modelos; use o servidor.
