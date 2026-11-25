# Examples

Single-page demos that run on the device. The three text demos use the published
[S5-nano ContractNLI + CLINC150 + SIB-200](https://huggingface.co/eduardoworrel/S5-nano-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2)
package (int4, 2.4 GB, WebGPU; CC BY-SA 4.0): each shows the decision, its margin and a "see what the model chose" button, and abstains when the margin is low.

| Demo | What it does |
|---|---|
| [`roteador/`](roteador) | Topic router: one pass picks among 7 topics (SIB-200, English); candidates, margin, and its cost against generating JSON. |
| [`contrato/`](contrato) | Contract check: paste an NDA, get yes / no / not mentioned for each claim, with margin and a checked passage. |
| [`sentinela/`](sentinela) | Message sentinel (**experimental**): S1 wakes S2 to summarise only the urgent messages. |
| [`policy-checker/`](policy-checker) | Privacy Policy Checker (GDPR / LGPD / CCPA) with the published S5-nano package; own `serve.mjs`. |
| [`audio-turn/`](audio-turn) | **Experimental.** Voice activity, end of turn, same speaker, speaker change and speaker count from an audio file, the microphone (on click; **not yet verified**: see [`spec/MULTIMODAL.md`](../spec/MULTIMODAL.md)) or example clips, each with probability and margin. Frozen Qwen3-ASR-0.6B encoder + Marimba readouts; one pass for all five. With the decoder files: the language of the audio and a typed word (yes / no, from five transcripts). |

The text demos (after `npm ci && npm run build` at the repository root):

```sh
node examples/serve.mjs            # http://127.0.0.1:8787/roteador/  (or /contrato/, /sentinela/)
```

Needs a recent Chrome or Edge with WebGPU. The weights (2.4 GB) download once from Hugging Face; nothing else leaves the page.
`?lang=en|pt` sets the page language. A local copy of the repository can be used with `node examples/serve.mjs --mirror eduardoworrel/S5-nano-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2=<dir>`; small files are still checked against their sha256 (the large weights are not hashed in the browser; see the manifest).
The `policy-checker` is started with its own `serve.mjs`.

The media demo runs 100 % in the browser and needs the weights in a **local directory**: the pages never fetch them from anywhere else. The package is on the Hub ([eduardoworrel/Marimba-ONNX-S1S2](https://huggingface.co/eduardoworrel/Marimba-ONNX-S1S2)); put its files in one directory (`marimba-audio/`) or produce them as described in [`../recipes/marimba-audio`](../recipes/marimba-audio/README.md), then:

```sh
npm ci && npm run build
node examples/serve-media.mjs --weights <dir>        # http://127.0.0.1:8788/audio-turn/
```

The server binds to 127.0.0.1, checks the sha256 of every pinned file against `manifests/*.json` before it starts (the page cannot
hash gigabytes), and serves them under `/weights/<package>/`. Where a page looks for them is configurable: `?weights=<url>` or the
`data-weights` attribute of `<html>` (default `../weights/`), and `?manifest=` / `data-manifest` for the manifest. `?lang=en|pt` sets the
language. Needs a recent Chrome or Edge with WebGPU (the audio demo falls back to WASM). Opening the
HTML from `file://` cannot load the models (the browser blocks the fetches); use the server.

Sizes: audio 746 MB (encoder) + 10 MB; language and word add the decoder (0.66 GB on WebGPU: 0.35 GB graph + 0.31 GB embeddings; the CPU build adds 0.41 GB). The page has a checkbox to leave it out; with it checked the page loads the decoder in `"auto"` mode, so if the decoder cannot load the page says so and the other five questions still work. Language and word were checked on Portuguese only, and the language probability is not within the 0.02 tolerance of the reference on noise (spec section 9.5).
Example clips of the audio demo: FLEURS (CC BY 4.0), attribution in [`audio-turn/clips/ATTRIBUTION.md`](audio-turn/clips/ATTRIBUTION.md).

Design, tolerances and limits: [`../spec/MULTIMODAL.md`](../spec/MULTIMODAL.md). Measurements, open points and licences: the same file, sections 5 to 8.

---

# Exemplos

Demos de uma página só que rodam no aparelho. As três demos de texto usam o pacote publicado
[S5-nano ContractNLI + CLINC150 + SIB-200](https://huggingface.co/eduardoworrel/S5-nano-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2)
(int4, 2,4 GB, WebGPU; CC BY-SA 4.0): cada uma mostra a decisão, a margem e o botão "ver o que o modelo escolheu", e se abstém quando a margem é baixa.

| Demo | O que faz |
|---|---|
| [`roteador/`](roteador) | Roteador de tópico: uma passada escolhe entre 7 tópicos (SIB-200, inglês); candidatos, margem e o custo contra gerar JSON. |
| [`contrato/`](contrato) | Contrato: cole um NDA e veja sim / não / não menciona para cada afirmação, com margem e trecho conferido. |
| [`sentinela/`](sentinela) | Sentinela de mensagens (**experimental**): o S1 acorda o S2 só para resumir as urgentes. |
| [`policy-checker/`](policy-checker) | Verificador de política de privacidade (GDPR / LGPD / CCPA) com o pacote S5-nano publicado; `serve.mjs` próprio. |
| [`audio-turn/`](audio-turn) | **Experimental.** Voz ativa, fim de turno, mesmo locutor, troca de locutor e contagem de locutores a partir de um arquivo, do microfone (ao clicar; **ainda não verificado**, ver [`spec/MULTIMODAL.md`](../spec/MULTIMODAL.md)) ou de clipes de exemplo, com probabilidade e margem. Codificador Qwen3-ASR-0.6B congelado + leituras do Marimba; uma passada para as cinco. Com os arquivos do decodificador: o idioma do áudio e uma palavra digitada (sim / não, a partir de cinco transcrições). |

Demos de texto (depois de `npm ci && npm run build` na raiz):

```sh
node examples/serve.mjs            # http://127.0.0.1:8787/roteador/  (ou /contrato/, /sentinela/)
```

Precisam de Chrome ou Edge recentes com WebGPU. Os pesos (2,4 GB) baixam uma vez do Hugging Face; nada mais sai da página.
`?lang=en|pt` troca o idioma da página.

A demo de mídia roda 100 % no navegador e precisam de **pesos locais**: não estão publicados e as páginas nunca os buscam em
outro lugar. Gere-os como em [`../recipes/marimba-audio`](../recipes/marimba-audio/README.md), coloque em uma pasta (`marimba-audio/`) e rode
`node examples/serve-media.mjs --weights <pasta>`. O servidor confere o sha256 de cada arquivo contra `manifests/*.json` antes de subir.
O caminho é configurável: `?weights=<url>` ou o atributo `data-weights` do `<html>`; `?lang=en|pt` troca o idioma. Abrir o HTML por
`file://` não carrega os modelos; use o servidor.
