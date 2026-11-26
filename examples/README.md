# Examples

Single-page demos that run on the device. The three text demos use the published
[S5-nano ContractNLI + CLINC150 + SIB-200](https://huggingface.co/eduardoworrel/S5-nano-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2)
package (int4, 2.4 GB, WebGPU; CC BY-SA 4.0): each shows the decision, its margin and a "see what the model chose" button, and abstains when the margin is low.

| Demo | What it does |
|---|---|
| [`roteador/`](roteador) | Topic router: one pass picks among 7 topics (SIB-200, English); candidates, margin, and its cost against generating JSON. |
| [`contrato/`](contrato) | Contract check: paste an NDA, get yes / no / not mentioned for each claim, with margin and a checked passage. |
| [`sentinela/`](sentinela) | Message sentinel (**experimental**): S1 wakes S2 to summarise only the urgent messages. |
| [`policy-checker/`](policy-checker) | Privacy Policy Checker (GDPR / LGPD / CCPA) with the published S5-nano GDPR + LGPD package. |
| [`audio-turn/`](audio-turn) | **Experimental.** Voice activity, end of turn, same speaker, speaker change and speaker count from an audio file, the microphone (on click; **not yet verified**: see [`spec/MULTIMODAL.md`](../spec/MULTIMODAL.md)) or example clips, each with probability and margin. Frozen Qwen3-ASR-0.6B encoder + Marimba readouts; one pass for all five. With the decoder files: the language of the audio and a typed word (yes / no, from five transcripts). |

The text demos are also published at <https://webai.blog.br/demos/s1s2/>: open them there without installing anything. To run them from this repository, nothing needs to be built or installed:

```sh
node examples/serve.mjs            # http://127.0.0.1:8787/roteador/  (or /contrato/, /sentinela/, /policy-checker/)
```

`serve.mjs` only serves the page files of this folder on 127.0.0.1. Everything else loads from the internet at a pinned version, listed in each page's import map: `@s1s2/web` 0.2.1, `onnxruntime-web` 1.31.1 (also its `.wasm` files) and `@huggingface/transformers` 4.3.1 from cdn.jsdelivr.net, and the models from Hugging Face at a pinned 40-character revision. You can also open the HTML from any static host. The pages run ONNX Runtime with one thread, so they need no `Cross-Origin-Embedder-Policy` headers (which would block the CDN and the Hub).
Needs a recent Chrome or Edge with WebGPU and a connection. The weights (2.4 GB per package; the router, contract and sentinel share one, the policy checker uses another) download once and the browser keeps them; nothing else leaves the page. `?lang=en|pt` sets the page language.

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
| [`policy-checker/`](policy-checker) | Verificador de política de privacidade (GDPR / LGPD / CCPA) com o pacote S5-nano GDPR + LGPD publicado. |
| [`audio-turn/`](audio-turn) | **Experimental.** Voz ativa, fim de turno, mesmo locutor, troca de locutor e contagem de locutores a partir de um arquivo, do microfone (ao clicar; **ainda não verificado**, ver [`spec/MULTIMODAL.md`](../spec/MULTIMODAL.md)) ou de clipes de exemplo, com probabilidade e margem. Codificador Qwen3-ASR-0.6B congelado + leituras do Marimba; uma passada para as cinco. Com os arquivos do decodificador: o idioma do áudio e uma palavra digitada (sim / não, a partir de cinco transcrições). |

As demos de texto também estão publicadas em <https://webai.blog.br/demos/s1s2/>: abra lá sem instalar nada. Para rodá-las a partir deste repositório, não é preciso build nem instalação:

```sh
node examples/serve.mjs            # http://127.0.0.1:8787/roteador/  (ou /contrato/, /sentinela/, /policy-checker/)
```

O `serve.mjs` só serve os arquivos das páginas desta pasta em 127.0.0.1. Todo o resto vem da internet em versão fixada, listada no import map de cada página: `@s1s2/web` 0.2.1, `onnxruntime-web` 1.31.1 (e seus `.wasm`) e `@huggingface/transformers` 4.3.1 pelo cdn.jsdelivr.net, e os modelos pelo Hugging Face em revisão fixa de 40 caracteres. As páginas rodam o ONNX Runtime com uma thread, então não precisam dos cabeçalhos `Cross-Origin-Embedder-Policy` (que bloqueariam o CDN e o Hub).
Precisam de Chrome ou Edge recentes com WebGPU e de conexão. Os pesos (2,4 GB por pacote; roteador, contrato e sentinela dividem um, o verificador de política usa outro) baixam uma vez e o navegador os guarda; nada mais sai da página. `?lang=en|pt` troca o idioma da página.

A demo de mídia roda 100 % no navegador e precisa dos pesos em uma **pasta local**: as páginas nunca os buscam em outro lugar. O pacote está no Hub
([eduardoworrel/Marimba-ONNX-S1S2](https://huggingface.co/eduardoworrel/Marimba-ONNX-S1S2)); coloque os arquivos em uma pasta (`marimba-audio/`) ou gere-os como em [`../recipes/marimba-audio`](../recipes/marimba-audio/README.md) e rode
`node examples/serve-media.mjs --weights <pasta>`. O servidor confere o sha256 de cada arquivo contra `manifests/*.json` antes de subir.
O caminho é configurável: `?weights=<url>` ou o atributo `data-weights` do `<html>`; `?lang=en|pt` troca o idioma. Abrir o HTML por
`file://` não carrega os modelos; use o servidor.
