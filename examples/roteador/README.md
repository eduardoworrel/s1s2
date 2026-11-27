# Topic router / Roteador de tópico

**EN.** One forward pass picks a topic (science/technology, travel, politics, sports, health, entertainment, geography) for each English sentence, with the margin of the answer. Below the margin you set, the page abstains. The "Compare" button times the decision against generating the answer as JSON on the same weights, on your device.

Open: `node examples/serve.mjs`, then `http://127.0.0.1:8787/roteador/` (`?lang=en`).

Model: the published S5-nano ContractNLI + CLINC150 + SIB-200 package, `manifest.sib_en.json` at a pinned revision. The CLINC150 head in that package decides in scope / out of scope; it does not choose among domains, so it is not used here. The SIB heads have no browser fidelity record. The abstention threshold is the demo's own and was not calibrated.

**PT.** Uma passada escolhe o tópico de cada frase em inglês, com a margem. Abaixo da margem escolhida, a página se abstém. O botão "Comparar" mede, no seu aparelho, a decisão contra gerar a resposta em JSON nos mesmos pesos.

Abrir: `node examples/serve.mjs` e `http://127.0.0.1:8787/roteador/`.

Modelo: o pacote publicado S5-nano ContractNLI + CLINC150 + SIB-200, `manifest.sib_en.json` em revisão fixa. A cabeça CLINC150 desse pacote decide dentro / fora de escopo; não escolhe entre domínios, então não é usada aqui. As cabeças SIB não têm registro de fidelidade no navegador. O limiar de abstenção é só do demo e não foi calibrado.
