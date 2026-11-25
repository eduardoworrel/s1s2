# Contract check / Contrato

**EN.** Paste an NDA (English, up to 2,048 tokens). The page asks eight claims at once (third parties, employees, return or destruction, kept copy, reverse engineering, compelled disclosure, spoken information, survival) and shows yes / no / not mentioned with the margin; below the margin you set it abstains. "Show the passage" asks the model to quote the supporting sentence and checks the quote against your text; a quote that is not in the text is marked and should not be trusted. You can add a claim of your own in English.

Open: `node examples/serve.mjs`, then `http://127.0.0.1:8787/contrato/` (`?lang=en`).

Model: the published S5-nano ContractNLI + CLINC150 + SIB-200 package, `manifest.cnli.json` at a pinned revision. The head reads English contracts and English claims, so the claims are sent in English whatever the page language. The card reports 78.1% balanced accuracy on the ContractNLI test, measured outside the browser. The evidence head in the package is not driven by this library yet, so the passage comes from generation. Sample contract written for the demo. Not legal advice.

**PT.** Cole um NDA (inglês, até 2.048 tokens). A página faz oito afirmações de uma vez (terceiros, funcionários, devolução ou destruição, cópia guardada, engenharia reversa, revelação por ordem legal, informação falada, sobrevivência) e mostra sim / não / não menciona com a margem; abaixo da margem escolhida, abstém. "Mostrar o trecho" pede ao modelo a frase que sustenta a resposta e confere a citação contra o seu texto; citação que não está no texto é marcada e não deve ser usada. Dá para acrescentar uma afirmação sua, em inglês.

Abrir: `node examples/serve.mjs` e `http://127.0.0.1:8787/contrato/`.

Modelo: o pacote publicado S5-nano ContractNLI + CLINC150 + SIB-200, `manifest.cnli.json` em revisão fixa. A cabeça lê contratos e afirmações em inglês, então as afirmações vão em inglês qualquer que seja o idioma da página. O cartão informa 78,1% de acerto balanceado no teste do ContractNLI, medido fora do navegador. A cabeça de evidência do pacote ainda não é usada por esta biblioteca, então o trecho vem de geração. Contrato de exemplo escrito para o demo. Não é parecer jurídico.
