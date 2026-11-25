# Message sentinel / Sentinela de mensagens (experimental)

**EN.** A stream of messages (here, WhatsApp messages to a criminal-defence lawyer). S1 reads each one in a single pass (yes / no: arrest in progress, or a hearing or deadline within 48 hours) and reports P(urgent) and the margin. Only when S1 wakes it, or is unsure, does S2 write a two-line summary on the same weights. Below the margin you set, the message is marked unsure and goes to a person.

Open: `node examples/serve.mjs`, then `http://127.0.0.1:8787/sentinela/` (`?lang=en`).

**Experimental, and this is the limit found in the lab.** On 300 synthetic messages (36 urgent) this model's S1, at P(urgent) 0.20, raised no false alarm and missed 3 of the 36 urgent ones. The S1 + S2 pair was **not proven** better than S1 plus a summary rule: on this model it tied (the interval crossed zero), and it won on only one model of the S5 family, the privacy S5-nano, which is not the one used here. The 0.20 threshold comes from an earlier test and was not recalibrated. Other model families did not hold up with this prompt. Do not use it where a missed urgent message has a cost.

Model: the published S5-nano ContractNLI + CLINC150 + SIB-200 package; `manifest.json` here pins the same files at the same revision and reads Sim / Não from the logits (no head). The example messages are synthetic, written for the demo.

**PT.** Um fluxo de mensagens (aqui, WhatsApp para um advogado criminalista). O S1 lê cada uma em uma passada (sim / não: prisão em curso, ou audiência ou prazo em até 48 horas) e informa P(urgente) e a margem. Só quando o S1 acorda, ou fica em dúvida, o S2 escreve um resumo de duas linhas nos mesmos pesos. Abaixo da margem escolhida, a mensagem fica como incerta e vai para uma pessoa.

Abrir: `node examples/serve.mjs` e `http://127.0.0.1:8787/sentinela/`.

**Experimental, e este é o limite achado no laboratório.** Em 300 mensagens sintéticas (36 urgentes), o S1 deste modelo, a P(urgente) 0,20, não deu falso alarme e omitiu 3 das 36 urgentes. O par S1 + S2 **não ficou provado** melhor que S1 mais uma regra de resumo: neste modelo empatou (o intervalo cruzou zero) e só venceu em um modelo da família S5, o S5-nano de privacidade, que não é o usado aqui. O limiar 0,20 vem de um teste anterior e não foi recalibrado. Outras famílias de modelo não se sustentaram com este prompt. Não use onde perder uma mensagem urgente tenha custo.

Modelo: o pacote publicado S5-nano ContractNLI + CLINC150 + SIB-200; o `manifest.json` daqui fixa os mesmos arquivos na mesma revisão e lê Sim / Não pelos logits (sem cabeça). As mensagens de exemplo são sintéticas, escritas para o demo.
