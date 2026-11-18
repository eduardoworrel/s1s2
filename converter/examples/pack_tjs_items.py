"""Parity prompts for pack_tjs, written for this check (no dataset rows, no test split).
Fills each source manifest's own `lab` templates (state/question/options|labels) with short hand-written texts.
usage: python pack_tjs_items.py <source dir> <out.jsonl> [per_manifest]"""
import json, re, sys
from pathlib import Path

NDA = [
    "1. Confidential Information means all non-public information disclosed by either party. 2. The Receiving Party shall not disclose Confidential Information to any third party. 3. Upon termination, the Receiving Party shall return or destroy all Confidential Information.",
    "The Recipient may share Confidential Information with its employees who need to know it. This Agreement is governed by the laws of the State of New York. Nothing in this Agreement grants any license to the Recipient.",
    "The parties agree that obligations of confidentiality survive for five years after termination. The Receiving Party may retain one archival copy for legal purposes.",
    "The Disclosing Party makes no warranty as to the accuracy of the information. The Recipient shall notify the Discloser promptly of any unauthorized disclosure.",
    "Information that is independently developed by the Receiving Party is not Confidential Information. The Receiving Party may disclose information when required by law, after notice.",
]
HYP = [
    "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
    "Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
    "Agreement shall not grant Receiving Party any right to Confidential Information.",
    "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.",
    "Receiving Party may independently develop information similar to Confidential Information.",
]
INTENTS = "banking: balance, transfer, freeze_account\ntravel: book_flight, book_hotel, travel_alert\nhome: shopping_list, reminder, alarm"
QUERIES = ["how much money is in my checking account", "set an alarm for 7 am", "what is the meaning of life",
           "book me a hotel in lisbon for friday", "who won the football match yesterday"]
SENT_EN = ["The new vaccine reduced hospital admissions by a third.", "The team scored twice in the final minutes to win the cup.",
           "Parliament approved the budget after a long debate.", "The river crosses three countries before reaching the sea.",
           "Engineers built a faster chip using less power."]
SENT_PT = ["A nova vacina reduziu as internações em um terço.", "O time marcou duas vezes nos minutos finais e ganhou a taça.",
           "O parlamento aprovou o orçamento depois de um longo debate.", "O rio atravessa três países antes de chegar ao mar.",
           "Engenheiros construíram um chip mais rápido que gasta menos energia."]
PASS_EN = ["We share your e-mail address with our advertising partners.", "You can ask us to delete your account at any time by writing to privacy@example.com.",
           "We keep order records for six years to meet tax rules.", "Our office is open Monday to Friday.",
           "Data may be transferred to servers in the United States under standard contractual clauses.",
           "Cookies help us remember your language.", "You may complain to the data protection authority.",
           "We collect your name, address and payment details when you order.", "Children under 13 may not use the service.",
           "The controller is Example Ltd, 1 Main Street, Dublin."]
PASS_PT = ["Compartilhamos seu e-mail com parceiros de publicidade.", "Você pode pedir a exclusão da sua conta a qualquer momento pelo e-mail privacidade@exemplo.com.",
           "Guardamos os registros de pedidos por seis anos para cumprir regras fiscais.", "Nosso escritório abre de segunda a sexta.",
           "Os dados podem ser transferidos para servidores nos Estados Unidos.", "Os cookies ajudam a lembrar seu idioma.",
           "Você pode reclamar à Autoridade Nacional de Proteção de Dados.", "Coletamos seu nome, endereço e dados de pagamento no pedido.",
           "Menores de 13 anos não podem usar o serviço.", "O controlador é a Exemplo Ltda., Rua Principal, 1, São Paulo."]
REQ_EN = [("the recipients of the data", "whether and with which third parties, or categories of third parties, the personal data are shared"),
          ("the right to erasure", "how the data subject can have personal data deleted"),
          ("the retention period", "how long personal data are kept"),
          ("international transfers", "whether personal data are sent to other countries and under which safeguards"),
          ("the right to complain", "the right to lodge a complaint with a supervisory authority")]
REQ_PT = [("dos destinatários dos dados", "se e com quais terceiros, ou categorias de terceiros, os dados pessoais são compartilhados"),
          ("do direito de eliminação", "como o titular pode pedir a eliminação dos dados pessoais"),
          ("do prazo de retenção", "por quanto tempo os dados pessoais são guardados"),
          ("da transferência internacional", "se os dados pessoais são enviados a outros países e com quais garantias"),
          ("do direito de petição", "o direito de peticionar à autoridade nacional")]


def fill(t, **kw):
    return re.sub(r"\{(\w+)\}", lambda m: kw[m.group(1)], t)


def items_for(name, lab, n):
    out = []
    opts = lab.get("options") or [lab["labels"]["yes"], lab["labels"]["no"]]
    task = lab.get("task")
    for i in range(n):
        if task == "cnli":
            inp, q = fill(lab["state"], nda=NDA[i % 5]), fill(lab["question"], hipotese=HYP[(i * 2) % 5])
        elif task == "clinc":
            inp, q = fill(lab["state"], intencoes=INTENTS), fill(lab["question"], consulta=QUERIES[i % 5])
        elif task in ("sib_en", "sib_pt"):
            S = SENT_EN if task == "sib_en" else SENT_PT
            inp, q = fill(lab["state"], frase=S[i % 5]), lab["question"]
        else:  # GDPR / LGPD: passage as input, requirement in the question
            pt = lab.get("language") == "pt"
            P, R = (PASS_PT, REQ_PT) if pt else (PASS_EN, REQ_EN)
            nome, defin = R[i % 5]
            inp, q = P[i % 10], fill(lab["question"], nome_de=nome, definicao=defin)
        ch = opts if i % 2 == 0 else list(reversed(opts))  # both option orders
        out.append({"id": f"{name}#{i}", "manifest": name, "input": inp,
                    "question": {"name": "q", "type": "choice", "choices": ch, "instructions": q}})
    return out


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    mans = sorted(p for p in src.glob("manifest*.json") if p.name != "manifest.json") or [src / "manifest.json"]
    per = per or max(5, -(-20 // len(mans)))
    rows = []
    for p in mans:
        rows += items_for(p.name, json.loads(p.read_text())["lab"], per)
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(len(rows), "rows", [p.name for p in mans])
