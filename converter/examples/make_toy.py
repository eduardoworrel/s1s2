"""Toy dataset for the converter's end-to-end proof (deterministic; not a benchmark).

Each row: a 4-sentence customer-service passage, the requirement "a refund", label Yes/No (does the passage address a
refund?), and for Yes rows the index of the sentence that shows it. One requirement keeps the task learnable by a 135M
model in a few CPU minutes (the proof is about the pipeline, not about the model). Written to examples/toy.jsonl.
"""
import json
import random
from pathlib import Path

REQ = {
    "a refund": ["We will refund the full amount within five days.", "You can ask for your money back at any time.",
                 "Refunds go to the card you paid with.", "We sent your refund this morning.",
                 "Returned items are refunded in full.", "Your money will be paid back next week."],
    "a delivery date": ["Your parcel arrives on Monday.", "Shipping takes three business days.",
                        "The courier will deliver it next Friday."],
    "a phone number": ["Call us at 555-0134.", "Our hotline is 555-0199, open all day.", "You can phone 555-0172 for help."],
    "a price": ["The plan costs twelve dollars a month.", "This model sells for 40 euros.", "The fee is five dollars per order."],
    "a password reset": ["Click the link to reset your password.", "You can choose a new password in settings.",
                         "We sent you a code to change your password."],
}
FILLER = ["Thank you for contacting our team.", "We hope you enjoy the product.", "Our office is closed on holidays.",
          "Please keep this message for your records.", "We value your feedback.", "The weather has been lovely lately.",
          "Our store has a new logo.", "Have a great day."]


def main(n=400, seed=7, out=Path(__file__).with_name("toy.jsonl")):
    rng = random.Random(seed)
    rows = []
    reqs = list(REQ)
    for i in range(n):
        req = "a refund"
        yes = i % 2 == 0
        sents = rng.sample(FILLER, 3)
        if yes:
            k = rng.randrange(4)
            sents.insert(k, rng.choice(REQ[req]))
        else:
            other = rng.choice([r for r in reqs if r != req])
            sents.insert(rng.randrange(4), rng.choice(REQ[other]))
        row = {"id": f"toy-{i:03d}", "text": " ".join(sents), "requirement": req, "label": "Yes" if yes else "No"}
        if yes:
            row["evidence"] = k
            row["sentences"] = sents
        rows.append(row)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
