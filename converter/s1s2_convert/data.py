"""Dataset reading and splitting.

Input: JSONL, one labelled example per line:
  {"text": str,                       passage / state the question is about (required)
   "question": str,                   full question text, or
   "requirement": str,                filled into --question-template ({requirement}); also fills the evidence question
   "label": str,                      one of the label set (required)
   "evidence": int | [int],           optional: index(es) of the sentence(s) that show the label (evidence head)
   "sentences": [str],                optional: the passage's sentences (default: s1s2 splitSentences(text))
   "split": "train" | "dev",          optional: fixed split (default: seeded random split, --dev-fraction)
   "id": str}                         optional
"""
import random
import re

from .util import sha256_file, read_jsonl

# s1s2 src/questions.ts splitSentences: break after . ! ? … (+ closing quotes/brackets) followed by whitespace.
_SENT = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)")


def split_sentences(text):
    out, start = [], 0
    for m in _SENT.finditer(text):
        out.append(text[start:m.end()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def one_line(s):
    """s1s2 decide.ts oneLine: whitespace collapsed."""
    return " ".join(s.split())


def load(path, question_template=None, labels=None, dev_fraction=0.2, seed=1):
    raw = read_jsonl(path)
    if not raw:
        raise SystemExit(f"{path}: empty dataset")
    rows, seen_labels = [], []
    for i, r in enumerate(raw):
        at = f"{path}:{i + 1}"
        if not isinstance(r.get("text"), str) or not r["text"].strip():
            raise SystemExit(f"{at}: text must be a non-empty string")
        if not isinstance(r.get("label"), str) or not r["label"].strip():
            raise SystemExit(f"{at}: label must be a non-empty string")
        q, req = r.get("question"), r.get("requirement")
        if q is None and req is None:
            raise SystemExit(f"{at}: question or requirement required")
        if q is None:
            if not question_template or "{requirement}" not in question_template:
                raise SystemExit(f"{at}: requirement given, so --question-template with {{requirement}} is required")
            q = question_template.replace("{requirement}", req)
        if r["label"] not in seen_labels:
            seen_labels.append(r["label"])
        ev = r.get("evidence")
        sents = r.get("sentences")
        if ev is not None:
            ev = [ev] if isinstance(ev, int) else ev
            if not (isinstance(ev, list) and all(isinstance(x, int) for x in ev)):
                raise SystemExit(f"{at}: evidence must be an int or a list of ints")
            if sents is None:
                sents = split_sentences(r["text"])
            sents = [one_line(s) for s in sents]
            if len(sents) < 2 or not all(sents):
                raise SystemExit(f"{at}: evidence needs at least 2 non-empty sentences")
            if any(x < 0 or x >= len(sents) for x in ev):
                raise SystemExit(f"{at}: evidence index outside the {len(sents)} sentences")
        if r.get("split") not in (None, "train", "dev"):
            raise SystemExit(f"{at}: split must be train or dev")
        rows.append({"id": str(r.get("id", i)), "text": r["text"], "question": q, "requirement": req, "label": r["label"],
                     "evidence": ev, "sentences": sents if ev is not None else None, "split": r.get("split")})
    label_set = labels or seen_labels
    for r in rows:
        if r["label"] not in label_set:
            raise SystemExit(f"row {r['id']}: label {r['label']!r} not in the label set {label_set}")
    if len(label_set) < 2:
        raise SystemExit(f"need at least 2 labels, got {label_set}")
    if all(r["split"] is None for r in rows):
        idx = list(range(len(rows)))
        random.Random(seed).shuffle(idx)
        n_dev = max(1, round(dev_fraction * len(rows)))
        dev = set(idx[:n_dev])
        for i, r in enumerate(rows):
            r["split"] = "dev" if i in dev else "train"
    elif any(r["split"] is None for r in rows):
        raise SystemExit("split: give it on every row or on none")
    splits = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev")}
    if not splits["train"] or not splits["dev"]:
        raise SystemExit(f"need train and dev rows, got {splits}")
    meta = {"sha256": sha256_file(path), "rows": len(rows), "splits": splits, "labels": label_set,
            "evidence_rows": sum(r["evidence"] is not None for r in rows)}
    return rows, meta
