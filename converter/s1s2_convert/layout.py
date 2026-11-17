"""Prompt layout of the package (spec section 3.1, plain-text markers).

The manifest strings written here ARE the training prompts: the training rows are encoded by s1s2 itself
(`s1s2 encode`, the runtime's tokenizer and renderer) from this manifest, so what the heads are trained on is what the
runtime sends. The Python port below (fill / escape / render, ported from s1s2 src/encoding.ts) is used only to
discover the marker token ids and the label tokens before a manifest exists, and as a second opinion: the encode stage
compares its ids with s1s2's and stops on any difference.

Layout (the plain-text marker layout of the lab-s12 studies):
  user turn  = "<state> {state}\\n <question> {question} <decide>"
  question   = "{instructions}\\n{options}"
  option     = " <option>{label} </option> \\n"     label = " " + dataset label
  template   = chat_template(system?, user turn, add_generation_prompt) + cue
"""
import json
import re

MARK = {"state": "<state>", "question": " <question>", "option": " <option>", "option_end": " </option>", "decide": " <decide>"}
APOS = " "
ORDER = ["state", "question", "option", "option_end", "decide"]
USER = MARK["state"] + APOS + "{state}\n" + MARK["question"] + APOS + "{question}" + MARK["decide"]
OPTION = MARK["option"] + "{label}" + MARK["option_end"] + APOS + "\n"
Q_CHOICE = "{instructions}\n{options}"
SENT_U, SENT_P, SENT_A = "\u0001USER\u0001", "\u0001PROMPT\u0001", "\u0001ANS\u0001"
PROBE_STATE, PROBE_Q, PROBE_OPTS = "Probe state.", "Probe question?", ["first", "second"]  # s1s2 assertManifestTokens


def fill(t, vals):
    """s1s2 encoding.fill: {key} replaced only for keys in vals, first to last, no recursion. -> (text, spans)."""
    out, spans, last = "", {}, 0
    for m in re.finditer(r"\{([a-z]+)\}", t):
        k = m.group(1)
        if k not in vals:
            continue
        out += t[last:m.start()]
        s = len(out)
        out += vals[k]
        spans.setdefault(k, (s, len(out)))
        last = m.end()
    return out + t[last:], spans


def escape_special(s):
    return re.sub(r"<\|([A-Za-z0-9_]+)\|>", "<¦\\1¦>", s)


def escape_literals(s, lits):
    for raw in lits:
        m = raw.strip()
        if m and m in s:
            s = s.replace(m, "‹" + m[1:])
    return s


def escaper(enc):
    lits = enc.get("escapeLiterals") or []
    return lambda s: escape_literals(escape_special(s) if enc.get("escapeSpecial") else s, lits)


def render(enc, markers, qtype, instructions, texts, state):
    """s1s2 renderTagged. -> (text, decide_char, option_chars)."""
    esc = escaper(enc)
    options, rel = "", []
    if qtype == "choice":
        for i, t in enumerate(texts):
            if i:
                options += enc.get("optionSeparator", "")
            o, sp = fill(enc["option"], {"label": esc(t), "key": chr(65 + i), "index": str(i + 1)})
            after = o[sp["label"][1]:]
            k = after.index(markers["option"]["text"])
            rel.append(len(options) + sp["label"][1] + k + len(markers["option"]["text"]) - 1)
            options += o
    qt, qsp = fill(enc["question"][qtype], {"instructions": esc(instructions), "options": options})
    at = qsp.get("options", (0, 0))[0]
    full, sp = fill(enc["template"], {"state": esc(state), "question": qt})
    qs, qe = sp["question"]
    k = full[qe:].index(markers["decide"]["text"])
    return full, qe + k + len(markers["decide"]["text"]) - 1, [qs + at + r for r in rel]


def span_last_token(encode, text, total, a, b, what):
    """s1s2 spanLastToken: position by prefix encoding (stable boundary required)."""
    pa, pb = encode(text[:a]), encode(text[:b])
    if pb[:len(pa)] != pa or total[:len(pb)] != pb or len(pb) == len(pa):
        raise SystemExit(f"layout: the {what} marker has no stable token boundary in this prompt")
    return len(pb) - 1


def positions(encode, enc, markers, qtype, instructions, texts, state):
    text, dc, ocs = render(enc, markers, qtype, instructions, texts, state)
    ids = encode(text)
    n = lambda c, mk, w: span_last_token(encode, text, ids, c - len(markers[mk]["text"]) + 1, c + 1, w)  # noqa: E731
    return {"text": text, "ids": ids, "decide": n(dc, "decide", "decide"), "options": [n(c, "option", f"option {i}") for i, c in enumerate(ocs)]}


def chat(tok, system, chat_kwargs, user, cue=""):
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": user}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs)
    if t.count(user) != 1:
        raise SystemExit("chat template: the user text does not appear exactly once in the rendered prompt")
    return t + cue


def build_encoding(tok, system=None, chat_kwargs=None, cue="Answer:"):
    """encoding + generation blocks of the manifest for this tokenizer's chat template."""
    ck = chat_kwargs or {}
    chat_u = chat(tok, system, ck, SENT_U, cue)
    for brace in "{}":
        if brace in chat_u.replace(SENT_U, ""):
            raise SystemExit(f"chat template text contains {brace!r}; the tagged fill would misread it")
    enc = {"kind": "tagged", "template": chat_u.replace(SENT_U, USER),
           "question": {"predicate": "{instructions}", "choice": Q_CHOICE, "score": "{instructions}"},
           "option": OPTION, "optionSeparator": "",
           "forbiddenInLabels": sorted({MARK["option"].strip(), MARK["option_end"].strip()}),
           "escapeSpecial": True, "escapeLiterals": [MARK[k] for k in ORDER]}
    turn = chat(tok, system, ck, SENT_P).replace(SENT_P, "{prompt}")
    msgs = ([{"role": "system", "content": system}] if system is not None else [])
    two = tok.apply_chat_template(msgs + [{"role": "user", "content": SENT_U}, {"role": "assistant", "content": cue + SENT_A},
                                          {"role": "user", "content": SENT_P}], tokenize=False, add_generation_prompt=True, **ck)
    if not two.startswith(chat_u):
        raise SystemExit("chat template: decision prompt + answer + new turn does not extend the decision prompt")
    cont = two[len(chat_u):].replace(SENT_A, "{answer}", 1).replace(SENT_P, "{prompt}", 1)
    return enc, {"turn": turn, "continue": cont}


def markers_skeleton():
    return {"decide": {"text": MARK["decide"], "at": "last-token", "tokens": []},
            "option": {"text": MARK["option_end"], "at": "last-token", "tokens": []}}


def option_text(label):
    return APOS + label


class Discovery:
    """Collects the token ids at the read positions (rows + s1s2's load-time probe) and label tokens in context."""

    def __init__(self, tok, enc):
        self.tok, self.enc = tok, enc
        self.encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        self.mk = markers_skeleton()
        self.ids = {"decide": set(), "option": set()}
        self.label_ctx = {}

    def add(self, qtype, instructions, texts, state, labels_after=()):
        p = positions(self.encode, self.enc, self.mk, qtype, instructions, texts, state)
        self.ids["decide"].add(p["ids"][p["decide"]])
        for o in p["options"]:
            self.ids["option"].add(p["ids"][o])
        for lab in labels_after:  # first token the label adds after the prompt (as the lab-s12 label contract reads it)
            b = self.encode(p["text"] + lab)
            if b[:len(p["ids"])] != p["ids"] or len(b) == len(p["ids"]):
                raise SystemExit(f"label {lab!r} merges with the end of the prompt (unstable boundary)")
            self.label_ctx.setdefault(lab, set()).add(b[len(p["ids"])])
        return p

    def probe(self):
        self.add("choice", PROBE_Q, PROBE_OPTS, PROBE_STATE)
        self.add("predicate", PROBE_Q, [], PROBE_STATE)

    def markers(self):
        out = json.loads(json.dumps(self.mk))
        for k in ("decide", "option"):
            text = out[k]["text"]
            toks = []
            for i in sorted(self.ids[k]):
                d = self.tok.decode([i])
                if not d or not text.endswith(d) and text[-1] not in d:
                    raise SystemExit(f"marker {k}: token {i} ({d!r}) at the read position does not hold the marker's end")
                toks.append({"id": int(i), "text": d})
            out[k]["tokens"] = toks
        return out

    def ref(self, i):
        d = self.tok.decode([i])
        if self.encode(d) != [i]:
            raise SystemExit(f"token {i} ({d!r}) does not round-trip to one token")
        return {"id": int(i), "text": d}
