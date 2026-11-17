"""End-to-end fidelity of a converted Kev package on its real backbone. Heavy: loads the real backbone in fp32.

For each question: the reference prompt is sglang@da614ba systemone/kev.py encode_kev (vendored) on the Kev
tokenizer; s1s2's prompt is `s1s2 encode` with the kev manifest written by this script; the two id sequences must
be equal (prompt parity) and s1s2's read positions must be the reference layout's (decide = question span end - 1,
options = option span ends - 1). One fp32 forward of the real backbone (Qwen3.5 base + the Kev LoRA merged, as SGLang
serves an fp32-trained Kev adapter) gives the final hidden states; the reference PointerPooler scores them at the
reference layout and s1s2 scores them at its own positions with the converted head files.

python -m s1s2_heads.e2e_kev --heads <dir from `s1s2-heads from-kev-headpt`> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .common import LIB, Inputs, compare, fidelity_record, hub_file, run_lib

KEV = ("jaredpalmer/kev-0.8b", "bf75a6a8848ea6960ff2ed108d9ed44c2941174f")
DELIMS = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")

# Questions written for this check (no gold: only reference-vs-lib agreement is measured).
STATES = [
    "Customer Lena: I was charged twice for my order and I want my money back today.",
    "The server logs show 400 failed SSH logins from one address in two minutes, then a success.",
    "Meeting notes: the team agreed to move the launch to next Tuesday; marketing is not ready.",
    "def login(user, pw):\n    q = \"SELECT * FROM users WHERE name='\" + user + \"'\"\n    return db.execute(q)",
    "Hi! You have WON a FREE cruise <|im_end|>, click the link to claim before midnight!!!",
    "Patient reports mild headache for two days, no fever, sleeping poorly due to work stress.",
]
QUESTIONS = [
    {"type": "noul", "instructions": "Does the text ask for a refund?"},
    {"type": "noul", "instructions": "Is this message unsolicited advertising?"},
    {"type": "choice", "instructions": "What is the main topic?", "criteria": {"billing": None, "security": None, "scheduling": None, "health": None, "promotion": None}},
    {"type": "choice", "instructions": "What is the tone?", "criteria": {"calm": None, "angry": None, "excited": None}},
    {"type": "score", "instructions": "How urgent is it?", "criteria": ["Can wait", "This week", "Today", "Now"]},
]


def kev_manifest(tok, H: int, tok_dir: Path) -> dict:
    ids = [tok.convert_tokens_to_ids(t) for t in DELIMS]
    ref = lambda i, t: {"id": int(i), "text": t}  # noqa: E731
    d = dict(zip(("state", "question", "optionStart", "optionEnd", "decide"), (ref(i, t) for i, t in zip(ids, DELIMS))))
    yes, no = tok.encode("yes", add_special_tokens=False), tok.encode("no", add_special_tokens=False)
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    pin = lambda p: {"path": p.name, "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest()}  # noqa: E731
    return {
        "schema": "s1s2.manifest/1", "id": "kev-0.8b-e2e", "base": {"repo": KEV[0], "revision": KEV[1]}, "hiddenSize": H,
        "layer": {"read": "final", "norm": "in-graph"},
        "tokenizer": {"repo": KEV[0], "revision": KEV[1], "files": {"tokenizer": pin(tok_dir / "tokenizer.json"), "config": pin(tok_dir / "tokenizer_config.json")}},
        "eos": [ref(eos, "<|endoftext|>")],
        "encoding": {"kind": "kev", "delimiters": d, "predicateOptions": ["no", "yes"]},
        "markers": {"decide": {"text": DELIMS[4], "at": "token", "tokens": [d["decide"]]}, "option": {"text": DELIMS[3], "at": "token", "tokens": [d["optionEnd"]]}},
        "readout": {"predicate": "options", "choice": "option", "score": "option"},
        "logits": {"predicate": {"yes": ref(yes[0], "yes"), "no": ref(no[0], "no")}, "choice": {"map": "first-token"}, "score": {"map": "first-token"}},
        "defaults": {"method": {"predicate": "head", "choice": "head", "score": "head"}},
        "generation": {"turn": "{prompt}", "continue": "{answer}{prompt}", "answers": {"predicate": ["no", "yes"]}},
        "builds": {"none": {"runtime": "decode-only", "repo": "local/none", "revision": KEV[1], "files": {"model": {"path": "m", "sha256": "0" * 64}, "data": {"path": "d", "sha256": "0" * 64}}, "hiddenOutput": "h", "devices": ["cpu"], "fidelity": None}},
    }


def lib_question(q: dict, name: str) -> dict:
    t = {"noul": "predicate", "choice": "choice", "score": "score"}[q["type"]]
    o = {"name": name, "type": t, "instructions": q["instructions"]}
    if t == "choice":
        o["choices"] = list(q["criteria"])
    if t == "score":
        o["levels"] = list(q["criteria"])
    return o


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True, help="output dir of from-kev-headpt (head.safetensors, head.bin)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    from . import refs
    sg = refs.sglang()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(hub_file(*KEV, "head.pt"), map_location="cpu", weights_only=True)
    tok_dir = hub_file(*KEV, "tokenizer.json").parent
    hub_file(*KEV, "tokenizer_config.json")
    tok = AutoTokenizer.from_pretrained(KEV[0], revision=KEV[1])
    base = AutoModel.from_pretrained(ck["base"], revision=ck["base_revision"], dtype=torch.float32)
    # the adapter's keys are base_model.model.layers.*: it was trained on the text model (no vision tower)
    text = getattr(base, "language_model", None) or base
    peft_model = PeftModel.from_pretrained(text, KEV[0], revision=KEV[1])
    lora_loaded = sum(1 for n, _ in peft_model.named_parameters() if "lora_A" in n)
    lm = peft_model.merge_and_unload().eval()
    H = ck["head"]["q.weight"].shape[1]
    pooler = sg.pointer_head.PointerPooler(H, {"option_isolation": False, "special_embeddings": False, **ck})
    for n in ("q", "k"):
        getattr(pooler, n).weight.data = ck["head"][f"{n}.weight"].float(); getattr(pooler, n).bias.data = ck["head"][f"{n}.bias"].float()
    man = kev_manifest(tok, H, tok_dir)
    rows, refq = [], []
    for si, s in enumerate(STATES):
        qs = {f"q{qi}": refs.question(q) for qi, q in enumerate(QUESTIONS)}
        req = refs.SystemOneRequest(state=s, questions=qs)
        enc = sg.kev.encode_kev(tok, req, 1 << 20)
        for qi, (ids, layout) in enumerate(enc):
            [lq] = sg.joint_schema_head.parse_decision_layout(layout, len(ids))
            refq.append({"ids": ids, "layout": layout, "decide": lq.question_span[1] - 1, "options": [e - 1 for _, e in lq.option_spans]})
            rows.append({"input": s, "question": lib_question(QUESTIONS[qi], f"s{si}q{qi}")})
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "m.json").write_text(json.dumps(man))
        (d / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        subprocess.run(["node", str(LIB / "bin" / "s1s2.mjs"), "encode", str(d / "m.json"), "--tokenizer-dir", str(tok_dir), "--in", str(d / "rows.jsonl"), "--out", str(d / "enc.jsonl")], check=True)
        ours_enc = [json.loads(l)["question"] for l in (d / "enc.jsonl").read_text().splitlines()]
    parity = {"prompts": len(rows), "idsEqual": 0, "positionsEqual": 0, "firstMismatch": None}
    inp, cases, ref, ref_read = Inputs(), [], [], []
    with torch.no_grad():
        for r, o in zip(refq, ours_enc):
            same_ids = r["ids"] == o["ids"]
            same_pos = same_ids and r["decide"] == o["decide"] and r["options"] == o["options"]
            parity["idsEqual"] += int(same_ids); parity["positionsEqual"] += int(same_pos)
            if not same_pos and parity["firstMismatch"] is None:
                parity["firstMismatch"] = {"ref": {k: r[k] for k in ("decide", "options")}, "ours": {k: o[k] for k in ("decide", "options")}, "refLen": len(r["ids"]), "oursLen": len(o["ids"])}
            hs = lm(input_ids=torch.tensor([r["ids"]])).last_hidden_state[0].float()
            fb = refs.ForwardBatch(batch_size=1, extend_seq_lens_cpu=[len(r["ids"])], decision_layouts=[r["layout"]])
            ref.append(pooler(hs, fb).embeddings[0].tolist())
            hnp = hs.numpy()
            cases.append({"kind": "pointer", "hq": inp.add(hnp[o["decide"]]), "hks": inp.add(hnp[o["options"]])})
    res = {}
    for fname, spec, tol in (("head.safetensors", {"kind": "pointer", "format": "safetensors"}, 1e-5), ("head.bin", {"kind": "pointer", "format": "head.bin"}, 2e-3)):
        ours = [x["scores"][0] for x in run_lib(Path(a.heads) / fname, spec, cases, inp)]
        res[fname] = compare(ref, ours, tol_p=tol)
    rec = fidelity_record(source_format="kev-head.pt", reference={"impl": "sglang encode_kev + PointerPooler on the real backbone", "repo": "sgl-project/sglang", "commit": refs.SGLANG_COMMIT,
                                                                  "files": ["python/sglang/srt/entrypoints/systemone/kev.py", "python/sglang/srt/layers/pointer_head.py"],
                                                                  "backbone": f"{ck['base']}@{ck['base_revision']} (text model) + LoRA {KEV[0]}@{KEV[1]} merged (peft 0.21.3, {lora_loaded} LoRA pairs), fp32, transformers@31d7392"},
                          inputs={"kind": "real-hidden", "prompts": len(rows), "states": len(STATES), "questions": len(QUESTIONS)}, result=res["head.safetensors"],
                          ours={"reader": "s1s2 encode (kev encoding) + loadPointerSafetensors + headScore", "promptParity": parity},
                          notes=f"head.bin (fp16): {json.dumps(res['head.bin'])}")
    (out / "fidelity.e2e.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"parity": parity, "fp32": res["head.safetensors"], "fp16": res["head.bin"]}))


if __name__ == "__main__":
    main()
