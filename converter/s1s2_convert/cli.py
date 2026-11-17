"""s1s2-convert: HF model + labelled JSONL -> S1S2 package (spec/s1s2-package.md 0.1-draft) that s1s2 loads.

Stages (each resumable: a finished stage leaves <work>/.done-<stage> and is skipped next time; `all` runs them in order):
  prepare   read + split the dataset, build the prompt layout for the model's chat template, discover marker / label
            tokens, write the draft manifest, encode every training/dev prompt WITH s1s2 (`s1s2 encode`) and check
            the Python rendering gives the same ids and read positions
  train     LoRA + decision head (lab-s12 001 recipe), dev-selected
  fuse      merge the LoRA into the base weights (same keys and files)
  evidence  evidence head on the fused model (lab-s12 005 recipe), only when rows carry `evidence`
  build     ONNX Runtime GenAI 0.19 int4 WebGPU graph with hidden_states (+ lab-s12 builder fixes)
  package   package directory + manifest.json + provenance, then `s1s2 validate --strict`
  parity    PyTorch (fused, fp32) vs s1s2 on the int4 graph (onnxruntime-node CPU) over the dev rows

Example:
  s1s2-convert all --model HuggingFaceTB/SmolLM2-135M-Instruct --data rows.jsonl \\
      --question-template "Does this passage address {requirement}? Answer with one option only." \\
      --work /tmp/s1s2-work --out /tmp/s1s2-pkg
"""
import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

from . import __version__
from . import data as D
from . import layout as LY
from .util import log, write_json, read_json, read_jsonl, write_jsonl, run, done, mark_done, lib_dir, now_iso, sha256_file

STAGES = ["prepare", "train", "fuse", "evidence", "build", "package", "parity"]
EVIDENCE_TEMPLATE = "This passage addresses {requirement}. Which of the passage's sentences below shows it? Answer with one option only."


def resolve_model(model, revision=None, repo=None, license_id=None):
    """-> {dir, repo, revision, license}. Local dir (PROVENANCE.json / HF cache path / --revision) or a Hub id."""
    p = Path(model)
    if p.exists():
        prov = json.loads((p / "PROVENANCE.json").read_text()) if (p / "PROVENANCE.json").exists() else {}
        m = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{40})", str(p.resolve()))
        rev = revision or prov.get("revision") or (m.group(3) if m else None)
        rid = repo or prov.get("repo") or (f"{m.group(1)}/{m.group(2)}" if m else f"local/{p.name}")
        d = p.resolve()
    else:
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download(model, revision=revision))
        m = re.search(r"/snapshots/([0-9a-f]{40})", str(d))
        rev, rid = (m.group(1) if m else revision), repo or model
    if not rev or not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise SystemExit(f"{model}: pinned 40-hex revision unknown (pass --revision)")
    lic = license_id
    if not lic and (d / "README.md").exists():
        fm = re.match(r"^---\n(.*?)\n---", (d / "README.md").read_text(), re.S)
        lm = re.search(r"^license:\s*([^\s#]+)", fm.group(1), re.M) if fm else None
        lic = lm.group(1) if lm else None
    spdx = {"apache-2.0": "Apache-2.0", "mit": "MIT", "llama3.2": "LicenseRef-llama3.2", "gemma": "LicenseRef-gemma"}
    return {"dir": str(d), "repo": rid, "revision": rev, "license": spdx.get((lic or "").lower(), lic or "NOASSERTION")}


def args_parser():
    ap = argparse.ArgumentParser(prog="s1s2-convert", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--work", required=True, help="work directory (intermediate files; resumable)")
    ap.add_argument("--out", help="package directory (package stage)")
    ap.add_argument("--model", help="HF model id or local directory (prepare)")
    ap.add_argument("--revision", help="pinned 40-hex revision (local dirs without PROVENANCE.json; Hub ids)")
    ap.add_argument("--repo", help="repo id recorded as base (default: from the model)")
    ap.add_argument("--license", dest="license_id", help="SPDX id of the base (default: README front matter)")
    ap.add_argument("--data", help="labelled JSONL (prepare)")
    ap.add_argument("--question-template", help="question for rows with `requirement` ({requirement})")
    ap.add_argument("--evidence-template", default=EVIDENCE_TEMPLATE, help="evidence question ({requirement})")
    ap.add_argument("--labels", help="comma-separated label set and order (default: order of first appearance)")
    ap.add_argument("--dev-fraction", type=float, default=0.2)
    ap.add_argument("--system", default=None, help="system message (default: none; the chat template's own default applies)")
    ap.add_argument("--chat-kwargs", default="{}", help='JSON kwargs for apply_chat_template, e.g. {"enable_thinking": false}')
    ap.add_argument("--cue", default="Answer:", help="text after the generation prompt (the answer cue)")
    ap.add_argument("--id", help="package id (default: <model name>-s1s2)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=None, help="peak lr of the head (kev head_lr; default: --lr)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="cap on optimizer steps (smoke runs)")
    ap.add_argument("--evidence-epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--precision", default="int4", choices=["int4", "int8", "fp16", "fp32"], help="weight precision of the GenAI build")
    ap.add_argument("--ep", default=None, help="builder execution provider (default: webgpu; cpu for fp32)")
    ap.add_argument("--builder-src", help="run the builder from this checkout (e.g. <fork>/src/python/py/models) instead of the patched pip copy")
    ap.add_argument("--lib", help="s1s2 checkout (default: the converter's parent directory)")
    ap.add_argument("--parity-build", help="build the parity stage runs (default: the first *-cpu build)")
    ap.add_argument("--parity-rows", type=int, default=0, help="dev rows for the parity stage (0 = all)")
    return ap


def stage_prepare(a, W):
    from transformers import AutoTokenizer
    if not a.model or not a.data:
        raise SystemExit("prepare: --model and --data are required")
    base = resolve_model(a.model, a.revision, a.repo, a.license_id)
    labels = a.labels.split(",") if a.labels else None
    rows, dmeta = D.load(a.data, a.question_template, labels, a.dev_fraction, a.seed)
    labels = dmeta["labels"]
    tok = AutoTokenizer.from_pretrained(base["dir"], local_files_only=True)
    encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    enc, gen = LY.build_encoding(tok, a.system, json.loads(a.chat_kwargs), a.cue)
    disc = LY.Discovery(tok, enc)
    disc.probe()
    texts = [LY.option_text(l) for l in labels]
    rng = random.Random(a.seed)
    plan = []
    for r in rows:
        for e in range(a.epochs if r["split"] == "train" else 1):
            perm = labels[:]
            rng.shuffle(perm)
            p = disc.add("choice", r["question"], [LY.option_text(l) for l in perm], r["text"], labels_after=texts)
            plan.append({"kind": "decision", "row": r["id"], "split": r["split"], "epoch": e, "perm": perm, "target": perm.index(r["label"]),
                         "py": {"ids": p["ids"], "decide": p["decide"], "options": p["options"]},
                         "req": {"input": r["text"], "question": {"type": "choice", "instructions": r["question"], "choices": [LY.option_text(l) for l in perm]}}})
        if r["evidence"] is not None:
            if not r["requirement"]:
                raise SystemExit(f"row {r['id']}: evidence rows need `requirement` (fills the evidence question)")
            instr = a.evidence_template.replace("{requirement}", r["requirement"])
            p = disc.add("choice", instr, [LY.APOS + s for s in r["sentences"]], r["text"])
            plan.append({"kind": "evidence", "row": r["id"], "split": r["split"], "targets": r["evidence"],
                         "py": {"ids": p["ids"], "decide": p["decide"], "options": p["options"]},
                         "req": {"input": r["text"], "evidence": {"requirement": r["requirement"], "sentences": r["sentences"]}}})
    # label tokens: one id per label at the cue, the same in every prompt
    ctx = {}
    for lab in texts:
        ids = disc.label_ctx[lab]
        if len(ids) != 1:
            raise SystemExit(f"label {lab!r}: first token after the prompt differs between prompts ({sorted(ids)})")
        ctx[lab] = next(iter(ids))
    if len(set(ctx.values())) != len(ctx):
        raise SystemExit(f"labels share a first token at the cue: {ctx}; the logits readout cannot tell them apart")
    alone = all(encode(t) == [ctx[t]] for t in texts)
    choice_logits = ({"map": "tokens", "tokens": [{"label": t, **disc.ref(ctx[t])} for t in texts]} if alone else {"map": "first-token"})
    pred = None
    for y, n in ((" Yes", " No"), ("Yes", "No"), (" yes", " no")):
        if len(encode(y)) == 1 and len(encode(n)) == 1:
            pred = {"yes": disc.ref(encode(y)[0]), "no": disc.ref(encode(n)[0])}
            break
    if pred is None:
        raise SystemExit("no single-token yes/no pair for logits.predicate")
    eos = {tok.eos_token_id}
    gc = Path(base["dir"]) / "generation_config.json"
    if gc.exists():
        e = json.loads(gc.read_text()).get("eos_token_id")
        eos |= set(e if isinstance(e, list) else [e] if e is not None else [])
    eos.discard(None)
    cfg = json.loads((Path(base["dir"]) / "config.json").read_text())
    hidden = (cfg.get("text_config") or {}).get("hidden_size", cfg.get("hidden_size"))
    bos = tok.bos_token
    warm = disc.ref(encode(bos)[0]) if bos and len(encode(bos)) == 1 else disc.ref(sorted(eos)[0])
    pid = a.id or f"{base['repo'].split('/')[-1].lower()}-s1s2"
    lab_texts = [LY.option_text(l) for l in labels]
    draft = {
        "schema": "s12.manifest/1", "id": pid, "base": {"repo": base["repo"], "revision": base["revision"]},
        "hiddenSize": hidden, "layer": {"read": "final", "norm": "in-graph"},
        "tokenizer": {"repo": f"local/{pid}", "revision": base["revision"], "files": {}},
        "eos": [disc.ref(i) for i in sorted(eos)], "warmup": warm, "encoding": enc, "markers": disc.markers(),
        "readout": {"predicate": "last", "choice": "option", "score": "last"},
        "logits": {"predicate": pred, "choice": choice_logits, "score": {"map": "first-token"}},
        "defaults": {"method": {"predicate": "logits", "choice": "head", "score": "logits"}, "maxSafeTokens": 2048},
        "generation": {**gen, "answers": {"predicate": [pred["no"]["text"], pred["yes"]["text"]]}},
        "builds": {},
    }
    if dmeta["evidence_rows"]:  # the evidence prompt is rendered from this block (sentencePrefix: labels start with a space)
        draft["evidence"] = {"instructions": a.evidence_template, "sentencePrefix": LY.APOS}
    if len({r["question"] for r in rows}) == 1:
        draft["tasks"] = {"decision": {"type": "choice", "instructions": rows[0]["question"],
                                       "choices": [{"value": t, **({"token": disc.ref(ctx[t])} if alone else {})} for t in lab_texts]}}
    # tokenizer files pinned now so the draft validates (the package stage re-pins the copies)
    for k, f in (("tokenizer", "tokenizer.json"), ("config", "tokenizer_config.json")):
        draft["tokenizer"]["files"][k] = {"path": f, "sha256": sha256_file(Path(base["dir"]) / f), "size": (Path(base["dir"]) / f).stat().st_size}
    write_json(W / "manifest.draft.json", draft)
    write_jsonl(W / "rows.jsonl", rows)
    write_jsonl(W / "plan.jsonl", plan)
    write_jsonl(W / "encode_in.jsonl", [p["req"] for p in plan])
    lib = lib_dir(a.lib)
    run(["node", lib / "bin" / "s1s2.mjs", "encode", W / "manifest.draft.json", "--tokenizer-dir", base["dir"],
         "--in", W / "encode_in.jsonl", "--out", W / "encoded.jsonl"], "prepare")
    encd = read_jsonl(W / "encoded.jsonl")
    bad = []
    for p, e in zip(plan, encd):
        lib_e = e.get("question") if p["kind"] == "decision" else e.get("evidence")
        if not lib_e or lib_e["ids"] != p["py"]["ids"] or lib_e["decide"] != p["py"]["decide"] or lib_e["options"] != p["py"]["options"]:
            bad.append(p["row"])
        if p["kind"] == "decision" and alone and lib_e and lib_e["labelIds"] != [ctx[LY.option_text(l)] for l in p["perm"]]:
            bad.append(p["row"] + " (label ids)")
    if bad or len(encd) != len(plan):
        raise SystemExit(f"prepare: s1s2 encoding differs from the Python rendering for {len(bad)} prompts (first: {bad[:5]})")
    write_json(W / "prepare.json", {"base": base, "dataset": {**dmeta, "name": Path(a.data).name}, "labels": labels, "label_texts": lab_texts,
                                    "logits_choice_map": choice_logits["map"], "question_template": a.question_template,
                                    "evidence_template": a.evidence_template if dmeta["evidence_rows"] else None,
                                    "cue": a.cue, "system": a.system, "chat_kwargs": json.loads(a.chat_kwargs), "id": pid,
                                    "epochs": a.epochs, "seed": a.seed, "prompts": len(plan), "lib_encode_identical": len(plan)})
    log("prepare", f"{len(rows)} rows ({dmeta['splits']}), labels {labels}, {len(plan)} prompts; s1s2 encoding == Python rendering on all")


def encoded_rows(W, kind, split):
    out = []
    for p, e in zip(read_jsonl(W / "plan.jsonl"), read_jsonl(W / "encoded.jsonl")):
        if p["kind"] != kind or p["split"] != split:
            continue
        x = e["question"] if kind == "decision" else e["evidence"]
        out.append({"row": p["row"], "ids": x["ids"], "decide": x["decide"], "options": x["options"],
                    **({"target": p["target"], "epoch": p["epoch"], "perm": p["perm"], "labelIds": x["labelIds"]} if kind == "decision" else {"targets": p["targets"]}),
                    "req": p["req"]})
    return out


def stage_train(a, W):
    from .train import train_decision
    prep = read_json(W / "prepare.json")
    train_decision(prep["base"]["dir"], encoded_rows(W, "decision", "train"), encoded_rows(W, "decision", "dev"), W / "train",
                   device=a.device, epochs=prep["epochs"], lr=a.lr, head_lr=a.head_lr, lora_r=a.lora_r, batch=a.batch, accum=a.accum,
                   eval_every=a.eval_every, seed=prep["seed"], max_steps=a.max_steps)


def stage_fuse(a, W):
    from .train import fuse
    prep = read_json(W / "prepare.json")
    tr = read_json(W / "train" / "train.json")
    info = fuse(prep["base"]["dir"], W / "train" / "adapter.safetensors", tr["lora_scale"], W / "fused")
    write_json(W / "fuse.json", info)
    log("fuse", f"{info['modules_fused']} modules fused, relative delta norm mean {info['relative_delta_norm']['mean']:.2e}")


def stage_evidence(a, W):
    prep = read_json(W / "prepare.json")
    if not prep["dataset"]["evidence_rows"]:
        log("evidence", "no evidence rows: skipped")
        return
    from .train import train_evidence
    train_evidence(W / "fused", W / "train" / "head", encoded_rows(W, "evidence", "train"), encoded_rows(W, "evidence", "dev"),
                   W / "evidence", device=a.device, epochs=a.evidence_epochs, seed=prep["seed"])


def stage_build(a, W):
    from .build import build
    ep = a.ep or ("cpu" if a.precision == "fp32" else "webgpu")
    info = build(W / "fused", W / f"build-{a.precision}", W, precision=a.precision, ep=ep, builder_src=a.builder_src)
    info["ep"] = ep
    write_json(W / f"build-{a.precision}.json", info)


def stage_package(a, W):
    from .package import assemble
    if not a.out:
        raise SystemExit("package: --out is required")
    prep = read_json(W / "prepare.json")
    draft = read_json(W / "manifest.draft.json")
    tr = read_json(W / "train" / "train.json")
    heads = {"decision": (W / "train" / "head", ["choice"])}
    evidence, ev_info = None, None
    if (W / "evidence" / "evidence.json").exists():
        ev_info = read_json(W / "evidence" / "evidence.json")
        heads["evidence"] = (W / "evidence" / "head", ["evidence"])
        evidence = {"instructions": prep["evidence_template"], "sentencePrefix": LY.APOS,
                    "source": f"s1s2-convert {__version__}: lab-s12 005 recipe on the fused model, epoch {ev_info['selected']['epoch']} chosen by dev loss; PyTorch fp32 reference, int4 see provenance.parity"}
    builds, bprov = [], {}
    for prec in ("int4", "int8", "fp16", "fp32"):
        if (W / f"build-{prec}.json").exists():
            bp = read_json(W / f"build-{prec}.json")
            # a WebGPU graph also runs on onnxruntime-node CPU for plain decoders (s1s2 runtime.ts: opt-in by "cpu")
            devs = [["webgpu", ["webgpu"]], ["cpu", ["cpu"]]] if bp.get("ep", "webgpu") == "webgpu" else [["cpu", ["cpu"]]]
            builds.append({"dir": W / f"build-{prec}", "subdir": prec, "dtype": prec, "devices": devs})
            for suffix, _ in devs:
                bprov[f"genai-{prec}-{suffix}"] = bp
    if not builds:
        raise SystemExit("package: no build (run the build stage)")
    base = prep["base"]
    prov = {"generator": {"name": "s1s2-convert", "version": __version__}, "createdAt": now_iso(),
            "source": {"format": "hf-transformers-causal-lm", "repo": base["repo"], "revision": base["revision"],
                       "files": ["config.json", "model.safetensors | model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"],
                       "data": "s1s2-convert jsonl (text, question|requirement, label, evidence?, sentences?, split?)"},
            "base": {"repo": base["repo"], "revision": base["revision"], "license": base["license"]},
            "dataset": {"sha256": prep["dataset"]["sha256"], "rows": prep["dataset"]["rows"], "name": prep["dataset"]["name"],
                        "splits": prep["dataset"]["splits"]},
            "training": {"recipe": tr["recipe"], "seed": tr["hyperparameters"]["seed"], "hyperparameters": tr["hyperparameters"],
                         "selection": tr["selection"],
                         "metrics": {"decision_dev_pytorch": tr["selected"], **({"evidence_dev_pytorch": ev_info["selected"]} if ev_info else {})},
                         "prompt": {"labels": prep["label_texts"], "questionTemplate": prep["question_template"], "cue": prep["cue"],
                                    "system": prep["system"], "chatKwargs": prep["chat_kwargs"],
                                    "usage": "decide({input: text, questions: [{type: 'choice', instructions: <question>, choices: labels}]})"},
                         "fuse": read_json(W / "fuse.json"), **({"evidence": {k: ev_info[k] for k in ("recipe", "hyperparameters", "selection", "rows")}} if ev_info else {})},
            "builds": bprov,
            "notes": "Metrics in training.metrics are on the dev split, read by PyTorch on the training weights (decision: LoRA model; evidence: fused model). Int4 behaviour is in provenance.parity when the parity stage ran."}
    draft.pop("evidence", None)
    m = assemble(draft, a.out, tokenizer_dir=W / "fused", builds=builds, heads=heads, provenance=prov, license_id=base["license"], evidence=evidence)
    lib = lib_dir(a.lib)
    run(["node", lib / "bin" / "s1s2.mjs", "validate", "--strict", a.out], "package")
    log("package", f"{m['id']}: builds {list(m['builds'])}, heads {list(heads)} -> {a.out}")


def stage_parity(a, W):
    """PyTorch reference (fused weights, fp32) vs s1s2 (package, int4 graph, onnxruntime-node CPU) on the dev rows."""
    import torch
    from . import head as H
    from .train import load_model, tensors
    from transformers import AutoTokenizer
    if not a.out:
        raise SystemExit("parity: --out (the package directory) is required")
    pkg = Path(a.out)
    m = read_json(pkg / "manifest.json")
    build = a.parity_build or next((b for b in m["builds"] if b.endswith("-cpu")), None)
    dec_rows = encoded_rows(W, "decision", "dev")
    ev_rows = {r["row"]: r for r in encoded_rows(W, "evidence", "dev")}
    if a.parity_rows:
        dec_rows = dec_rows[:a.parity_rows]
    rows_in = [{"id": r["row"], "input": r["req"]["input"], "question": r["req"]["question"],
                **({"evidence": ev_rows[r["row"]]["req"]["evidence"]} if r["row"] in ev_rows else {})} for r in dec_rows]
    write_jsonl(W / "parity_in.jsonl", rows_in)
    lib = lib_dir(a.lib)
    node = Path(__file__).resolve().parent / "node" / "run_package.mjs"
    run(["node", node, lib, pkg, build, W / "parity_in.jsonl", W / "parity_lib.jsonl"], "parity", env={"S12_CACHE": str(W / "s12-cache")})
    got = {r["id"]: r for r in read_jsonl(W / "parity_lib.jsonl")}
    # reference
    tok = AutoTokenizer.from_pretrained(str(W / "fused"), local_files_only=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    model = load_model(W / "fused", "cpu").eval()
    dec = model.get_decoder()
    lm_head = model.get_output_embeddings()
    dh, _ = H.load(pkg / "heads" / "decision")
    eh = H.load(pkg / "heads" / "evidence")[0] if (pkg / "heads" / "evidence").exists() else None
    prep = read_json(W / "prepare.json")
    res = []
    with torch.no_grad():
        for r in dec_rows:
            ids, att = tensors([r], pad, "cpu")
            h = dec(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state.float()[0]
            ph = torch.softmax(H.scores(dh, h, r["decide"], r["options"]), -1)
            L = lm_head(h[-1:]).float()[0]
            pl = torch.softmax(L[torch.tensor(r["labelIds"])], -1)
            x = {"row": r["row"], "gold": r["target"], "ref_head": ph.tolist(), "ref_logits": pl.tolist()}
            if eh is not None and r["row"] in ev_rows:
                e = ev_rows[r["row"]]
                ids, att = tensors([e], pad, "cpu")
                he = dec(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state.float()[0]
                x["ref_evidence"] = torch.softmax(H.scores(eh, he, e["decide"], e["options"]), -1).tolist()
                x["gold_evidence"] = e["targets"]
            res.append(x)
    gap = lambda p: sorted(p, reverse=True)[0] - sorted(p, reverse=True)[1]  # noqa: E731
    am = lambda p: max(range(len(p)), key=lambda i: p[i])  # noqa: E731

    def summary(ref_key, lib_key, gold_key=None):
        pairs = [(x[ref_key], got[x["row"]][lib_key]["probabilities"], x.get(gold_key) if gold_key else x["gold"]) for x in res
                 if ref_key in x and lib_key in got[x["row"]]]
        if not pairs:
            return None
        agree = sum(am(a) == am(b) for a, b, _ in pairs)
        gold_ok = lambda p, g: am(p) in g if isinstance(g, list) else am(p) == g  # noqa: E731
        ma, mb = [gap(a) for a, _, _ in pairs], [gap(b) for _, b, _ in pairs]
        n = len(pairs)
        mu_a, mu_b = sum(ma) / n, sum(mb) / n
        cov = sum((x - mu_a) * (y - mu_b) for x, y in zip(ma, mb))
        va, vb = sum((x - mu_a) ** 2 for x in ma), sum((y - mu_b) ** 2 for y in mb)
        return {"n": n, "decision_agreement": agree, "acc_pytorch_fp32": sum(gold_ok(a, g) for a, _, g in pairs) / n,
                "acc_lib": sum(gold_ok(b, g) for _, b, g in pairs) / n,
                "max_abs_prob_diff": max(max(abs(x - y) for x, y in zip(a, b)) for a, b, _ in pairs),
                "margin_pearson": cov / (va * vb) ** 0.5 if va > 0 and vb > 0 else None,
                "margin_mean_pytorch": mu_a, "margin_mean_lib": mu_b}
    out = {"package": str(pkg), "build": build, "reference": "PyTorch fp32 on the fused checkpoint (CPU)",
           "runtime": f"s1s2 S12.load build {build} (onnxruntime-node CPU, {m['builds'][build].get('dtype')} graph)", "dev_rows": len(res),
           "head": summary("ref_head", "head"), "logits": summary("ref_logits", "logits"),
           "evidence": summary("ref_evidence", "evidence", "gold_evidence"), "labels": prep["label_texts"]}
    write_json(W / f"PARITY-{build}.json", out)
    m["provenance"].setdefault("parity", {})[build] = {k: out[k] for k in ("reference", "runtime", "dev_rows", "head", "logits", "evidence")}
    # spec 7: fidelity of the measured build against the reference reader (this graph on onnxruntime-node CPU)
    n = out["head"]["n"]
    m["builds"][build]["fidelity"] = {
        "reference": "pytorch-fp32 (fused checkpoint; s1s2-convert parity)", "n": n,
        "decisionChanged": {"head": 1 - out["head"]["decision_agreement"] / n, "logits": 1 - out["logits"]["decision_agreement"] / n},
        "measuredAt": now_iso(),
        "notes": (f"dev split, onnxruntime-node CPU EP; evidence head agreement {out['evidence']['decision_agreement']}/{out['evidence']['n']}; "
                  if out["evidence"] else "dev split, onnxruntime-node CPU EP; ") + "order inversion not measured; details in provenance.parity"}
    write_json(pkg / "manifest.json", m)
    run(["node", lib / "bin" / "s1s2.mjs", "validate", "--strict", pkg], "parity")
    for k in ("head", "logits", "evidence"):
        s = out[k]
        if s:
            log("parity", f"{build} {k}: agreement {s['decision_agreement']}/{s['n']}, acc PyTorch {s['acc_pytorch_fp32']:.3f} vs lib {s['acc_lib']:.3f}, "
                f"max |dp| {s['max_abs_prob_diff']:.3f}, margin r {s['margin_pearson'] if s['margin_pearson'] is None else round(s['margin_pearson'], 3)}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0].startswith("from-"):  # published head formats -> S1S2 heads (converters/heads, s1s2-heads)
        from s1s2_heads.cli import main as heads_main  # pip install -e ../converters/heads
        return heads_main(argv) or 0
    a = args_parser().parse_args(argv)
    W = Path(a.work)
    W.mkdir(parents=True, exist_ok=True)
    fns = {"prepare": stage_prepare, "train": stage_train, "fuse": stage_fuse, "evidence": stage_evidence, "build": stage_build,
           "package": stage_package, "parity": stage_parity}
    todo = STAGES if a.stage == "all" else [a.stage]
    for s in todo:
        if done(W, s) and a.stage == "all":
            log(s, "done earlier: skipped")
            continue
        fns[s](a, W)
        mark_done(W, s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
