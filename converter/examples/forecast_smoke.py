"""Smoke test of the forecast mechanism end to end on SmolLM2-135M-Instruct (CPU, a few hundred prompts).

NOT a measurement of the reader: the labels are two programmatic rules of the answer text defined below, the model is tiny,
and nothing here claims accuracy. It checks that the pieces connect: layer export (fp32 and int4 builds), answers, labels,
head training, the manifest block, loading in s1s2 on Node CPU, forecast on the decide prefill, and the head output across
two precisions against a PyTorch reference.

  python converter/examples/forecast_smoke.py <stage> --work /tmp/fc-smoke --model <SmolLM2-135M-Instruct snapshot dir>
stages: prompts | collect | label | train | reference | build | package   (then: node test/smoke/forecast_parity.mjs <work>/pkg)
"""
import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
LIB = HERE.parents[1]

from s1s2_convert import forecast as F  # noqa: E402
from s1s2_convert.util import read_json, write_json, read_jsonl, write_jsonl, sha256_file, log  # noqa: E402

LAYER = 14
QUESTION = {"name": "q", "type": "predicate", "instructions": "Does the request ask for a list?"}
TOPICS = ["the moon", "rivers", "bread", "volcanoes", "chess", "the ocean", "bees", "trains", "tea", "mountains", "libraries", "rain", "music",
          "gardens", "the internet", "coffee", "birds", "bridges", "winter", "football", "clocks", "forests", "ice", "maps", "paper"]
TEMPLATES = ["List three facts about {t}.", "What is {t}?", "Write one sentence about {t}.", "Give me a numbered list of ideas about {t}.",
             "Explain {t} to a child.", "Write a short poem about {t}.", "Name five things related to {t}.", "Why do people like {t}?",
             "Describe {t} in a few words.", "Tell me a story about {t}.", "Give me steps to learn about {t}.", "Is {t} important? Answer briefly."]


def label_rules(text):
    """The programmatic labels of this smoke test (defined by the one running it, not by the library)."""
    return {"numbered_list": int(bool(re.search(r"(?m)^\s*\d+[.)]\s", text))), "long_answer": int(len(text.split()) > 40)}


def draft_manifest(model, work):
    m = read_json(LIB / "manifests" / "smollm2-135m-instruct.en.json")
    m["id"] = "smollm2-135m-instruct.forecast-smoke"
    m["builds"] = {}
    m.pop("lab", None)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prompts", "collect", "label", "train", "reference", "build", "package"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--head", default="logistic")
    ap.add_argument("--ref-rows", type=int, default=100)
    a = ap.parse_args(argv)
    W = Path(a.work)
    W.mkdir(parents=True, exist_ok=True)
    man = W / "manifest.draft.json"
    if not man.exists():
        write_json(man, draft_manifest(a.model, W))
    if a.stage == "prompts":
        rng = random.Random(7)
        combos = [(t, T) for T in TEMPLATES for t in TOPICS]
        rng.shuffle(combos)
        write_jsonl(W / "prompts.jsonl", [{"id": f"p{i:03d}", "prompt": T.format(t=t)} for i, (t, T) in enumerate(combos[:a.n])])
    elif a.stage == "collect":
        F.collect(a.model, read_json(man), W / "prompts.jsonl", W / "responses.jsonl", max_new_tokens=48, limit=a.limit)
    elif a.stage == "label":
        rows = read_jsonl(W / "responses.jsonl")
        write_jsonl(W / "labels.jsonl", [{"id": r["id"], "labels": label_rules(r["response"])} for r in rows])
        for k in ("numbered_list", "long_answer"):
            log("smoke", f"{k}: {sum(label_rules(r['response'])[k] for r in rows)}/{len(rows)}")
    elif a.stage == "train":
        write_json(W / "question.json", QUESTION)
        F.main(["train", "--model", a.model, "--manifest", str(man), "--responses", str(W / "responses.jsonl"), "--labels", str(W / "labels.jsonl"),
                "--question", str(W / "question.json"), "--layer", str(LAYER), "--head", a.head, "--platt", "--id", "smoke-gate", "--out", str(W / "head"), "--lib", str(LIB)])
    elif a.stage == "reference":
        import torch
        from safetensors.torch import load_file
        rows = read_jsonl(W / "responses.jsonl")[:a.ref_rows]
        X, _ = F.features(a.model, str(man), rows, QUESTION, LAYER, "last", W / "refwork", str(LIB))
        t = load_file(str(W / "head" / "forecast.safetensors"))
        blk = read_json(W / "head" / "forecast.manifest.json")["forecast"]
        z = F.head_scores(t, (X - t["mean"]) / t["std"]) / torch.tensor(blk["calibration"]["temperature"])
        if "platt" in blk["calibration"]:
            z = torch.tensor(blk["calibration"]["platt"]["a"]) * z + torch.tensor(blk["calibration"]["platt"]["b"])
        names = [b["name"] for b in blk["behaviors"]]
        write_jsonl(W / "reference.jsonl", [{"id": r["id"], "prompt": r["prompt"], "behaviors": {n: {"p": float(torch.sigmoid(z[i, j]))} for j, n in enumerate(names)}} for i, r in enumerate(rows)])
    elif a.stage == "build":
        for prec, ep in (("fp32", "cpu"), ("int8", "webgpu"), ("int4", "webgpu")):
            info = F.build_with_layer(a.model, W / f"build-{prec}", W, LAYER, precision=prec, ep=ep)
            write_json(W / f"build-{prec}.json", info)
    elif a.stage == "package":
        pkg = W / "pkg"
        shutil.rmtree(pkg, ignore_errors=True)
        pkg.mkdir()
        m = draft_manifest(a.model, W)
        repo, rev = "local/smollm2-forecast-smoke", "0" * 39 + "1"
        pin = lambda p, rel: {"path": rel, "sha256": sha256_file(p), "size": Path(p).stat().st_size}  # noqa: E731
        for f in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(Path(a.model) / f, pkg / f)
        m["tokenizer"] = {"repo": repo, "revision": rev, "files": {"tokenizer": pin(pkg / "tokenizer.json", "tokenizer.json"), "config": pin(pkg / "tokenizer_config.json", "tokenizer_config.json")}}
        for prec, devs in (("fp32", ["cpu"]), ("int8", ["webgpu", "cpu"]), ("int4", ["webgpu", "cpu"])):
            src = W / f"build-{prec}"
            (pkg / prec).mkdir()
            for f in ("model.onnx", "model.onnx.data", "genai_config.json"):
                shutil.copyfile(src / f, pkg / prec / f)
            g = read_json(src / "genai_config.json")["model"]["decoder"]
            for dev in devs:  # the repo's convention: one entry per device, same files (a WebGPU graph also runs on Node CPU)
                m["builds"][f"genai-{prec}-{dev}"] = {
                    "runtime": "genai", "repo": repo, "revision": rev, "hiddenOutput": "hidden_states", **F.build_patch(LAYER),
                    "files": {"model": pin(pkg / prec / "model.onnx", f"{prec}/model.onnx"), "data": pin(pkg / prec / "model.onnx.data", f"{prec}/model.onnx.data"),
                              "genaiConfig": pin(pkg / prec / "genai_config.json", f"{prec}/genai_config.json")},
                    "devices": [dev], "dtype": prec, "kvHeadDim": g["head_size"], "fidelity": None}
        blk = read_json(W / "head" / "forecast.manifest.json")["forecast"]
        shutil.copyfile(W / "head" / "forecast.safetensors", pkg / "forecast.safetensors")
        blk["head"]["file"] = pin(pkg / "forecast.safetensors", "forecast.safetensors")
        m["forecast"] = blk
        write_json(pkg / "manifest.json", m)
        log("smoke", f"package: {pkg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
