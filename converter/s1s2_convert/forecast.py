"""Forecast reader tooling (spec/FORECAST.md): export a build that exposes a mid-layer residual stream, and train the
forecast head offline.

  python -m s1s2_convert train-forecast-head collect ...   answers of the model to each prompt (greedy), written to a jsonl
  python -m s1s2_convert train-forecast-head train ...     features + your labels -> head file + `forecast` manifest block

The converter contains no judge and no labels: behavior labels are a jsonl you supply ({"id", "labels": {"<behavior>": 0|1}}).
With a LoRA fused into the weights, train on the FUSED model (`--model` = the fused directory): a head trained on the base
does not transfer. Layer convention: `--layer L` is the OUTPUT of decoder block L (0-indexed, 0 <= L <= num_layers-2);
the GenAI builder option is `aux_hidden_state_layers=L+1` (entry i = residual stream entering layer i) and the Hugging Face
reference is `hidden_states[L+1]`.
"""
import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

from .util import log, write_json, read_json, read_jsonl, write_jsonl, run, lib_dir, sha256_file, now_iso

BINS = 32
POOLINGS = ("last", "mean", "bins32")
AUX_OUTPUT = "aux_hidden_states"


def check_layer(layer, num_layers):
    if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer <= num_layers - 2:
        raise SystemExit(f"--forecast-layer {layer}: must be an integer in [0, {num_layers - 2}] "
                         f"(the output of decoder block L; a model with {num_layers} blocks; the last block is read after the final norm as hidden_states)")
    return layer


def builder_option(layer):
    """GenAI builder extra option for the output of block `layer`: entry i is the residual stream entering layer i."""
    return f"aux_hidden_state_layers={layer + 1}"


def num_layers_of(model_dir):
    cfg = read_json(Path(model_dir) / "config.json")
    return (cfg.get("text_config") or {}).get("num_hidden_layers", cfg.get("num_hidden_layers"))


def build_with_layer(fused_dir, out_dir, work, layer, **kw):
    """`build.build` with the forecast layer exposed as `aux_hidden_states` (the manifest build gets forecastOutput / forecastLayer)."""
    from .build import build
    check_layer(layer, num_layers_of(fused_dir))
    info = build(fused_dir, out_dir, work, extra=[*(kw.pop("extra", None) or []), builder_option(layer)], **kw)
    info["forecast"] = {"output": AUX_OUTPUT, "layer": layer}
    return info


def build_patch(layer):
    """Keys to add to a manifest build whose graph was exported with `build_with_layer`."""
    return {"forecastOutput": AUX_OUTPUT, "forecastLayer": layer}


def truncate_graph(onnx_in, onnx_out, output=AUX_OUTPUT):
    """Optional: keep only the part of an exported graph that computes `output` (blocks after the forecast layer, the final
    norm and the LM head are dropped). The result is a forecast-only graph: no logits, no KV outputs, so the s1s2 backends
    cannot run it; it is for running the gate alone. Weights are re-saved next to the new file."""
    import onnx
    from onnx.utils import Extractor
    m = onnx.load(str(onnx_in), load_external_data=True)
    inputs = [i.name for i in m.graph.input]
    sub = Extractor(m).extract_model(inputs, [output])
    onnx.save_model(sub, str(onnx_out), save_as_external_data=True, all_tensors_to_one_file=True, location=Path(onnx_out).name + ".data")
    return {"inputs": inputs, "output": output, "nodes": len(sub.graph.node), "nodesFull": len(m.graph.node)}


# ---------------------------------------------------------------- features

def pool(h, pooling):
    """h: [n, hidden] torch tensor of the context tokens 0..n-1 -> pooled vector (same rules as src/forecast.ts)."""
    import torch
    n = h.shape[0]
    if pooling == "last":
        return h[-1]
    if pooling == "mean":
        return h.mean(0)
    if n < BINS:
        raise ValueError(f"bins32 needs at least {BINS} tokens, got {n}")
    return torch.cat([h[(b * n) // BINS:((b + 1) * n) // BINS].mean(0) for b in range(BINS)])


def fill_turn(turn, prompt, system="", think=""):
    out = turn
    for k, v in (("{prompt}", prompt), ("{system}", system), ("{think}", think)):
        out = out.replace(k, v)
    return out


def sha_of_weights(model_dir):
    """sha256 of model.safetensors, or of model.safetensors.index.json for a sharded checkpoint (provenance only)."""
    d = Path(model_dir)
    for n in ("model.safetensors", "model.safetensors.index.json"):
        if (d / n).exists():
            return sha256_file(d / n)
    raise SystemExit(f"{d}: no model.safetensors[.index.json]")


# ---------------------------------------------------------------- collect

def collect(model_dir, manifest, prompts, out, max_new_tokens=48, limit=0):
    """Greedy answers of the model to each prompt ({"id", "prompt"} per line) through the manifest's generation turn."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32, local_files_only=True).eval()
    eos = [t["id"] for t in manifest["eos"]]
    rows = read_jsonl(prompts)
    rows = rows[:limit] if limit else rows
    res = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            ids = tok(fill_turn(manifest["generation"]["turn"], r["prompt"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
            g = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos, pad_token_id=eos[0])
            res.append({"id": r["id"], "prompt": r["prompt"], "response": tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)})
            if (i + 1) % 50 == 0:
                log("forecast", f"collected {i + 1}/{len(rows)}")
    write_jsonl(out, res)
    log("forecast", f"{len(res)} answers -> {out}")
    return res


def join_labels(rows, labels):
    """rows: dicts with `id`; labels: jsonl rows {"id", "labels": {name: 0|1}}. -> (kept rows, behaviors, y [n, k] list, dropped ids)."""
    by = {}
    for l in labels:
        if l["id"] in by:
            raise SystemExit(f"labels: duplicate id {l['id']!r}")
        by[l["id"]] = l["labels"]
    names = []
    for l in by.values():
        for k in l:
            if k not in names:
                names.append(k)
    if not names:
        raise SystemExit("labels: no behavior names")
    kept, y, dropped = [], [], []
    for r in rows:
        l = by.get(r["id"])
        if l is None or any(k not in l for k in names):
            dropped.append(r["id"])
            continue
        bad = [k for k in names if l[k] not in (0, 1, True, False)]
        if bad:
            raise SystemExit(f"labels: {r['id']!r}: {bad} must be 0 or 1")
        kept.append(r)
        y.append([float(l[k]) for k in names])
    return kept, names, y, dropped


# ---------------------------------------------------------------- features from the model

def features(model_dir, manifest_path, rows, question, layer, pooling, work, lib=None, tokenizer_dir=None):
    """Pooled layer-L activation at the decide prompt of each row, in the layout `s1s2 encode` gives the lib."""
    import torch
    from transformers import AutoModelForCausalLM
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    write_jsonl(work / "encode_in.jsonl", [{"input": r["prompt"], "question": question} for r in rows])
    libd = lib_dir(lib)
    run(["node", libd / "bin" / "s1s2.mjs", "encode", manifest_path, "--tokenizer-dir", tokenizer_dir or model_dir,
         "--in", work / "encode_in.jsonl", "--out", work / "encoded.jsonl"], "forecast")
    enc = read_jsonl(work / "encoded.jsonl")
    bad = [i for i, e in enumerate(enc) if "error" in e or "question" not in e]
    if bad:
        raise SystemExit(f"encode failed for {len(bad)} rows (first: {enc[bad[0]].get('error')})")
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32, local_files_only=True).eval()
    check_layer(layer, model.config.num_hidden_layers)
    feats = []
    with torch.no_grad():
        for e in enc:
            q = e["question"]
            ids = torch.tensor([q["ids"][:q["last"] + 1]])  # context tokens 0..last (what the lib pools; a score lead is not part of it)
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states[layer + 1][0]
            feats.append(pool(hs.float(), pooling))
    return torch.stack(feats), [len(e["question"]["ids"]) for e in enc]


# ---------------------------------------------------------------- training

def standardize(Xtr):
    mean = Xtr.mean(0)
    std = Xtr.std(0, unbiased=False).clamp_min(1e-6)
    return mean, std


def fit_head(kind, X, y, l2=1e-2, hidden=64, epochs=300, lr=1e-2, seed=1):
    """kind: logistic | ridge | mlp. X standardized [n, d], y [n, k] in {0,1}. -> dict of fp32 tensors (names as in spec/FORECAST.md)."""
    import torch
    torch.manual_seed(seed)
    n, d = X.shape
    k = y.shape[1]
    if kind == "ridge":  # closed form on targets +-1: W = (X'X + l2 n I)^-1 X' t ; bias = mean of t
        t = y * 2 - 1
        tm = t.mean(0, keepdim=True)
        A = X.T @ X + l2 * n * torch.eye(d)
        W = torch.linalg.solve(A, X.T @ (t - tm))
        return {"weight": W.T.contiguous(), "bias": tm[0].clone()}
    if kind == "logistic":
        lin = torch.nn.Linear(d, k)
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        params, fwd = list(lin.parameters()), lin
    else:
        l1, l2m = torch.nn.Linear(d, hidden), torch.nn.Linear(hidden, k)
        params, fwd = [*l1.parameters(), *l2m.parameters()], (lambda x: l2m(torch.relu(l1(x))))
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(fwd(X), y) + l2 * sum((p ** 2).sum() for p in params if p.dim() > 1) / max(1, n)
        loss.backward()
        opt.step()
    if kind == "logistic":
        return {"weight": lin.weight.detach().clone(), "bias": lin.bias.detach().clone()}
    return {"w1": l1.weight.detach().clone(), "b1": l1.bias.detach().clone(), "w2": l2m.weight.detach().clone(), "b2": l2m.bias.detach().clone()}


def head_scores(t, X):
    import torch
    if "weight" in t:
        return X @ t["weight"].T + t["bias"]
    return torch.relu(X @ t["w1"].T + t["b1"]) @ t["w2"].T + t["b2"]


def bce(z, y):
    import torch
    return float(torch.nn.functional.binary_cross_entropy_with_logits(z, y))


def fit_temperature(z, y):
    """One temperature T > 0 for all behaviors minimising the BCE of sigmoid(z / T) (golden-section on log T in [-3, 3])."""
    import torch
    f = lambda s: bce(z / math.exp(s), y)  # noqa: E731
    a, b, g = -3.0, 3.0, (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(60):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return math.exp((a + b) / 2)


def fit_platt(z, y, steps=500):
    """Per-behavior a, b minimising the BCE of sigmoid(a z + b) (LBFGS on the held-out scores)."""
    import torch
    k = z.shape[1]
    a = torch.ones(k, requires_grad=True)
    b = torch.zeros(k, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * z + b, y)
        loss.backward()
        return loss
    opt.step(closure)
    return a.detach().tolist(), b.detach().tolist()


def write_head(path, tensors, mean, std):
    from safetensors.torch import save_file
    out = {"mean": mean.float().contiguous(), "std": std.float().contiguous()}
    out.update({k: v.float().contiguous() for k, v in tensors.items()})
    save_file(out, str(path))


def train(args):
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    rows = read_jsonl(args.responses)
    kept, names, y, dropped = join_labels(rows, read_jsonl(args.labels))
    if len(kept) < 20:
        raise SystemExit(f"only {len(kept)} rows have labels; need at least 20 (dropped {len(dropped)})")
    log("forecast", f"{len(kept)} labelled rows ({len(dropped)} without labels), behaviors {names}")
    question = read_json(args.question)
    X, lens = features(args.model, args.manifest, kept, question, args.layer, args.pooling, out / "work", args.lib, args.tokenizer_dir)
    Y = torch.tensor(y)
    idx = list(range(len(kept)))
    random.Random(args.seed).shuffle(idx)
    nh = max(5, int(len(idx) * args.heldout))
    ho, tr = idx[:nh], idx[nh:]
    mean, std = standardize(X[tr])
    Z = (X - mean) / std
    kind = args.head
    t = fit_head(kind, Z[tr], Y[tr], l2=args.l2, hidden=args.hidden, epochs=args.epochs, seed=args.seed)
    zho = head_scores(t, Z[ho])
    T = fit_temperature(zho, Y[ho]) if kind != "ridge" else 1.0
    platt = None
    if args.platt or kind == "ridge":  # ridge scores are not logits: calibrate them
        a, b = fit_platt(zho / T, Y[ho])
        platt = {"a": a, "b": b}
    write_head(out / "forecast.safetensors", t, mean, std)
    dims = {"in": int(X.shape[1]), "out": len(names)}
    if "w1" in t:
        dims["hidden"] = int(t["w1"].shape[0])
    block = {
        "id": args.id, "layer": args.layer, "pooling": args.pooling,
        "head": {"type": "mlp" if "w1" in t else "linear", "file": {"path": "forecast.safetensors", "sha256": sha256_file(out / "forecast.safetensors"),
                                                                      "size": (out / "forecast.safetensors").stat().st_size}, "dims": dims},
        "behaviors": [{"name": n} for n in names],
        "calibration": {"temperature": T, **({"platt": platt} if platt else {})},
        "trainedOn": {"baseId": manifest["id"], "weightsSha256": sha_of_weights(args.model),
                      **({"loraSha256": sha256_file(args.lora_file)} if args.lora_file else {})},
    }
    write_json(out / "forecast.manifest.json", {"forecast": block, "buildPatch": build_patch(args.layer)})
    write_json(out / "report.json", {
        "createdAt": now_iso(), "head": kind, "rows": len(kept), "train": len(tr), "heldout": len(ho), "promptTokens": {"min": min(lens), "max": max(lens)},
        "prevalence": {n: float(Y[:, i].mean()) for i, n in enumerate(names)},
        "heldoutLogLoss": bce(head_scores(t, Z[ho]) / T if not platt else torch.tensor(platt["a"]) * head_scores(t, Z[ho]) / T + torch.tensor(platt["b"]), Y[ho]),
        "note": "diagnostic of the fit on a small held-out split (also used to fit the calibration); not an accuracy result",
        "labelsDropped": len(dropped), "args": {k: v for k, v in vars(args).items() if k not in ("fn",)}})
    log("forecast", f"head {kind}: {out / 'forecast.safetensors'}; manifest block {out / 'forecast.manifest.json'}")
    return block


def main(argv=None):
    ap = argparse.ArgumentParser(prog="s1s2-convert train-forecast-head", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="greedy answers of the model to each prompt")
    c.add_argument("--model", required=True, help="model directory (the FUSED checkpoint when a LoRA is used)")
    c.add_argument("--manifest", required=True, help="manifest with eos and generation.turn")
    c.add_argument("--prompts", required=True, help='jsonl {"id", "prompt"}')
    c.add_argument("--out", required=True, help="responses jsonl")
    c.add_argument("--max-new-tokens", type=int, default=48)
    c.add_argument("--limit", type=int, default=0, help="first N prompts only (pilot: read the answers before running all)")
    t = sub.add_parser("train", help="features at the forecast layer + your labels -> head file and manifest block")
    t.add_argument("--model", required=True)
    t.add_argument("--manifest", required=True, help="manifest (encoding for `s1s2 encode`, id for trainedOn.baseId)")
    t.add_argument("--responses", required=True, help="output of collect")
    t.add_argument("--labels", required=True, help='jsonl {"id", "labels": {"<behavior>": 0|1}} supplied by you')
    t.add_argument("--question", required=True, help="json of the decide question the head will read after (as in decide({questions: [..]}))")
    t.add_argument("--layer", type=int, required=True)
    t.add_argument("--pooling", choices=POOLINGS, default="last")
    t.add_argument("--head", choices=("logistic", "ridge", "mlp"), default="logistic")
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--l2", type=float, default=1e-2)
    t.add_argument("--platt", action="store_true", help="also fit Platt a, b per behavior on the held-out split (always for ridge)")
    t.add_argument("--heldout", type=float, default=0.25)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--id", default="forecast-1")
    t.add_argument("--lora-file", help="adapter file whose sha256 goes to trainedOn.loraSha256 (provenance)")
    t.add_argument("--tokenizer-dir")
    t.add_argument("--lib", help="s1s2 checkout (default: the converter's parent directory)")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.model, read_json(a.manifest), a.prompts, a.out, a.max_new_tokens, a.limit)
    else:
        train(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
