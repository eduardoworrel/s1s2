"""Training: LoRA + decision head jointly, then the evidence head.

Decision recipe (the trainer behind the GDPR/LGPD/CCPA specialists in the model cards, which follows Kev @2074cc44;
reimplemented here, same numbers):
  LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down of the decoder layers (+ in_proj_qkv/z/a/b, out_proj on a
  hybrid Gated-DeltaNet backbone) (the trainer's LoRA targets); pointer head dp=256 from scratch (head.py); loss = cross
  entropy of the head over the options, mean per row (the trainer's per-row loss); AdamW, weight decay 0.01, OneCycleLR
  pct_start 0.1 with the same lr for the head, grad-norm clip 1.0, effective batch 8 = 4 x 2 accumulation, rows of one
  step grouped in passes by length; bf16 autocast + gradient checkpointing on CUDA (fp32 on CPU); --head-lr (kev's
  head_lr option) gives the head its own peak lr, default the same; selection: the checkpoint with the lowest head loss
  on the dev split (evaluated every --eval-every steps and at the end of each epoch). Option order is re-drawn every
  epoch (seeded).
  Epochs, lr and fusion dtype come from the recipe (RECIPES):
    kev (default): the Kev-faithful recipe: 2 epochs; lr by Kev's rule, 1e-4 if the checkpoint has
      <= 1B parameters, 5e-5 above (LoRA and head); LoRA fused in fp32. On the three-law privacy benchmark (held-out
      test, scored once) it was more accurate than `short`; these are the variants called joint_k in the model cards.
    short: the shorter recipe fixed before any result: 1 epoch, lr 5e-5, fusion in the base
      dtype.
  Not reproduced (declared): kev data augmentation (none-of-the-above / distractor options), later kev stages,
  temperature calibration, resumable checkpoints.

Evidence recipe (the citation-head trainer): head only, backbone frozen (the fused decision model),
  initialised from the decision head; AdamW lr 1e-4, weight decay 0.01, batch 32 items, up to 8 epochs, clip 1.0,
  seed 1, fp32; loss per item = -log sum_{k in targets} p_k; epoch chosen by the dev loss (epoch 0 = the init head).
"""
import math
import random
import re
import time
from pathlib import Path

from . import head as H
from .util import log, write_json, read_jsonl

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_TARGETS_HYBRID = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


KEV_LR_LIMIT = 1e9  # Kev's rule: <= 1B parameters -> 1e-4; above -> 5e-5
KEV_LR = {"up_to_1B": 1e-4, "above": 5e-5}
KEV_LR_SOURCE = "kev @2074cc44 AGENTS.md:19-20 and docs/model-cards/kev-4b.md:149 (0.8B 1e-4; 4B/9B 5e-5); 1B limit: our interpolation (Kev has no model between 0.8B and 4B)"
RECIPES = {
    "kev": {"name": "Kev-faithful recipe: 2 epochs, Kev's learning-rate rule, fp32 fusion",
            "epochs": 2, "lr": "kev-rule", "fuse_fp32": True},
    "short": {"name": "short recipe: 1 epoch, lr 5e-5, base-dtype fusion", "epochs": 1, "lr": 5e-5, "fuse_fp32": False},
}
DEFAULT_RECIPE = "kev"


def recipe_id(name):
    """Recipe name, checked against RECIPES."""
    if name not in RECIPES:
        raise SystemExit(f"unknown recipe {name!r} (known: {', '.join(sorted(RECIPES))})")
    return name


def checkpoint_params(model_dir):
    """Parameters stored in the checkpoint's safetensors (headers only; a tied embedding counts once)."""
    from safetensors import safe_open
    n = 0
    for f_name in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f_name), "pt") as f:
            for k in f.keys():
                n += math.prod(f.get_slice(k).get_shape())
    if not n:
        raise SystemExit(f"{model_dir}: no safetensors to count parameters")
    return n


def kev_lr(n_params):
    return KEV_LR["up_to_1B"] if n_params <= KEV_LR_LIMIT else KEV_LR["above"]


def recipe_lr(recipe, model_dir, lr=None):
    """-> (lr, rule): the explicit lr if given, else the recipe's (the Kev rule counts the checkpoint's parameters)."""
    r = RECIPES[recipe_id(recipe)]
    if lr is not None:
        return lr, {"explicit": True}
    if r["lr"] != "kev-rule":
        return r["lr"], {"fixed": r["lr"]}
    n = checkpoint_params(model_dir)
    return kev_lr(n), {"kev_rule": {"checkpoint_params": n, "limit": KEV_LR_LIMIT, "lrs": KEV_LR, "source": KEV_LR_SOURCE}}


def lora_regex(hybrid):
    names = LORA_TARGETS + (LORA_TARGETS_HYBRID if hybrid else [])
    return r"^(?!.*(visual|vision|mtp)).*\.layers\.\d+\..*\.(" + "|".join(names) + r")$"


def load_model(model_dir, device):
    import torch
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype, local_files_only=True)
    return m.to(device)


def passes(rows, batch):
    """Rows of one optimizer step in passes of up to `batch`, longest first (as the original trainer, no token cap)."""
    order = sorted(rows, key=lambda r: -len(r["ids"]))
    return [order[i:i + batch] for i in range(0, len(order), batch)]


def tensors(rows, pad_id, device):
    import torch
    L = max(len(r["ids"]) for r in rows)
    t = torch.full((len(rows), L), pad_id, dtype=torch.long)
    a = torch.zeros((len(rows), L), dtype=torch.long)
    for i, r in enumerate(rows):  # right padding: real positions unchanged (causal)
        t[i, :len(r["ids"])] = torch.tensor(r["ids"])
        a[i, :len(r["ids"])] = 1
    return t.to(device), a.to(device)


def train_decision(model_dir, train_rows, dev_rows, out, *, device="cpu", epochs=1, lr=5e-5, head_lr=None, lora_r=16, batch=4, accum=2,
                   eval_every=0, seed=1, max_steps=0, recipe=DEFAULT_RECIPE, lr_rule=None):
    """train_rows: [{ids, decide, options, target, epoch}] (one entry per row per epoch); dev_rows: [{ids, decide, options, target}]."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    model = load_model(model_dir, device)
    dec = model.get_decoder()
    cfg = dec.config
    hybrid = "linear_attention" in set(getattr(cfg, "layer_types", None) or [])
    for p in model.parameters():
        p.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=lora_regex(hybrid), bias="none"))
    head = H.new_head(cfg.hidden_size).to(device)
    lora = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not lora:
        raise SystemExit("no LoRA parameters were created (target modules?)")
    by_epoch = {}
    for r in train_rows:
        by_epoch.setdefault(r["epoch"], []).append(r)
    rng = random.Random(seed)
    steps_plan = []
    for e in range(epochs):
        rows = list(by_epoch[e])
        rng.shuffle(rows)
        per = batch * accum
        steps_plan += [(e, rows[i:i + per]) for i in range(0, len(rows), per)]
    if max_steps:
        steps_plan = steps_plan[:max_steps]
    total = len(steps_plan)
    head_lr = head_lr or lr  # kev train.py:588 max_lr=[lr, head_lr or lr]; the lab recipe uses the same lr
    groups = [{"params": [p for _, p in lora], "lr": lr}, {"params": list(head.parameters()), "lr": head_lr}]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, head_lr], total_steps=max(total, 1), pct_start=0.1)
    cuda = device == "cuda"
    if cuda:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if cuda else (lambda: torch.autocast("cpu", enabled=False))
    n_lora = sum(p.numel() for _, p in lora)
    log("train", f"LoRA on {len({n.split('.lora_')[0] for n, _ in lora})} modules ({n_lora / 1e6:.2f}M params), head "
        f"{sum(p.numel() for p in head.parameters()) / 1e6:.2f}M; {len(train_rows)} train encodings, {len(dev_rows)} dev, {total} steps")

    @torch.no_grad()
    def evaluate():
        model.eval(); head.eval()
        ce, acc = 0.0, 0
        for g in passes(dev_rows, batch * 2):
            ids, att = tensors(g, pad_id, device)
            with autocast():
                h = dec(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state.float()
            for b, r in enumerate(g):
                z = H.scores(head, h[b], r["decide"], r["options"])
                ce += float(F.cross_entropy(z[None], torch.tensor([r["target"]], device=z.device)))
                acc += int(torch.argmax(z)) == r["target"]
        model.train(); head.train()
        return {"dev_loss": ce / len(dev_rows), "dev_acc_head": acc / len(dev_rows)}

    def snapshot():
        return ({n: p.detach().float().cpu().contiguous() for n, p in lora},
                {k: v.detach().float().cpu().clone() for k, v in head.state_dict().items()})

    history, best = [], None
    m0 = evaluate()
    history.append({"step": 0, **m0})
    log("train", f"step 0: dev loss {m0['dev_loss']:.4f} acc {m0['dev_acc_head']:.3f}")
    model.train(); head.train()
    params = [p for _, p in lora] + list(head.parameters())
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    for step, (e, rows) in enumerate(steps_plan, 1):
        for g in passes(rows, batch):
            ids, att = tensors(g, pad_id, device)
            with autocast():
                h = dec(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state.float()
            loss = sum(F.cross_entropy(H.scores(head, h[b], r["decide"], r["options"])[None],
                                       torch.tensor([r["target"]], device=h.device)) for b, r in enumerate(g)) / len(rows)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}")
            loss.backward()
            run_loss += float(loss.detach()) * len(rows)
        run_n += len(rows)
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        end_epoch = step == total or steps_plan[step][0] != e
        if end_epoch or (eval_every and step % eval_every == 0):
            m = {"train_loss": run_loss / run_n, **evaluate()}
            run_loss, run_n = 0.0, 0
            history.append({"step": step, "epoch": e, "grad_norm": round(norm, 4), **m})
            log("train", f"step {step}/{total} (epoch {e}): train loss {m['train_loss']:.4f}, dev loss {m['dev_loss']:.4f} acc {m['dev_acc_head']:.3f} ({time.time() - t0:.0f}s)")
            if best is None or m["dev_loss"] < best[0]["dev_loss"]:
                best = ({"step": step, **m}, snapshot())
    sel, (lora_sd, head_sd) = best
    save_file(lora_sd, str(out / "adapter.safetensors"))
    head.load_state_dict(head_sd)
    contract = {"kind": "decision", "read": {"query": "decide marker", "keys": "option markers"}, "selected": sel}
    c = H.save(head, out / "head", contract)
    recipe = recipe_id(recipe)
    info = {"recipe": f"{RECIPES[recipe]['name']}: joint LoRA + pointer head (Kev @2074cc44)", "recipe_id": recipe,
            "hyperparameters": {"lora_r": lora_r, "lora_alpha": 2 * lora_r, "lora_dropout": 0.05, "lora_targets": lora_regex(hybrid),
                                "head_dim": H.HEAD_DIM, "lr": lr, "head_lr": head_lr, "weight_decay": 0.01, "schedule": "OneCycleLR pct_start 0.1",
                                "grad_clip": 1.0, "batch": batch, "accum": accum, "epochs": epochs, "steps": total, "seed": seed, "lr_rule": lr_rule,
                                "precision": "bf16 autocast + gradient checkpointing" if cuda else "fp32 (CPU)", "device": device},
            "selection": "lowest dev head loss", "selected": sel, "history": history, "lora_scale": 2.0,
            "head": c["files"], "seconds": round(time.time() - t0, 1)}
    write_json(out / "train.json", info)
    return info


def hidden_states(model_dir, rows, device="cpu", batch=8):
    """Final post-norm hidden states at [decide, *options] for each row (fp32, CPU tensors)."""
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    model = load_model(model_dir, device).eval()
    dec = model.get_decoder()
    out = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    with torch.no_grad():
        for i0 in range(0, len(order), batch):
            idx = order[i0:i0 + batch]
            ids, att = tensors([rows[i] for i in idx], pad_id, device)
            h = dec(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state.float()
            for b, i in enumerate(idx):
                r = rows[i]
                out[i] = h[b, [r["decide"], *r["options"]]].cpu()
    return out


def train_evidence(fused_dir, init_head_dir, train_rows, dev_rows, out, *, device="cpu", lr=1e-4, batch=32, epochs=8, seed=1):
    """rows: [{ids, decide, options, targets}] (targets = evidence sentence indexes)."""
    import torch
    out = Path(out)
    torch.manual_seed(seed)
    tr = [r for r in train_rows if r["targets"]]
    dv = [r for r in dev_rows if r["targets"]]
    if not tr or not dv:
        raise SystemExit("evidence: need train and dev rows with evidence targets")
    log("evidence", f"hidden states for {len(tr)} train + {len(dv)} dev evidence prompts")
    htr, hdv = hidden_states(fused_dir, tr, device), hidden_states(fused_dir, dv, device)
    head, _ = H.load(init_head_dir)
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    def item_loss(hh, r):
        z = head(hh[0], hh[1:])
        lp = torch.log_softmax(z, -1)
        return -torch.logsumexp(lp[torch.tensor(r["targets"])], 0), int(torch.argmax(z)) in r["targets"]

    @torch.no_grad()
    def dev_loss():
        head.eval()
        res = [item_loss(h, r) for h, r in zip(hdv, dv)]
        head.train()
        return {"dev_loss": float(sum(l for l, _ in res) / len(res)), "dev_acc": sum(a for _, a in res) / len(res)}

    rng = random.Random(seed)
    hist = [{"epoch": 0, **dev_loss()}]
    best = (hist[0], {k: v.clone() for k, v in head.state_dict().items()})
    for e in range(1, epochs + 1):
        idx = list(range(len(tr)))
        rng.shuffle(idx)
        for i0 in range(0, len(idx), batch):
            b = idx[i0:i0 + batch]
            loss = sum(item_loss(htr[i], tr[i])[0] for i in b) / len(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        m = {"epoch": e, **dev_loss()}
        hist.append(m)
        log("evidence", f"epoch {e}: dev loss {m['dev_loss']:.4f} acc {m['dev_acc']:.3f}")
        if m["dev_loss"] < best[0]["dev_loss"]:
            best = (m, {k: v.clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1])
    c = H.save(head, out / "head", {"kind": "evidence", "read": {"query": "decide marker", "keys": "one option marker per sentence"},
                                    "selected": best[0], "init": str(init_head_dir)})
    info = {"recipe": "evidence head: head only on the fused model, init from the decision head",
            "hyperparameters": {"lr": lr, "weight_decay": 0.01, "batch": batch, "epochs": epochs, "grad_clip": 1.0, "seed": seed, "precision": "fp32"},
            "selection": "lowest dev loss (epoch 0 = init head)", "selected": best[0], "history": hist, "head": c["files"],
            "rows": {"train": len(tr), "dev": len(dv)}}
    write_json(out / "evidence.json", info)
    return info


def fuse(base_dir, adapter, scale, dest, fp32=False):
    """W' = W + scale * B @ A on the SAME keys and files of the base checkpoint; other files copied byte for byte.
    Same as the original trainer's merge step.
    fp32=False: the sum is rounded to the weight's original dtype (the trainer's default path).
    fp32=True (the Kev-faithful recipe): W' = fp32(float64(W) + float64(scale) * sum_k B[:, k] A[k, :]), the outer products
    summed in a fixed order (elementwise only: the same result on any machine) and rounded once to fp32; EVERY floating
    tensor written in fp32 (the unfused ones losslessly); config.json dtype float32 (the GenAI builder loads with dtype
    "auto"); the index total_size recomputed."""
    import json
    import os
    import shutil
    import hashlib
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    base_dir, dest = Path(base_dir), Path(dest)
    sd = load_file(str(adapter))
    pairs = {}
    for n, t in sd.items():
        m = re.match(r"^base_model\.model\.(.+)\.lora_([AB])\.default\.weight$", n)
        if not m:
            raise SystemExit(f"unexpected LoRA key: {n}")
        pairs.setdefault(m.group(1), {})[m.group(2)] = t
    idx = base_dir / "model.safetensors.index.json"
    if idx.exists():
        wmap = json.loads(idx.read_text())["weight_map"]
    else:
        with safe_open(str(base_dir / "model.safetensors"), "pt") as f:
            wmap = {k: "model.safetensors" for k in f.keys()}
    delta = {}
    for mod, ab in pairs.items():
        if not mod.startswith("model."):
            raise SystemExit(f"LoRA module outside the decoder: {mod}")
        rel = mod[len("model."):]
        cand = [k for k in (f"model.{rel}.weight", f"model.language_model.{rel}.weight") if k in wmap]
        if len(cand) != 1:
            raise SystemExit(f"{mod}: {len(cand)} candidate keys in the checkpoint ({cand})")
        delta[cand[0]] = ab if fp32 else (ab["B"].float() @ ab["A"].float()) * scale
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    used, norms = set(), []
    for f_name in sorted(set(wmap.values())):
        outd = {}
        with safe_open(str(base_dir / f_name), "pt") as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                if k in delta and fp32:
                    d = _delta_f64(delta[k], scale)
                    if tuple(d.shape) != tuple(t.shape):
                        raise SystemExit(f"{k}: delta {tuple(d.shape)} != weight {tuple(t.shape)}")
                    norms.append(float(d.norm() / t.double().norm()))
                    t = (t.double() + d).to(torch.float32)
                    del d
                    used.add(k)
                elif k in delta:
                    d = delta[k]
                    if tuple(d.shape) != tuple(t.shape):
                        raise SystemExit(f"{k}: delta {tuple(d.shape)} != weight {tuple(t.shape)}")
                    norms.append(float(d.norm() / t.float().norm()))
                    t = (t.float() + d).to(t.dtype)
                    used.add(k)
                elif fp32 and t.is_floating_point():
                    t = t.to(torch.float32)
                outd[k] = t.contiguous()
        save_file(outd, str(tmp / f_name), metadata=meta)
    if used != set(delta):
        raise SystemExit(f"deltas not applied: {sorted(set(delta) - used)[:5]}")
    for p in sorted(base_dir.iterdir()):
        if p.is_file() and not p.name.endswith(".safetensors") and p.name != "PROVENANCE.json":
            shutil.copyfile(p, tmp / p.name)
    if fp32:
        _set_fp32(tmp)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    info = {"modules_fused": len(delta), "scale": scale, "files": sorted(set(wmap.values())),
            "relative_delta_norm": {"mean": sum(norms) / len(norms), "max": max(norms)},
            "adapter_sha256": hashlib.sha256(Path(adapter).read_bytes()).hexdigest()}
    info.update({"dtype": "float32", "rule": "fp32(float64(W) + float64(scale) * sum_k B[:,k] A[k,:])"}
                if fp32 else {"dtype": "base"})
    return info


def _delta_f64(ab, scale):
    """scale * B @ A in float64 as outer products summed in the order k = 0..r-1 (deterministic across machines)."""
    import torch
    B, A = ab["B"].to(torch.float64), ab["A"].to(torch.float64)
    d = torch.zeros((B.shape[0], A.shape[1]), dtype=torch.float64)
    for k in range(B.shape[1]):
        d += B[:, k:k + 1] * A[k:k + 1, :]
    return d * float(scale)


DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def _set_fp32(folder):
    """After an fp32 fusion: dtype float32 in config.json (and its text_config); index total_size from the real bytes."""
    import json
    from safetensors import safe_open
    folder = Path(folder)
    cfg_f = folder / "config.json"
    if cfg_f.exists():
        cfg = json.loads(cfg_f.read_text())
        for c in (cfg, cfg.get("text_config") or {}):
            for k in ("torch_dtype", "dtype"):
                if k in c:
                    c[k] = "float32"
        if "torch_dtype" not in cfg and "dtype" not in cfg:
            cfg["dtype"] = "float32"
        cfg_f.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    idx = folder / "model.safetensors.index.json"
    if idx.exists():
        j = json.loads(idx.read_text())
        total = 0
        for f_name in sorted(set(j["weight_map"].values())):
            with safe_open(str(folder / f_name), "pt") as f:
                for k in f.keys():
                    sl = f.get_slice(k)
                    total += DTYPE_BYTES[sl.get_dtype()] * math.prod(sl.get_shape())
        j.setdefault("metadata", {})["total_size"] = total
        idx.write_text(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
