"""Merge a published LoRA adapter into its base model (the model both int4 formats approximate).

W' = (W.float() + scale * B.float() @ A.float()).to(W.dtype) on the same keys and files as the base; every other file of
the base snapshot (config, tokenizer, chat template) is copied unchanged. scale = lora_alpha / r. Used by pack_mlc and
pack_litert; it needs torch, safetensors and huggingface_hub only (no peft).
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

from .util import log

LORA_KEY = re.compile(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$")


def target_keys(pairs, weight_names):
    """Map each adapter module ('model.layers.0.self_attn.q_proj') to its weight key in the base checkpoint. The key is
    `model.<rest>.weight`, or `model.language_model.<rest>.weight` for multimodal wrappers; exactly one must exist."""
    target = {}
    for mod, ab in pairs.items():
        rel = mod[len("model."):] if mod.startswith("model.") else mod
        cand = [k for k in (f"model.{rel}.weight", f"model.language_model.{rel}.weight") if k in weight_names]
        if len(cand) != 1:
            raise SystemExit(f"adapter module {mod}: {len(cand)} matching base weights {cand}")
        target[cand[0]] = ab
    return target


def read_adapter(path):
    from safetensors.torch import load_file
    pairs = {}
    for n, t in load_file(str(path)).items():
        m = LORA_KEY.match(n)
        if not m:
            raise SystemExit("unexpected adapter key " + n)
        pairs.setdefault(m.group(1), {})[m.group(2)] = t
    return pairs


def fuse(base_dir, adapter_path, scale, out, base_label="", lora_label=""):
    """base_dir: local snapshot of the base; adapter_path: adapter_model.safetensors. Writes the merged model to `out`."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    base = Path(base_dir)
    pairs = read_adapter(adapter_path)
    idx = base / "model.safetensors.index.json"
    if idx.exists():
        wmap = json.loads(idx.read_text())["weight_map"]
    else:
        with safe_open(str(base / "model.safetensors"), "pt") as f:
            wmap = {k: "model.safetensors" for k in f.keys()}
    target = target_keys(pairs, set(wmap))
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    used = set()
    for name in sorted(set(wmap.values())):
        tensors = {}
        with safe_open(str(base / name), "pt") as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                if k in target:
                    delta = (target[k]["B"].float() @ target[k]["A"].float()) * scale
                    if delta.shape != t.shape:
                        raise SystemExit(f"{k}: delta {tuple(delta.shape)} != weight {tuple(t.shape)}")
                    t = (t.float() + delta).to(t.dtype)
                    used.add(k)
                tensors[k] = t.contiguous()
        save_file(tensors, str(out / name), metadata=meta)
    if used != set(target):
        raise SystemExit("adapter deltas not applied: " + str(sorted(set(target) - used))[:200])
    for p in base.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors"):
            shutil.copyfile(p, out / p.name)
    info = {"base": base_label, "lora": lora_label, "scale": scale, "modules": len(target),
            "adapterSha256": hashlib.sha256(Path(adapter_path).read_bytes()).hexdigest()}
    (out / "FUSE.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def fuse_from_hub(base_repo, base_rev, lora_repo, lora_rev, out, stage="fuse"):
    """Download both at the pinned revisions and merge into `out`."""
    from huggingface_hub import hf_hub_download, snapshot_download
    base = snapshot_download(base_repo, revision=base_rev, allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"])
    cfg = json.loads(Path(hf_hub_download(lora_repo, "adapter_config.json", revision=lora_rev)).read_text())
    ad = hf_hub_download(lora_repo, "adapter_model.safetensors", revision=lora_rev)
    scale = cfg["lora_alpha"] / cfg["r"]
    log(stage, f"fusing {lora_repo}@{lora_rev[:8]} into {base_repo}@{base_rev[:8]} (scale {scale})")
    return fuse(base, ad, scale, out, f"{base_repo}@{base_rev}", f"{lora_repo}@{lora_rev}")
