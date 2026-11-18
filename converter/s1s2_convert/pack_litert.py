"""Repackage a published ONNX Runtime GenAI S1S2 repo as a LiteRT.js (int4) S1S2 repo.

The graph comes from litert-torch's Hugging Face exporter: int4 block-32 quantized fully connected layers, the token
embedder kept outside the graph (the runtime looks rows up itself, so the vocabulary x hidden table never enters the
graph or the Wasm heap), prefill signature of 16 tokens plus a decode signature, KV cache of 2,048 positions, logits
only. The weights are then moved out of the FlatBuffer into one blob that @litertjs/core streams into WebGPU buffers.
There is no hidden-state output, so the manifests read the label logits (no heads).

Stages:
  pack  fuse the LoRA into the base (or take --fused), export, rebundle, copy tokenizer/licences from the source repo,
        write one manifest per source manifest (logits readout)
  pin   write the files commit sha into the manifests and validate them with s1s2

Run it in an environment with litert-torch installed (versions pinned in converter/README.md).

Example:
  python -m s1s2_convert.pack_litert pack --repo you/Qwen3-0.6B-Task-ONNX-genai-int4-S1S2 --out /tmp/litert \\
      --lora you/Qwen3-0.6B-Task-LoRA --lora-revision <sha>
  python -m s1s2_convert.pack_litert pin --out /tmp/litert --revision <40-hex sha of the uploaded files commit>
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .fuse_lora import fuse_from_hub
from .pack_tjs import LAB_KEEP, PLACEHOLDER, pfile
from .util import lib_dir, log, now_iso, read_json, write_json

STAGE = "pack-litert"
BUILD = "litert-int4"
SUBDIR = "litert"
BUNDLE_FORMAT = "s1s2-litert-bundle/1"
RECIPE = Path(__file__).resolve().parent / "data" / "litert_int4_block32_fc.json"
PREFILL = 16
SOURCE_PATTERNS = ["manifest*.json", "tokenizer.json", "tokenizer_config.json", "tokenizer/*", "chat_template.jinja",
                   "LICENSE*", "NOTICE", "README.md"]


# ---------------------------------------------------------------- export + bundle

def export_graph(fused, raw_dir, cache_length):
    """litert-torch export of the fused model: logits only, external embedder, int4 block-32 fully connected."""
    from litert_torch.generative.export_hf import export as E
    E.export(model=str(fused), output_dir=str(raw_dir), keep_temporary_files=True, bundle_litert_lm=False,
             cache_length=cache_length, prefill_lengths=[PREFILL], externalize_embedder=True, single_token_embedder=True,
             quantization_recipe=str(RECIPE), apply_gpu_composites=False, externalize_weights=False, use_bool_mask=False,
             prefill_logits=False)
    tfl = Path(raw_dir) / "model_quantized.tflite"
    if not tfl.exists():
        raise SystemExit(f"{tfl} missing after the export")
    return tfl


def signatures(graph_bytes):
    from ai_edge_litert import schema_py_generated as sch
    m = sch.Model.GetRootAsModel(graph_bytes, 0)
    tnames = {v: k for k, v in vars(sch.TensorType).items() if not k.startswith("_")}
    sigs = {}
    for i in range(m.SignatureDefsLength()):
        sd = m.SignatureDefs(i)
        sg = m.Subgraphs(sd.SubgraphIndex())

        def io(n, get):
            r = []
            for j in range(n):
                t = get(j)
                tt = sg.Tensors(t.TensorIndex())
                r.append({"name": t.Name().decode(), "shape": tt.ShapeAsNumpy().tolist() if tt.ShapeLength() else [],
                          "dtype": tnames[tt.Type()].lower()})
            return r
        sigs[sd.SignatureKey().decode()] = {"inputs": io(sd.InputsLength(), sd.Inputs), "outputs": io(sd.OutputsLength(), sd.Outputs)}
    return sigs


def write_embeddings_f16(hf_dir, dst):
    """Token embedding rows as little-endian float16 [vocab, hidden], row-major (rows are read on demand by the runtime)."""
    import numpy as np
    from safetensors import safe_open
    files = sorted(Path(hf_dir).glob("*.safetensors"))
    for f in files:
        with safe_open(str(f), "pt") as sf:
            keys = [k for k in sf.keys() if k.endswith("embed_tokens.weight")]
            if not keys:
                continue
            sl = sf.get_slice(keys[0])
            V, H = sl.get_shape()
            with open(dst, "wb") as fh:
                for r0 in range(0, V, 16384):
                    fh.write(sl[r0:min(V, r0 + 16384)].float().numpy().astype(np.dtype("<f2")).tobytes())
            return V, H
    raise SystemExit(f"no embed_tokens.weight in {hf_dir}")


def bundle(tflite, hf_dir, out, threshold=256):
    """tflite -> out/{model.graph.tflite, model.weights.bin, embed.f16.bin, manifest.json}"""
    from .tools import externalize_tflite_flatbuffer as ext
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_ext"
    tmp.mkdir(exist_ok=True)
    ext.externalize(Path(tflite), tmp, group_name="tflite_weights", num_elements_threshold=threshold)
    graph = [p for p in sorted(tmp.iterdir()) if p.suffix == ".tflite"][0]
    graph.replace(out / "model.graph.tflite")
    (tmp / "tflite_weights").replace(out / "model.weights.bin")
    shutil.rmtree(tmp)
    sigs = signatures((out / "model.graph.tflite").read_bytes())
    V, H = write_embeddings_f16(hf_dir, out / "embed.f16.bin")
    cfg = read_json(Path(hf_dir) / "config.json")
    cfg = cfg.get("text_config", cfg)
    files = {}
    for p in sorted(out.iterdir()):
        if p.name == "manifest.json":
            continue
        files[p.name] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    man = {"format": BUNDLE_FORMAT, "vocab": V, "hidden": H, "signatures": sigs, "files": files, "externalized": True,
           "embedding": "f16", "pad_token_id": cfg.get("pad_token_id"), "eos_token_id": cfg.get("eos_token_id"),
           "source_tflite": Path(tflite).name}
    write_json(out / "manifest.json", man)
    return man


# ---------------------------------------------------------------- manifests

def new_manifest(src, name, suffix, files, sibling, versions, note, cache_length):
    m = json.loads(json.dumps(src))
    m["schema"] = "s1s2.manifest/1"
    short = name.split("/")[-1].lower()
    m["id"] = f"{short}.{suffix}" if suffix else short
    m["tokenizer"]["repo"] = name
    m["tokenizer"]["revision"] = PLACEHOLDER
    # logits only: heads read the hidden state, which this graph does not output
    m.setdefault("defaults", {}).setdefault("method", {}).update({"predicate": "logits", "choice": "logits", "score": "logits"})
    m["builds"] = {BUILD: {"runtime": "litert", "repo": name, "revision": PLACEHOLDER, "files": files, "hiddenOutput": None,
                           "devices": ["webgpu"], "dtype": "int4", "maxSafeTokens": cache_length - 8, "fidelity": None}}
    lab = src.get("lab") or {}
    keep = {k: lab[k] for k in LAB_KEEP if k in lab}
    if keep:
        m["lab"] = {"note": note, **keep}
    else:
        m.pop("lab", None)
    m["provenance"] = {
        "generator": {"name": "s1s2-convert pack_litert", "version": __version__}, "createdAt": now_iso(),
        "base": dict(src["base"]),
        "source": {"format": "onnxruntime-genai", "repo": sibling["repo"], "revision": sibling["revision"]},
        "builds": {BUILD: {"tool": "s1s2-convert pack_litert", "version": __version__, "tools": versions,
                           "args": ["fuse LoRA into base (bf16)",
                                    f"litert-torch export: int4 block-32 fully connected, external embedder, prefill {PREFILL}, cache {cache_length}, logits only",
                                    "externalize weights into model.weights.bin", "embed.f16.bin: token rows, float16"]}},
        "notes": "Same fine-tuned weights as the source GenAI build, quantized again to 4 bits; a different quantization, not a copy of the source graph.",
    }
    return m


def tool_versions():
    import importlib.metadata as md
    names = ["litert-torch", "ai-edge-litert-nightly", "litert-converter", "litert-quantizer-nightly", "torch", "transformers", "numpy"]
    r = {}
    for n in names:
        try:
            r[n] = md.version(n)
        except md.PackageNotFoundError:
            pass
    return r


# ---------------------------------------------------------------- stages

def stage_pack(a):
    from huggingface_hub import HfApi, snapshot_download
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.force:
        raise SystemExit(f"{out} is not empty (pass --force)")
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    if Path(a.repo).is_dir():
        src, rev = Path(a.repo), a.revision or "local"
    else:
        rev = a.revision or api.model_info(a.repo).sha
        src = Path(snapshot_download(a.repo, revision=rev, allow_patterns=SOURCE_PATTERNS))
    man_paths = sorted(src.glob("manifest*.json"))
    if not man_paths:
        raise SystemExit(f"{a.repo}: no manifest*.json at the root")
    base = read_json(src / "manifest.json" if (src / "manifest.json").exists() else man_paths[0])["base"]
    work = Path(a.work) if a.work else out.parent / f".{out.name}.work"
    work.mkdir(parents=True, exist_ok=True)
    fused = Path(a.fused) if a.fused else work / "fused"
    if not a.fused and not (fused / "FUSE.json").exists():
        if not (a.lora and a.lora_revision):
            raise SystemExit("pass --lora REPO --lora-revision SHA (or --fused DIR with the merged model)")
        fuse_from_hub(base["repo"], base["revision"], a.lora, a.lora_revision, fused, STAGE)
    name = a.name or (a.repo.replace("-ONNX-genai-int4-S1S2", "-LiteRT-S1S2") if not Path(a.repo).is_dir() else None)
    if not name:
        raise SystemExit("--name NEW_REPO_ID is required for a local --repo")
    raw = work / "raw"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)
    tfl = export_graph(fused, raw, a.cache_length)
    dst = out / SUBDIR
    shutil.rmtree(dst, ignore_errors=True)
    bundle(tfl, fused, dst)
    shutil.rmtree(raw, ignore_errors=True)
    for p in sorted(src.glob("LICENSE*")) + sorted((src / "tokenizer").glob("*")):
        rel = p.relative_to(src)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out / rel)
    if (src / "NOTICE").exists():
        shutil.copyfile(src / "NOTICE", out / "NOTICE")
    with open(out / "NOTICE", "a") as f:
        f.write(f"\nLiteRT.js build of {a.repo} (revision {rev}): the fused weights exported with litert-torch (Apache-2.0). "
                "litert/model.graph.tflite and the weight blob were produced with LiteRT's externalize_tflite_flatbuffer tool "
                "(Apache-2.0, Copyright 2025 Google LLC).\n")
    files = {"manifest": pfile(out, f"{SUBDIR}/manifest.json"), "graph": pfile(out, f"{SUBDIR}/model.graph.tflite"),
             "weights": pfile(out, f"{SUBDIR}/model.weights.bin"), "embed": pfile(out, f"{SUBDIR}/embed.f16.bin")}
    note = f"Alternative LiteRT.js (int4) build of {a.repo}; prompts and option words are the same. Readout: label logits (no heads)."
    sibling = {"repo": a.repo, "revision": rev}
    versions = tool_versions()
    written = []
    for mp in man_paths:
        sm = read_json(mp)
        suffix = mp.name[len("manifest."):-len(".json")] if mp.name != "manifest.json" else (sm.get("lab", {}).get("task") or sm.get("lab", {}).get("language") or "")
        write_json(out / mp.name, new_manifest(sm, name, suffix, files, sibling, versions, note, a.cache_length))
        written.append(mp.name)
    write_json(out / ".pack-litert.json", {"source": sibling, "name": name, "manifests": written, "createdAt": now_iso()})
    log(STAGE, f"manifests: {written}; files: {sorted(files)}")


def stage_pin(a):
    out = Path(a.out)
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision or ""):
        raise SystemExit("--revision: 40-hex commit sha of the files commit")
    meta = read_json(out / ".pack-litert.json")
    for mn in meta["manifests"]:
        m = read_json(out / mn)
        m["tokenizer"]["revision"] = a.revision
        m["builds"][BUILD]["revision"] = a.revision
        write_json(out / mn, m)
    lib = lib_dir(a.lib)
    subprocess.run(["node", str(lib / "bin" / "s1s2.mjs"), "validate", "--strict", *[str(out / mn) for mn in meta["manifests"]]], check=True)
    log(STAGE, f"pinned {meta['manifests']} to {a.revision}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="s1s2-convert pack-litert", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["pack", "pin"])
    ap.add_argument("--out", required=True, help="output folder (the new repo's contents)")
    ap.add_argument("--repo", help="pack: source GenAI S1S2 repo id (or a local folder with the same files)")
    ap.add_argument("--revision", help="pack: source revision (default: current main, recorded); pin: files commit sha")
    ap.add_argument("--name", help="pack: new repo id (default: <source> with -LiteRT-S1S2)")
    ap.add_argument("--lora", help="pack: the LoRA repo of the specialist")
    ap.add_argument("--lora-revision", help="pack: pinned LoRA revision")
    ap.add_argument("--fused", help="pack: folder with the already merged HF model (skips the fuse)")
    ap.add_argument("--work", help="pack: scratch folder (default: next to --out)")
    ap.add_argument("--cache-length", type=int, default=2048, help="KV cache positions (the longest prompt is this minus 8)")
    ap.add_argument("--lib", help="s1s2 checkout (default: the converter's parent directory)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    {"pack": stage_pack, "pin": stage_pin}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
