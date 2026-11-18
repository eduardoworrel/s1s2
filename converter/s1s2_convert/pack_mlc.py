"""Repackage a published ONNX Runtime GenAI S1S2 repo as a WebLLM (MLC, q4f16_1) S1S2 repo.

WebLLM runs a model through a compiled WebGPU library (`model-lib.wasm`) and returns only logits. s1s2 reads the heads'
hidden state from those logits (src/backends/webllm.ts): for a tied or untied lm_head W, logits = W h, so h is the
least-squares solution over N vocabulary rows, h = P . logits[rows] with P = (W_n^T W_n)^-1 W_n^T. This tool writes the
rows and P next to the weights (`recoveryRows`, `recoveryP` in the manifest), so the same heads as the GenAI build work.
Without the recovery files (`--readout logits`) the manifests read the label logits instead and carry no heads.

Stages:
  pack     fuse the LoRA into the base (or take --fused), `mlc_llm convert_weight --quantization q4f16_1`, `gen_config`,
           put the prebuilt/compiled library next to the weights, compute the recovery rows + P from the quantized
           lm_head, copy tokenizer/heads/licences from the source repo and write one manifest per source manifest
  pin      write the files commit sha into the manifests and validate them with s1s2
  compile  `mlc_llm compile --device webgpu` for a model the prebuilt libraries do not cover (needs emscripten and
           the MLC web runtime; see README)

Example:
  python -m s1s2_convert.pack_mlc pack --repo you/Qwen3-0.6B-Task-ONNX-genai-int4-S1S2 --out /tmp/mlc \\
      --lora you/Qwen3-0.6B-Task-LoRA --lora-revision <sha> \\
      --prebuilt-lib v0_2_86/base/Qwen3-0.6B-q4f16_1_cs1k-webgpu.wasm --conv-template qwen2
  # upload /tmp/mlc to a new Hub repo, then:
  python -m s1s2_convert.pack_mlc pin --out /tmp/mlc --revision <40-hex sha of that commit>
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import __version__
from .fuse_lora import fuse_from_hub
from .pack_tjs import LAB_KEEP, PLACEHOLDER, link_or_copy, pfile
from .util import log, now_iso, read_json, sha256_file, write_json, lib_dir

STAGE = "pack-mlc"
BUILD = "webllm-q4f16_1"
QUANT = "q4f16_1"
GROUP = 32            # q4f16_1: 4-bit symmetric, group size 32, 8 nibbles per uint32, value = (nibble - 7) * scale
ZERO = 7
HIDDEN_NAME = "recovered-from-logits"
# Prebuilt WebLLM model libraries (Apache-2.0). Each library is compiled for one architecture and shape; weights are
# data, so a fine-tune of the same base uses the same library. Pinned commit of mlc-ai/binary-mlc-llm-libs.
PREBUILT_COMMIT = "5b7251f3bdce957be2a7bf73a6f6c560d9874df4"
PREBUILT_BASE = f"https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/{PREBUILT_COMMIT}/web-llm-models/"
SOURCE_PATTERNS = ["manifest*.json", "tokenizer.json", "tokenizer_config.json", "tokenizer/*", "chat_template.jinja",
                   "heads/*/head.bin", "heads/*/head.safetensors", "LICENSE*", "NOTICE", "README.md"]
COPY_ROOT = ["NOTICE"]


# ---------------------------------------------------------------- recovery matrix

def read_mlc_params(build_dir):
    """name -> numpy array for every parameter of an MLC build (tensor-cache.json + params_shard_*.bin)."""
    import numpy as np
    d = Path(build_dir)
    cache = read_json(d / ("tensor-cache.json" if (d / "tensor-cache.json").exists() else "ndarray-cache.json"))
    out = {}
    for shard in cache["records"]:
        raw = (d / shard["dataPath"]).read_bytes()
        for r in shard["records"]:
            buf = raw[r["byteOffset"]: r["byteOffset"] + r["nbytes"]]
            if r["dtype"] == "float32" and r.get("format") == "f32-to-bf16":
                u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
                a = u.view(np.float32)
            else:
                a = np.frombuffer(buf, dtype=np.dtype(r["dtype"]))
            out[r["name"]] = a.reshape(r["shape"])
    return out


def head_param_names(params):
    """(q_weight, q_scale) names of the output projection: lm_head when the model has one, else the tied embedding."""
    for stem in ("lm_head", "language_model.lm_head"):
        if f"{stem}.q_weight" in params:
            return f"{stem}.q_weight", f"{stem}.q_scale"
    cands = sorted(k for k in params if k.endswith("embed_tokens.q_weight"))
    if len(cands) != 1:
        raise SystemExit(f"cannot find the output projection among MLC parameters (embed_tokens candidates: {cands})")
    return cands[0], cands[0][:-len("q_weight")] + "q_scale"


def dequant_rows(q_weight, q_scale, rows):
    """Rows of a q4f16_1 matrix as float32: nibble j of word w is (w >> 4j) & 15, value (nibble - 7) * scale of its group of 32."""
    import numpy as np
    qw = np.asarray(q_weight)[rows]                       # [n, K/8] uint32
    sc = np.asarray(q_scale)[rows].astype(np.float32)     # [n, K/32]
    shifts = (np.arange(8, dtype=np.uint32) * 4)
    nib = ((qw[:, :, None] >> shifts) & 0xF).reshape(len(rows), -1).astype(np.float32)
    if nib.shape[1] != sc.shape[1] * GROUP:
        raise SystemExit(f"q_weight K={nib.shape[1]} does not match q_scale groups {sc.shape[1]} x {GROUP}")
    return ((nib.reshape(len(rows), -1, GROUP) - ZERO) * sc[:, :, None]).reshape(len(rows), -1)


def recovery_matrix(q_weight, q_scale, n_rows, seed=0):
    """-> (rows int32 [N], P float32 [D, N]) with h = P @ logits[rows], by the normal equations in float64."""
    import numpy as np
    V = q_weight.shape[0]
    n = min(n_rows, V)
    rows = np.sort(np.random.default_rng(seed).choice(V, size=n, replace=False)).astype(np.int32)
    E = dequant_rows(q_weight, q_scale, rows).astype(np.float64)     # [N, D]
    if E.shape[1] > n:
        raise SystemExit(f"{n} rows cannot determine a {E.shape[1]}-dimensional hidden state; raise --recovery-rows")
    P = np.linalg.solve(E.T @ E, E.T)                                  # [D, N]
    return rows, P.astype(np.float32), float(np.linalg.cond(E.T @ E))


def write_recovery(build_dir, out_dir, n_rows, fmt="f16", seed=0):
    import numpy as np
    params = read_mlc_params(build_dir)
    qn, sn = head_param_names(params)
    rows, P, cond = recovery_matrix(params[qn], params[sn], n_rows, seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "recovery.rows.i32").write_bytes(rows.astype("<i4").tobytes())
    (out / f"recovery.P.{fmt}").write_bytes(P.astype("<f2" if fmt == "f16" else "<f4").tobytes())
    return {"rows": int(len(rows)), "hidden": int(P.shape[0]), "format": fmt, "seed": seed, "cond": cond, "source": qn}


# ---------------------------------------------------------------- MLC tool wrappers

def mlc_env(a):
    env = dict(os.environ)
    if getattr(a, "mlc_pythonpath", None):
        env["PYTHONPATH"] = a.mlc_pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    return env


def tool_versions(py):
    """Versions of the MLC wheels in the interpreter that runs the conversion (recorded in the manifest provenance)."""
    code = ("import importlib.metadata as m, json;"
            "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() "
            "if d.metadata['Name'] in ('mlc-llm-nightly-cpu','mlc-ai-nightly-cpu','mlc-llm','mlc-ai','mlc-llm-nightly','mlc-ai-nightly')}))")
    r = subprocess.run([py, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip() or "{}")
    except ValueError:
        return {}


def mlc(a, *args):
    cmd = [a.mlc_python, "-m", "mlc_llm", *map(str, args)]
    log(STAGE, "$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=mlc_env(a))


def fetch(url_or_path, dst):
    dst = Path(dst)
    if re.match(r"^https?://", url_or_path):
        log(STAGE, f"download {url_or_path}")
        with urllib.request.urlopen(url_or_path) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        shutil.copyfile(url_or_path, dst)


# ---------------------------------------------------------------- manifests

def new_manifest(src, name, suffix, files, heads, readout, sibling, versions, recovery, note, lib_source):
    m = json.loads(json.dumps(src))
    m["schema"] = "s1s2.manifest/1"
    short = name.split("/")[-1].lower()
    m["id"] = f"{short}.{suffix}" if suffix else short
    m["tokenizer"]["repo"] = name
    m["tokenizer"]["revision"] = PLACEHOLDER
    build = {"runtime": "webllm", "repo": name, "revision": PLACEHOLDER, "files": files,
             "hiddenOutput": HIDDEN_NAME if readout == "head" else None,
             "devices": ["webgpu"], "dtype": QUANT, "modelId": name.split("/")[-1], "maxSafeTokens": 4096, "fidelity": None}
    if readout == "head":
        build["heads"] = heads(next(iter(src["builds"].values())).get("heads", {}))
    else:  # logits readout: no heads (they read the hidden state), every question type reads the label logits
        m.setdefault("defaults", {}).setdefault("method", {}).update({"predicate": "logits", "choice": "logits", "score": "logits"})
    m["builds"] = {BUILD: build}
    lab = src.get("lab") or {}
    keep = {k: lab[k] for k in LAB_KEEP if k in lab}
    if keep:
        m["lab"] = {"note": note, **keep}
    else:
        m.pop("lab", None)
    args = ["fuse LoRA into base (bf16)", f"mlc_llm convert_weight --quantization {QUANT}", "mlc_llm gen_config", f"model library: {lib_source}"]
    if recovery:
        args.append(f"hidden-state recovery: {recovery['rows']} random vocabulary rows of the quantized output projection "
                    f"(seed {recovery['seed']}), P as {recovery['format']}")
    m["provenance"] = {
        "generator": {"name": "s1s2-convert pack_mlc", "version": __version__}, "createdAt": now_iso(),
        "base": dict(src["base"]),
        "source": {"format": "onnxruntime-genai", "repo": sibling["repo"], "revision": sibling["revision"]},
        "builds": {BUILD: {"tool": "s1s2-convert pack_mlc", "version": __version__, "args": args, "tools": versions}},
        "notes": "Same fine-tuned weights as the source GenAI build, quantized again to 4 bits by MLC; it is a different "
                 "quantization, not a copy of the source graph.",
    }
    return m


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
    m0 = read_json(src / "manifest.json" if (src / "manifest.json").exists() else man_paths[0])
    base = m0["base"]
    work = Path(a.work) if a.work else out.parent / f".{out.name}.work"
    work.mkdir(parents=True, exist_ok=True)
    fused = Path(a.fused) if a.fused else work / "fused"
    if not a.fused and not (fused / "FUSE.json").exists():
        if not (a.lora and a.lora_revision):
            raise SystemExit("pass --lora REPO --lora-revision SHA (or --fused DIR with the merged model)")
        fuse_from_hub(base["repo"], base["revision"], a.lora, a.lora_revision, fused, STAGE)
    name = a.name or a.repo.replace("-ONNX-genai-int4-S1S2", f"-MLC-{QUANT}-S1S2") if not Path(a.repo).is_dir() else a.name
    if not name:
        raise SystemExit("--name NEW_REPO_ID is required for a local --repo")
    # weights + config
    for f in out.glob("params_shard_*.bin"):
        f.unlink()
    mlc(a, "convert_weight", fused, "--quantization", QUANT, "--device", "cpu", "-o", out)
    cmd = ["gen_config", fused, "--quantization", QUANT, "--conv-template", a.conv_template,
           "--prefill-chunk-size", a.prefill_chunk_size, "--context-window-size", a.context_window_size, "-o", out]
    mlc(a, *cmd)
    if not (out / "ndarray-cache.json").exists():      # web-llm fetches the older name
        shutil.copyfile(out / "tensor-cache.json", out / "ndarray-cache.json")
    # library
    if a.prebuilt_lib:
        lib_source = "prebuilt " + a.prebuilt_lib + " (mlc-ai/binary-mlc-llm-libs@" + PREBUILT_COMMIT[:10] + ")"
        fetch(PREBUILT_BASE + a.prebuilt_lib, out / "model-lib.wasm")
    elif a.model_lib:
        lib_source = "compiled " + Path(a.model_lib).name if not a.model_lib.startswith("http") else a.model_lib
        fetch(a.model_lib, out / "model-lib.wasm")
    else:
        raise SystemExit("pass --prebuilt-lib <path under web-llm-models/> or --model-lib <file|url> (see README: compile)")
    # recovery
    recovery = None
    if a.readout == "head":
        recovery = write_recovery(out, out, a.recovery_rows, a.recovery_format)
        log(STAGE, f"recovery: {recovery}")
    # files copied from the source repo (tokenizer for s1s2, heads, licences)
    for p in sorted(src.glob("LICENSE*")) + sorted((src / "tokenizer").glob("*")) + (sorted(src.glob("heads/*/head.*")) if a.readout == "head" else []):
        rel = p.relative_to(src)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out / rel)
    for rel in COPY_ROOT:
        if (src / rel).exists():
            shutil.copyfile(src / rel, out / rel)
    if (out / "NOTICE").exists():
        with open(out / "NOTICE", "a") as f:
            f.write(f"\nWebLLM (MLC) q4f16_1 build of {a.repo} (revision {rev}): the fused weights quantized by MLC LLM "
                    "(Apache-2.0); model-lib.wasm is a WebLLM model library (Apache-2.0).\n")
    files = {"config": pfile(out, "mlc-chat-config.json"), "lib": pfile(out, "model-lib.wasm")}
    if recovery:
        files["recoveryRows"] = pfile(out, "recovery.rows.i32")
        files["recoveryP"] = pfile(out, f"recovery.P.{a.recovery_format}")

    def heads(hs):
        r = {}
        for k, h in hs.items():
            if not (out / h["file"]["path"]).exists():
                raise SystemExit(f"head {k}: {h['file']['path']} missing in the source repo")
            r[k] = {**h, "file": pfile(out, h["file"]["path"])}
            if r[k]["file"]["sha256"] != h["file"]["sha256"]:
                raise SystemExit(f"head {k}: sha256 differs from the source manifest")
        return r

    note = (f"Alternative WebLLM (MLC q4f16_1) build of {a.repo}; prompts and option words are the same. "
            + ("Heads read the hidden state recovered from the logits." if recovery else "Readout: label logits (no heads)."))
    versions = tool_versions(a.mlc_python)
    sibling = {"repo": a.repo, "revision": rev}
    written = []
    for mp in man_paths:
        sm = read_json(mp)
        suffix = mp.name[len("manifest."):-len(".json")] if mp.name != "manifest.json" else (sm.get("lab", {}).get("task") or sm.get("lab", {}).get("language") or "")
        write_json(out / mp.name, new_manifest(sm, name, suffix, files, heads, a.readout, sibling, versions, recovery, note, lib_source))
        written.append(mp.name)
    write_json(out / ".pack-mlc.json", {"source": sibling, "name": name, "manifests": written, "recovery": recovery,
                                         "readout": a.readout, "createdAt": now_iso()})
    log(STAGE, f"manifests: {written}; files: {sorted(files)}")


def stage_pin(a):
    out = Path(a.out)
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision or ""):
        raise SystemExit("--revision: 40-hex commit sha of the files commit")
    meta = read_json(out / ".pack-mlc.json")
    for mn in meta["manifests"]:
        m = read_json(out / mn)
        m["tokenizer"]["revision"] = a.revision
        m["builds"][BUILD]["revision"] = a.revision
        write_json(out / mn, m)
    lib = lib_dir(a.lib)
    subprocess.run(["node", str(lib / "bin" / "s1s2.mjs"), "validate", "--strict", *[str(out / mn) for mn in meta["manifests"]]], check=True)
    log(STAGE, f"pinned {meta['manifests']} to {a.revision}")


def stage_compile(a):
    """mlc_llm compile --device webgpu for a model no prebuilt library covers. Needs emscripten (emcc on PATH) and
    MLC_LLM_SOURCE_DIR pointing at a checkout built with web/prep_emcc_deps.sh (mlc_wasm_runtime.bc). The library must
    come from the same MLC/TVM revision as the @mlc-ai/web-llm release that will load it."""
    cfg = Path(a.out) / "mlc-chat-config.json"
    if not cfg.exists():
        raise SystemExit(f"{cfg} missing: run `pack` first (it writes the config the library is compiled against)")
    mlc(a, "compile", cfg, "--device", "webgpu", "-o", Path(a.out) / "model-lib.wasm")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="s1s2-convert pack-mlc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["pack", "pin", "compile"])
    ap.add_argument("--out", required=True, help="output folder (the new repo's contents)")
    ap.add_argument("--repo", help="pack: source GenAI S1S2 repo id (or a local folder with the same files)")
    ap.add_argument("--revision", help="pack: source revision (default: current main, recorded); pin: files commit sha")
    ap.add_argument("--name", help="pack: new repo id (default: <source> with -MLC-q4f16_1-S1S2)")
    ap.add_argument("--lora", help="pack: the LoRA repo of the specialist")
    ap.add_argument("--lora-revision", help="pack: pinned LoRA revision")
    ap.add_argument("--fused", help="pack: folder with the already merged HF model (skips the fuse)")
    ap.add_argument("--work", help="pack: scratch folder (default: next to --out)")
    ap.add_argument("--prebuilt-lib", help="pack: path of a prebuilt library under mlc-ai/binary-mlc-llm-libs web-llm-models/, "
                                           "e.g. v0_2_86/base/Qwen3-0.6B-q4f16_1_cs1k-webgpu.wasm")
    ap.add_argument("--model-lib", help="pack: your own compiled library (file or URL)")
    ap.add_argument("--conv-template", default="qwen2", help="pack: MLC conversation template written to the config (prompts are plain text, so it is not used by s1s2)")
    ap.add_argument("--prefill-chunk-size", type=int, default=1024)
    ap.add_argument("--context-window-size", type=int, default=4096)
    ap.add_argument("--readout", choices=["head", "logits"], default="head",
                    help="head: heads read the hidden state recovered from the logits (default); logits: label logits, no heads")
    ap.add_argument("--recovery-rows", type=int, default=8192, help="vocabulary rows used to recover the hidden state")
    ap.add_argument("--recovery-format", choices=["f16", "f32"], default="f16")
    ap.add_argument("--mlc-python", default=sys.executable, help="python with mlc-llm and mlc-ai installed (see README for the pinned versions)")
    ap.add_argument("--mlc-pythonpath", help="a mlc-llm checkout's python/ folder to use instead of the installed package's model code")
    ap.add_argument("--lib", help="s1s2 checkout (default: the converter's parent directory)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    {"pack": stage_pack, "pin": stage_pin, "compile": stage_compile}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
