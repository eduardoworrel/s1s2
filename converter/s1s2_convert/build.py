"""int4 (or int8) ONNX Runtime GenAI build of the fused checkpoint, with the hidden-state output the heads read.

Builder: onnxruntime-genai 0.19.0 `models/builder.py`, run from a copy of the installed package with two builder fixes
applied (s1s2_convert/patches/onnxruntime-genai-0.19.0-builder-fixes.patch, shipped here: two commits on top of
onnxruntime-genai 0.19.0 d01b2441, files builder.py + builders/):
  38cbf9c3  SmolLM3 NoPE layers: disable RoPE instead of unfusing it (0.19.0 raises KeyError on position_ids)
  b4bb8aec  Qwen3.5/3.8 hybrid layers: linear_attn_op=linear_attention for dense CUDA/WebGPU exports
Or `--builder-src <fork>/src/python/py/models` to run a checkout directly.
Options (the same as the builds measured in the model cards):
  -p int4 -e webgpu --extra_options include_hidden_states=true exclude_mtp=true
"""
import shutil
import subprocess
from pathlib import Path

from .util import log, run, sha256_file, python_exe

PATCH = Path(__file__).resolve().parent / "patches" / "onnxruntime-genai-0.19.0-builder-fixes.patch"
FORK = {"repo": "https://github.com/microsoft/onnxruntime-genai",
        "base": "d01b2441a72d4e3223415b9b08474c6e65fbc028 (Version 0.19.0)",
        "commits": ["38cbf9c3f5752df736fec060172960f42b2a2a2a SmolLM3 NoPE: disable RoPE instead of unfusing it",
                    "b4bb8aec7176501b0971d01d6e7335c99ddbd7ba Qwen3.5 linear_attn_op=linear_attention for dense CUDA/WebGPU"]}
GENAI_VERSION = "0.19.0"


def patched_builder(work):
    """Copy of the installed onnxruntime_genai/models with the fork patch applied. -> (dir, provenance)."""
    import onnxruntime_genai as og
    if og.__version__ != GENAI_VERSION:
        raise SystemExit(f"onnxruntime-genai {og.__version__} installed; the patch is for {GENAI_VERSION}")
    src = Path(og.__file__).resolve().parent / "models"
    dst = Path(work) / "builder-src"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    r = subprocess.run(["patch", "-p5", "-d", str(dst), "-i", str(PATCH)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"builder patch did not apply:\n{r.stdout}\n{r.stderr}")
    return dst, {"tool": "onnxruntime-genai builder", "version": GENAI_VERSION, "source": f"pip onnxruntime-genai=={GENAI_VERSION} + s1s2_convert/patches/{PATCH.name}",
                 "patches": [{"path": "converter/s1s2_convert/patches/" + PATCH.name, "sha256": sha256_file(PATCH),
                              "description": "; ".join(FORK["commits"])}]}


def build(fused_dir, out_dir, work, precision="int4", ep="webgpu", builder_src=None, extra=None):
    out_dir = Path(out_dir)
    if builder_src:
        bdir = Path(builder_src)
        prov = {"tool": "onnxruntime-genai builder", "version": GENAI_VERSION, "source": f"checkout {bdir}"}
        try:
            prov["source"] += " @ " + subprocess.run(["git", "-C", str(bdir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception:
            pass
    else:
        bdir, prov = patched_builder(work)
    cache = Path(work) / "builder-cache"
    cache.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    opts = ["include_hidden_states=true", "exclude_mtp=true", *(extra or [])]
    args = ["-i", str(fused_dir), "-o", str(out_dir), "-p", precision, "-e", ep, "-c", str(cache), "--extra_options", *opts]
    run([python_exe(), bdir / "builder.py", *args], "build", env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, cwd=bdir)
    for f in ("genai_config.json", "model.onnx", "model.onnx.data"):
        if not (out_dir / f).exists():
            raise SystemExit(f"build: {out_dir / f} missing after the builder ran")
    log("build", f"{precision} {ep} build in {out_dir}")
    return {**prov, "args": ["-p", precision, "-e", ep, "--extra_options", *opts]}
