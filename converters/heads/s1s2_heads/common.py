"""Shared pieces of the head converters: pinned Hub downloads, head writers, the s1s2 scorer, fidelity records."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE.parents[2]  # s1s2 checkout (converters/heads/s1s2_heads -> s1s2)
NODE_SCORER = HERE / "node" / "score.mjs"
SPEC_VERSION = "0.2-draft"


def sha256(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned(p: Path, rel: str) -> dict:
    return {"path": rel, "sha256": sha256(p), "size": p.stat().st_size}


def hub_file(repo: str, revision: str, filename: str, cache: str | None = None) -> Path:
    """One file of a Hub repo at a 40-hex revision (never a branch name)."""
    if len(revision) != 40:
        raise SystemExit(f"--revision must be a 40-hex commit, got {revision!r}")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, revision=revision, cache_dir=cache))


def lib_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(LIB), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def f16_bits(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype="<f4").astype("<f2").tobytes()


def write_head_bin(path: Path, q, qb, k, kb, *, scale: float, temperature: float, score_offset: float = 0.0, extra: dict | None = None) -> None:
    """head.bin (alias3-web layout, spec 6.1) with an explicit `scale` so every reader agrees."""
    D, H = q.shape
    meta = {"headDim": int(D), "hiddenSize": int(H), "temperature": float(temperature), "score_offset": float(score_offset), "scale": float(scale), **(extra or {})}
    js = json.dumps(meta).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(js))); f.write(js)
        for a in (q, qb, k, kb):
            f.write(f16_bits(a))


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(path), metadata=metadata or {})


@dataclass
class Inputs:
    """float32 blocks shared by the reference and the s1s2 scorer."""
    chunks: list[np.ndarray] = field(default_factory=list)
    size: int = 0

    def add(self, a: np.ndarray) -> dict:
        a = np.asarray(a, dtype=np.float32)
        a2 = a.reshape(1, -1) if a.ndim == 1 else a
        b = {"offset": self.size, "rows": int(a2.shape[0]), "dim": int(a2.shape[1])}
        self.chunks.append(a2.reshape(-1)); self.size += a2.size
        return b

    def write(self, p: Path) -> None:
        np.concatenate(self.chunks).astype("<f4").tofile(p)


def run_lib(head_path: Path | None, head_spec: dict, cases: list[dict], inputs: Inputs, norm: tuple[Path, dict] | None = None) -> list[dict]:
    if not (LIB / "dist" / "index.js").exists():
        raise SystemExit(f"s1s2 is not built: run `npm ci && npm run build` in {LIB}")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inputs.write(d / "in.f32")
        job = {"head": {"path": str(head_path), "spec": head_spec} if head_path else None, "cases": cases}
        if norm:
            job["norm"] = {"path": str(norm[0]), "spec": norm[1]}
        (d / "job.json").write_text(json.dumps(job))
        subprocess.run(["node", str(NODE_SCORER), str(d / "job.json"), str(d / "in.f32"), str(d / "out.json")], check=True)
        return json.loads((d / "out.json").read_text())["results"]


def softmax(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def compare(ref: list[list[float]], ours: list[list[float]], *, tol_p: float, tol_z: float | None = None) -> dict:
    """Per question: same argmax, |dp|, |dz| (dz relative to max(1, |z|))."""
    n = agree = 0
    max_dp = max_dz = 0.0
    for zr, zo in zip(ref, ours, strict=True):
        zr, zo = np.asarray(zr, np.float64), np.asarray(zo, np.float64)
        pr, po = softmax(zr), softmax(zo)
        n += 1
        agree += int(np.argmax(pr) == np.argmax(po))
        max_dp = max(max_dp, float(np.abs(pr - po).max()))
        max_dz = max(max_dz, float((np.abs(zr - zo) / np.maximum(1.0, np.abs(zr))).max()))
    ok = agree == n and max_dp <= tol_p and (tol_z is None or max_dz <= tol_z)
    return {"n": n, "decisionAgreement": agree, "decisionChanged": n - agree, "maxAbsDp": max_dp, "maxRelDz": max_dz,
            "tolerance": {"maxAbsDp": tol_p, **({"maxRelDz": tol_z} if tol_z is not None else {})}, "pass": bool(ok)}


def fidelity_record(*, source_format: str, reference: dict, inputs: dict, result: dict, ours: dict, notes: str = "") -> dict:
    """spec 0.2 section 9.2: one record per conversion."""
    return {"schema": "s1s2.conversion-fidelity/1", "spec": SPEC_VERSION, "sourceFormat": source_format, "reference": reference,
            "ours": {"impl": "s1s2", "commit": lib_commit(), **ours}, "inputs": inputs, **result, **({"notes": notes} if notes else {})}


def write_outputs(out: Path, fragment: dict, fidelity: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "heads.fragment.json").write_text(json.dumps(fragment, indent=1, ensure_ascii=False) + "\n")
    (out / "fidelity.json").write_text(json.dumps(fidelity, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: fidelity.get(k) for k in ("sourceFormat", "n", "decisionAgreement", "maxAbsDp", "maxRelDz", "pass")}))


def rng_hidden(seed: int, n: int, dim: int, scale: float = 1.0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal((n, dim)) * scale).astype(np.float32)
