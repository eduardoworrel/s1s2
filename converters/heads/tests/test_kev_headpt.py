"""`from-kev-headpt` on a tiny synthetic head.pt (no download): python -m pytest converters/heads/tests (needs torch and a
built s1s2, `npm run build`).

Kev checkpoints written before head.pt carried the fitted temperature (e.g. jaredpalmer/kev-0.6b@dece6dba) have no
`temperature` key; Kev reads them with its own default, kev/checkpoint.py Meta (kev@2074cc44): temperature = 1.0.
"""
import argparse
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1s2_heads import formats  # noqa: E402

REVISION = "0" * 40


def _head_pt(path: Path, **meta) -> Path:
    g = torch.Generator().manual_seed(0)
    H, D = 16, 8
    head = {"q.weight": torch.randn(D, H, generator=g), "q.bias": torch.randn(D, generator=g),
            "k.weight": torch.randn(D, H, generator=g), "k.bias": torch.randn(D, generator=g)}
    torch.save({"base": "Qwen/Qwen3-0.6B-Base", "head": head, "head_dim": D, "lora": 16, **meta}, path)
    return path


def _convert(tmp_path: Path, monkeypatch, **meta) -> tuple[dict, dict]:
    pt = _head_pt(tmp_path / "head.pt", **meta)
    monkeypatch.setattr(formats, "hub_file", lambda repo, revision, filename, cache=None: pt)
    out = tmp_path / "out"
    formats.from_kev_headpt(argparse.Namespace(repo="test/kev-tiny", revision=REVISION, out=str(out), n=8, seed=0))
    return (json.loads((out / "heads.fragment.json").read_text()), json.loads((out / "fidelity.json").read_text()))


def test_missing_temperature_uses_kev_default(tmp_path, monkeypatch):
    frag, fid = _convert(tmp_path, monkeypatch)
    assert fid["pass"] and fid["decisionAgreement"] == fid["n"] == 8
    assert frag["heads"]["pointer-fp32"]["temperature"] == 1.0
    assert frag["heads"]["pointer-fp32"]["scale"] == pytest.approx(1 / 8 ** 0.5)
    assert frag["provenance"]["source"]["defaults"] == {"temperature": 1.0}


def test_present_temperature_is_kept(tmp_path, monkeypatch):
    frag, fid = _convert(tmp_path, monkeypatch, temperature=1.37)
    assert fid["pass"]
    assert frag["heads"]["pointer-fp32"]["temperature"] == 1.37
    assert "defaults" not in frag["provenance"]["source"]
