"""Reference implementations, run as published (vendor/ holds verbatim copies, vendor/SOURCES.json pins them).

Only the surroundings are stubbed: SGLang's server types (pooler output, /v1/systemone request classes, the
forward batch) and, for the Clef head on CPU, the CUDA flash-attention kernel (replaced by per-request non-causal
scaled-dot-product attention with the same packed-sequence arguments). The arithmetic of every head is the vendored
file's own code. transformers (decision/, models/jev/) is the installed package at the pinned commit 31d7392.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VENDOR = Path(__file__).resolve().parent / "vendor"
TRANSFORMERS_COMMIT = "31d7392997c13162c58f2e3c6555b5e060d6f60b"
SGLANG_COMMIT = "da614bad49dad9c0cbae5b3c0be04581c1f16a9e"


def _module(name: str, file: Path | None = None, **attrs) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    if file is None:
        m = types.ModuleType(name)
        m.__path__ = []  # package
    else:
        spec = importlib.util.spec_from_file_location(name, file)
        m = importlib.util.module_from_spec(spec)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    if file is not None:
        spec.loader.exec_module(m)
    return m


@dataclass
class EmbeddingPoolerOutput:  # sglang.srt.layers.pooler (stub: a container)
    embeddings: Any


# /v1/systemone request classes (stub: the attributes encode_kev / encode_joint_schema read)
@dataclass
class Criteria:
    true: Any = None
    false: Any = None
    _set: tuple = ()

    def model_dump(self, exclude_unset: bool = False) -> dict:
        keys = self._set if exclude_unset else ("true", "false")
        return {k: getattr(self, k) for k in keys}


@dataclass
class SystemOneNoulQuestion:
    instructions: Any = None
    criteria: Criteria | None = None
    type: str = "noul"


@dataclass
class SystemOneChoiceQuestion:
    criteria: dict = field(default_factory=dict)
    instructions: Any = None
    type: str = "choice"


@dataclass
class SystemOneScoreQuestion:
    criteria: list = field(default_factory=list)
    instructions: Any = None
    type: str = "score"


@dataclass
class SystemOneRequest:
    state: Any
    questions: dict
    images: list = field(default_factory=list)


def question(q: dict):
    """A s1s2 style question dict -> the stub class (type noul|choice|score)."""
    if q["type"] == "noul":
        c = q.get("criteria")
        crit = None if c is None else Criteria(true=c.get("true"), false=c.get("false"), _set=tuple(k for k in ("true", "false") if k in c))
        return SystemOneNoulQuestion(instructions=q.get("instructions"), criteria=crit)
    if q["type"] == "choice":
        return SystemOneChoiceQuestion(criteria=dict(q["criteria"]), instructions=q.get("instructions"))
    return SystemOneScoreQuestion(criteria=list(q["criteria"]), instructions=q.get("instructions"))


def sglang():
    """The vendored SGLang modules: pointer_head, joint_schema_head, systemone.kev, systemone.joint_schema."""
    for p in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.entrypoints", "sglang.srt.entrypoints.systemone"):
        _module(p)
    _module("sglang.srt.layers.pooler", None, EmbeddingPoolerOutput=EmbeddingPoolerOutput)
    proto = _module("sglang.srt.entrypoints.systemone.protocol", None, SystemOneChoiceQuestion=SystemOneChoiceQuestion,
                    SystemOneNoulQuestion=SystemOneNoulQuestion, SystemOneQuestion=object, SystemOneRequest=SystemOneRequest,
                    SystemOneScoreQuestion=SystemOneScoreQuestion)
    sys.modules["sglang.srt.entrypoints.systemone.protocol"].__path__ = None
    jsh = _module("sglang.srt.layers.joint_schema_head", VENDOR / "sglang_joint_schema_head.py")
    ph = _module("sglang.srt.layers.pointer_head", VENDOR / "sglang_pointer_head.py")
    kev = _module("sglang.srt.entrypoints.systemone.kev", VENDOR / "sglang_systemone_kev.py")
    js = _module("sglang.srt.entrypoints.systemone.joint_schema", VENDOR / "sglang_systemone_joint_schema.py")
    return types.SimpleNamespace(pointer_head=ph, joint_schema_head=jsh, kev=kev, joint_schema=js, protocol=proto)


@dataclass
class ForwardBatch:  # the fields PointerPooler / JointSchemaPooler read
    batch_size: int
    extend_seq_lens_cpu: list
    decision_layouts: list
    extend_prefix_lens_cpu: list | None = None
    input_ids: Any = None


class _TorchCpuFlash:
    """Stands in for module-global `torch` in joint_schema_head: identical except aten._flash_attention_forward,
    which runs on CPU as non-causal attention within each packed sequence (cu_seqlens), softmax scale 1/sqrt(d)."""

    def __init__(self, torch):
        self._t = torch
        outer = self

        class _Aten:
            def __getattr__(self, n):
                return getattr(torch.ops.aten, n)

            @staticmethod
            def _flash_attention_forward(q, k, v, cu_q, cu_k, max_q, max_k, dropout, is_causal, return_debug_mask, *a, **kw):
                assert dropout == 0.0 and not is_causal
                F = torch.nn.functional
                outs = []
                for i in range(len(cu_q) - 1):
                    a0, a1, b0, b1 = int(cu_q[i]), int(cu_q[i + 1]), int(cu_k[i]), int(cu_k[i + 1])
                    qq, kk, vv = (x.transpose(0, 1).unsqueeze(0) for x in (q[a0:a1], k[b0:b1], v[b0:b1]))
                    outs.append(F.scaled_dot_product_attention(qq, kk, vv).squeeze(0).transpose(0, 1))
                return (torch.cat(outs, 0),)

        class _Ops:
            aten = _Aten()

            def __getattr__(self, n):
                return getattr(torch.ops, n)

        self.ops = _Ops()

    def __getattr__(self, n):
        return getattr(self._t, n)


def clef_cpu(jsh):
    """Patch the joint head module for CPU (see _TorchCpuFlash). Returns the module."""
    import torch
    if not isinstance(jsh.torch, _TorchCpuFlash):
        jsh.torch = _TorchCpuFlash(torch)
    return jsh
