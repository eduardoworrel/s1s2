"""`s1s2-heads from-<format>`: one converter per published head format (the published formats listed in spec section 11).

Each converter reads the source format with its declared name (never guessed from file names), writes the head in a
form s1s2 reads, a manifest fragment (heads / logits / norm / encoding pieces + provenance.source) and a
`fidelity.json` record: the format's own reference code and s1s2's reader on the same inputs, decisions and
probabilities compared against declared tolerances (spec 0.2 section 9.2).

Inputs are synthetic hidden states (seeded Gaussian) unless a converter says otherwise: they test the head
arithmetic and its declaration (scale, temperature, softcap, rows, norm) on the REAL published weights. End-to-end
positions/prompt parity on a real backbone is `e2e_kev.py`.
"""
from __future__ import annotations

import argparse
import re
import json
import math
from pathlib import Path

import numpy as np

from .common import (Inputs, compare, fidelity_record, hub_file, pinned, rng_hidden, run_lib, sha256, write_head_bin,
                     write_outputs, write_safetensors)

# Probabilities: fp32 head math in both implementations, different summation order -> 1e-5. fp16 head.bin -> 2e-3.
TOL_FP32 = 1e-5
TOL_FP16 = 2e-3


def _source(fmt: str, repo: str, revision: str, files: list[str], **kw) -> dict:
    return {"format": fmt, "repo": repo, "revision": revision, "files": files, **kw}


def _cases_pointer(inp: Inputs, H: int, n: int, seed: int, scale: float = 1.0):
    hid = rng_hidden(seed, n * 6, H, scale)
    cases, ref_rows = [], []
    i = 0
    for c in range(n):
        k = 2 + c % 5
        hq, hks = hid[i], hid[i + 1:i + 1 + k]
        i += 6
        cases.append({"kind": "pointer", "hq": inp.add(hq), "hks": inp.add(hks)})
        ref_rows.append((hq, hks))
    return cases, ref_rows


# ---------------------------------------------------------------- Kev head.pt (pointer, fp32)
KEV_META_DEFAULTS = {"head_dim": 256, "temperature": 1.0}  # kev@2074cc44 kev/checkpoint.py Meta: head_dim, temperature


def from_kev_headpt(a) -> None:
    import torch
    from . import refs
    sg = refs.sglang()
    p = hub_file(a.repo, a.revision, "head.pt")
    # sglang@da614ba configs/model_config.py load_pointer_head: defaults, then torch.load(weights_only=True)
    from_cfg = {"option_isolation": False, "special_embeddings": False}
    raw = torch.load(p, map_location="cpu", weights_only=True)
    # Fields older head.pt files did not write get Kev's own reader defaults (kev@2074cc44 kev/checkpoint.py Meta), e.g.
    # jaredpalmer/kev-0.6b@dece6dba predates "head.pt carries the fitted temperature" -> temperature 1.0 (raw logits).
    defaults = {k: v for k, v in KEV_META_DEFAULTS.items() if k not in raw}
    meta = {**from_cfg, **defaults, **raw}
    for name in ("option_isolation", "special_embeddings"):
        if meta[name]:
            raise SystemExit(f"head.pt {name} is not supported (changes the prompt or attention mask)")
    sd = {k: v.float().numpy() for k, v in meta["head"].items()}
    D, H = sd["q.weight"].shape
    T, scale = float(meta["temperature"]), 1 / math.sqrt(meta["head_dim"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_safetensors(out / "head.safetensors", sd, {"scale": repr(scale), "temperature": repr(T), "score_offset": "0.0", "source": f"{a.repo}@{a.revision}:head.pt"})
    write_head_bin(out / "head.bin", sd["q.weight"], sd["q.bias"], sd["k.weight"], sd["k.bias"], scale=scale, temperature=T)
    pooler = sg.pointer_head.PointerPooler(H, meta)
    pooler.q.weight.data, pooler.q.bias.data = torch.tensor(sd["q.weight"]), torch.tensor(sd["q.bias"])
    pooler.k.weight.data, pooler.k.bias.data = torch.tensor(sd["k.weight"]), torch.tensor(sd["k.bias"])
    inp = Inputs()
    cases, rows = _cases_pointer(inp, H, a.n, a.seed, 2.0)
    ref = []
    for hq, hks in rows:
        k = len(hks)
        hs = torch.tensor(np.vstack([hks, hq[None]]))  # option i ends at token i+1, the question span ends at the last token
        layout = sg.joint_schema_head.pack_decision_layout(k + 1, [sg.joint_schema_head.LayoutQuestion(0, (k, k + 1), tuple((i, i + 1) for i in range(k)))])
        fb = refs.ForwardBatch(batch_size=1, extend_seq_lens_cpu=[k + 1], decision_layouts=[layout])
        ref.append(pooler(hs, fb).embeddings[0].tolist())
    files = {}
    for fname, spec, tol in (("head.safetensors", {"kind": "pointer", "format": "safetensors"}, TOL_FP32), ("head.bin", {"kind": "pointer", "format": "head.bin"}, TOL_FP16)):
        ours = [r["scores"][0] for r in run_lib(out / fname, spec, cases, inp)]
        files[fname] = compare(ref, ours, tol_p=tol)
    src = _source("kev-head.pt", a.repo, a.revision, ["head.pt"], base=meta.get("base"), baseRevision=meta.get("base_revision"),
                  **({"defaults": defaults} if defaults else {}))
    fragment = {"heads": {
        "pointer-fp32": {"file": pinned(out / "head.safetensors", "heads/head.safetensors"), "types": ["predicate", "choice", "score"], "kind": "pointer", "format": "safetensors", "scale": scale, "temperature": T, "source": "kev-head.pt"},
        "pointer-fp16": {"file": pinned(out / "head.bin", "heads/head.bin"), "types": ["predicate", "choice", "score"], "kind": "pointer", "format": "head.bin", "scale": scale, "source": "kev-head.pt"}},
        "readout": {"predicate": "options", "choice": "option", "score": "option"},
        "encoding": {"kind": "kev", "predicateOptions": ["no", "yes"], "delimiters_text": ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]},
        "provenance": {"source": src}}
    rec = fidelity_record(source_format="kev-head.pt", reference={"impl": "sglang PointerPooler", "repo": "sgl-project/sglang", "commit": refs.SGLANG_COMMIT, "file": "python/sglang/srt/layers/pointer_head.py"},
                          inputs={"kind": "synthetic-hidden", "n": a.n, "seed": a.seed, "options": "2..6"}, result=files["head.safetensors"],
                          ours={"reader": "loadPointerSafetensors + headScore"},
                          notes="score questions: Kev scores levels as options (kev_options renders each level), so readout.score = 'option'. "
                                f"head.bin (fp16 weights): {json.dumps(files['head.bin'])}")
    write_outputs(out, fragment, rec)


# ---------------------------------------------------------------- GGUF decision.type = kev
def _gguf_sparse(url: str, dst: Path, want: tuple[str, ...]) -> tuple[dict, dict]:
    """GGUF without downloading it: range-fetch the header (key-values + tensor infos), then only the wanted
    tensors' bytes, into a sparse file of the full size that gguf-py's GGUFReader memory-maps."""
    import requests
    from gguf import GGUFReader
    size = int(requests.head(url, allow_redirects=True).headers["Content-Length"])
    n = 1 << 25
    while True:
        r = requests.get(url, headers={"Range": f"bytes=0-{n - 1}"})
        with open(dst, "wb") as f:
            f.truncate(size); f.seek(0); f.write(r.content)
        try:
            rd = GGUFReader(dst)
            break
        except Exception:
            if n >= 1 << 28:
                raise
            n *= 4
    fetched = {}
    for t in rd.tensors:
        if t.name in want:
            off, nb = int(t.data_offset), int(t.n_bytes)
            r = requests.get(url, headers={"Range": f"bytes={off}-{off + nb - 1}"})
            assert len(r.content) == nb
            with open(dst, "r+b") as f:
                f.seek(off); f.write(r.content)
            fetched[t.name] = {"offset": off, "bytes": nb}
    return {"size": size, "header_bytes": n}, fetched


def from_gguf_kev(a) -> None:
    import torch
    from gguf import GGUFReader
    from . import refs
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{a.repo}/resolve/{a.revision}/{a.file}"
    sparse = out / ".sparse.gguf"
    info, fetched = _gguf_sparse(url, sparse, ("cls.output.weight", "cls.output.bias"))
    rd = GGUFReader(sparse)
    kv = {f.name: f.contents() for f in rd.fields.values() if not f.name.startswith("tokenizer.ggml")}
    arch = kv["general.architecture"]
    dtype = kv.get(f"{arch}.decision.type")
    if dtype != "kev":
        raise SystemExit(f"{a.file}: {arch}.decision.type is {dtype!r}; this converter reads only 'kev'")
    from gguf.quants import dequantize
    tens = {t.name: t for t in rd.tensors}
    # llama.cpp convert (PR #29818): the Kev head is the decision classifier, q|k concatenated on the output axis,
    # stored as cls.output.{weight,bias}; ggml ne = [hidden, 2*head_dim] -> numpy [2*head_dim, hidden]
    deq = lambda t: dequantize(t.data, t.tensor_type).astype(np.float32).reshape([int(x) for x in reversed(t.shape)])
    W2 = deq(tens["cls.output.weight"])
    B = deq(tens["cls.output.bias"]).reshape(-1) if "cls.output.bias" in tens else np.zeros(W2.shape[0], np.float32)
    D = W2.shape[0] // 2
    temps = {k.split(".decision.temperature.")[1]: float(v) for k, v in kv.items() if ".decision.temperature." in k}
    tmpl = kv.get("tokenizer.chat_template.systemone")
    q, k_, qb, kb = W2[:D], W2[D:], B[:D], B[D:]
    sd = {"q.weight": q, "q.bias": qb, "k.weight": k_, "k.bias": kb}
    scale = 1 / math.sqrt(D)
    T = temps.get("choice", next(iter(temps.values()), 1.0))
    write_safetensors(out / "head.safetensors", sd, {"scale": repr(scale), "temperature": repr(T), "score_offset": "0.0"})
    result = {}
    if a.compare_headpt:
        repo, rev = a.compare_headpt.split("@")
        ck = torch.load(hub_file(repo, rev, "head.pt"), map_location="cpu", weights_only=True)
        diffs = {n: float(np.abs(sd[n] - ck["head"][n].float().numpy()).max()) for n in sd}
        result = {"tensorMaxAbsDiffVsHeadPt": diffs, "headPt": a.compare_headpt, "temperatureHeadPt": float(ck["temperature"])}
    # fidelity: gguf-extracted head vs the SGLang pointer math over the same tensors (as read by GGUFReader)
    sg = refs.sglang()
    pooler = sg.pointer_head.PointerPooler(W2.shape[1], {"head_dim": D, "temperature": T})
    for n, v in sd.items():
        getattr(pooler, n.split(".")[0]).__getattr__(n.split(".")[1]).data = torch.tensor(v)
    inp = Inputs()
    cases, rows = _cases_pointer(inp, W2.shape[1], a.n, a.seed, 2.0)
    ref = []
    for hq, hks in rows:
        kk = len(hks)
        hs = torch.tensor(np.vstack([hks, hq[None]]))
        layout = sg.joint_schema_head.pack_decision_layout(kk + 1, [sg.joint_schema_head.LayoutQuestion(0, (kk, kk + 1), tuple((i, i + 1) for i in range(kk)))])
        ref.append(pooler(hs, refs.ForwardBatch(1, [kk + 1], [layout])).embeddings[0].tolist())
    ours = [r["scores"][0] for r in run_lib(out / "head.safetensors", {"kind": "pointer", "format": "safetensors"}, cases, inp)]
    cmp = compare(ref, ours, tol_p=TOL_FP32)
    sparse.unlink()
    fragment = {"heads": {"pointer": {"file": pinned(out / "head.safetensors", "heads/head.safetensors"), "types": ["predicate", "choice", "score"], "kind": "pointer", "format": "safetensors", "scale": scale, "temperature": T, "source": "gguf-kev"}},
                "x-gguf": {"decisionTemperature": temps, "templateSystemone": tmpl, "keys": sorted(k for k in kv if ".decision." in k)},
                "provenance": {"source": _source("gguf-kev", a.repo, a.revision, [a.file], bytesFetched={**fetched, **info})}}
    rec = fidelity_record(source_format="gguf-kev", reference={"impl": "gguf-py GGUFReader + sglang PointerPooler", "repo": "ggml-org/llama.cpp", "commit": "9fba51d2c9a1b5cb5f0c47986c031334e2d7663a", "file": "gguf-py/gguf/gguf_reader.py"},
                          inputs={"kind": "synthetic-hidden", "n": a.n, "seed": a.seed}, result=cmp, ours={"reader": "loadPointerSafetensors + headScore"},
                          notes=json.dumps(result) if result else "")
    write_outputs(out, fragment, rec)


# ---------------------------------------------------------------- alias3-web head.bin + norm.bin
def from_alias3_web(a) -> None:
    import subprocess
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hb, nb, mjs = (hub_file(a.repo, a.revision, f) for f in ("heads/head.bin", "heads/norm.bin", "aliasweb.mjs"))
    meta = json.loads(hb.read_bytes()[4:4 + int.from_bytes(hb.read_bytes()[:4], "little")])
    norm_meta = json.loads(nb.read_bytes()[4:4 + int.from_bytes(nb.read_bytes()[:4], "little")])
    D = meta["headDim"]
    for f, dst in ((hb, "head.bin"), (nb, "norm.bin")):
        (out / dst).write_bytes(f.read_bytes())  # bytes unchanged: the scale is declared in the manifest
    # fidelity: s1s2's own test runs the vendored reader (test/alias3web.test.mjs); run it on these files
    from .common import LIB
    env = {**__import__("os").environ, "ALIAS3WEB_DIR": str(Path(hb).parent.parent)}
    r = subprocess.run(["node", "--test", str(LIB / "test" / "alias3web.test.mjs")], env=env, capture_output=True, text=True)
    passed = r.returncode == 0 and re.search(r"skipped 0\b", r.stdout) is not None
    scale = meta.get("scale", 1 / math.sqrt(D))
    fragment = {"heads": {"pointer": {"file": pinned(out / "head.bin", "heads/head.bin"), "types": ["predicate", "choice", "score"], "kind": "pointer", "format": "head.bin", "scale": scale, "source": "alias3-web-head.bin"}},
                "norm": {"file": pinned(out / "norm.bin", "heads/norm.bin"), "kind": "rmsnorm", "eps": 0, "weight": "w"},
                "readout": {"predicate": "options", "choice": "option", "score": "option"},
                "provenance": {"source": _source("alias3-web-head.bin", a.repo, a.revision, ["heads/head.bin", "heads/norm.bin", "aliasweb.mjs"], aliaswebSha256=sha256(mjs), normHeader=norm_meta)}}
    rec = fidelity_record(source_format="alias3-web-head.bin", reference={"impl": "aliasweb.mjs loadHead/headLogit/norm (vendored verbatim, s1s2 test/reference/alias3web-2ab2e60.mjs)", "repo": a.repo, "commit": a.revision, "file": "aliasweb.mjs"},
                          inputs={"kind": "synthetic-hidden", "n": 32, "seed": 2026, "note": "pre-norm states, std 2"},
                          result={"n": 32, "pass": passed, "maxAbsDp": 0.0 if passed else None, "decisionAgreement": 32 if passed else None, "maxRelDz": 0.0 if passed else None,
                                  "tolerance": {"maxAbsDp": 0.0}},
                          ours={"reader": "loadHead (scale default 1/sqrt(headDim)) + loadNorm/applyNorm (eps 0) + headScore"},
                          notes="exact equality required (same fp64 arithmetic order). " + ("node test passed" if passed else r.stdout[-2000:]))
    write_outputs(out, fragment, rec)


# ---------------------------------------------------------------- linear readouts: decision_config (pplx/d3) and JEV judge_config
def _jev_reference(config_kwargs: dict, weight: np.ndarray, bias: np.ndarray | None, hidden_rows: list[np.ndarray], qtypes: list[str], nopts: list[int]):
    """transformers@31d7392 JevForDecision.forward with the backbone replaced by the given last_hidden_state."""
    import torch
    from types import SimpleNamespace
    from transformers import JevConfig
    from transformers.models.jev.modeling_jev import JevForDecision
    cfg = JevConfig(**config_kwargs)
    model = JevForDecision.__new__(JevForDecision)
    torch.nn.Module.__init__(model)
    model.config = cfg
    model.proj = torch.nn.Linear(cfg.hidden_size, cfg.num_verbalizers, bias=bias is not None)
    model.proj.weight.data = torch.tensor(weight)
    if bias is not None:
        model.proj.bias.data = torch.tensor(bias)

    class Backbone(torch.nn.Module):
        def forward(self, input_ids=None, use_cache=None, **kw):
            return SimpleNamespace(last_hidden_state=self.h)
    model.model = Backbone()
    from transformers.decision import Question
    out = []
    for h, t, n in zip(hidden_rows, qtypes, nopts):
        model.model.h = torch.tensor(h[None, None, :])  # one prompt, one token: the mark follows it
        logits = JevForDecision.forward(model, input_ids=torch.zeros(1, 1, dtype=torch.long), marks=torch.tensor([[1]]),
                                        num_options=torch.tensor([[n]]), question_type_ids=torch.tensor([[Question.questions_types.index(t)]]))
        out.append(logits.double().tolist())
    return out


def _temperature_ref(z: list[float], T: float) -> list[float]:
    """transformers DecisionConfig.get_logits_processor: TemperatureLogitsWarper(T) on the option scores."""
    import torch
    from transformers.generation import TemperatureLogitsWarper
    return TemperatureLogitsWarper(float(T))(None, torch.tensor([z], dtype=torch.float64))[0].tolist()


QT = {"noul": "predicate", "choice": "choice", "score": "score"}


def _linear_cases(inp: Inputs, H: int, n: int, seed: int, ranges: dict, scale: float):
    hid = rng_hidden(seed, n, H, scale)
    cases, rows, qtypes, nopts = [], [], [], []
    kinds = list(ranges)
    for i in range(n):
        t = kinds[i % len(kinds)]
        width = ranges[t][1] - ranges[t][0]
        k = 2 if t == "noul" else max(2, min(width, 2 + (i % 7)))
        cases.append({"kind": "linear", "type": QT[t], "n": k, "h": inp.add(hid[i])})
        rows.append(hid[i]); qtypes.append(t); nopts.append(k)
    return cases, rows, qtypes, nopts


def from_decision_config(a) -> None:
    from safetensors.numpy import load_file
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dc = json.loads(hub_file(a.repo, a.revision, "decision_config.json").read_text())
    # sglang load_decision_config rules
    if dc["format_version"] != 1 or dc.get("pooling", "last") != "last" or dc.get("attention_mode", "causal") not in ("causal", "noncausal_full_attention"):
        raise SystemExit(f"decision_config.json not supported: {dc.get('format_version')}, {dc.get('pooling')}, {dc.get('attention_mode')}")
    st = load_file(str(hub_file(a.repo, a.revision, a.readout_file)))
    W = st["weight"].astype(np.float32)
    K, H = W.shape
    T = float(dc["temperature"])
    write_safetensors(out / "head.safetensors", {"weight": W})
    rng = {"noul": [0, K], "choice": [0, K], "score": [0, K]}
    inp = Inputs()
    cases, rows, qt, no = _linear_cases(inp, H, a.n, a.seed, rng, 1.0)
    ref_raw = _jev_reference({"hidden_size": H, "num_verbalizers": K, "offsets": {"noul": 0, "choice": 0, "score": 0}, "softcap": None}, W, None, rows, qt, no)
    ref = [_temperature_ref(z, T) for z in ref_raw]
    spec = {"kind": "linear", "format": "safetensors", "tensors": {"weight": "weight", "bias": None}, "temperature": T, "read": "last"}
    ours = [r["scores"][0] for r in run_lib(out / "head.safetensors", spec, cases, inp)]
    cmp = compare(ref, ours, tol_p=TOL_FP32, tol_z=1e-4)
    noncausal = dc.get("attention_mode") == "noncausal_full_attention"
    fragment = {"heads": {"readout": {"file": pinned(out / "head.safetensors", "heads/head.safetensors"), "types": ["predicate", "choice", "score"], "kind": "linear", "format": "safetensors",
                                      "tensors": {"weight": "weight", "bias": None}, "rows": {"predicate": {"start": 0}, "choice": {"start": 0}, "score": {"start": 0}}, "temperature": T, "read": "last", "source": "decision-config-readout"}},
                "x-decision-config": {"codes": dc.get("codes"), "token_ids": dc.get("token_ids"), "prompt": dc.get("prompt"), "attention_mode": dc.get("attention_mode", "causal")},
                "attention": "noncausal" if noncausal else "causal",
                "provenance": {"source": _source("decision-config-readout", a.repo, a.revision, ["decision_config.json", a.readout_file])}}
    rec = fidelity_record(source_format="decision-config-readout", reference={"impl": "transformers JevForDecision.forward (decision_config branch of JevConfig) + TemperatureLogitsWarper", "repo": "huggingface/transformers", "commit": "31d7392997c13162c58f2e3c6555b5e060d6f60b", "file": "src/transformers/models/jev/modeling_jev.py"},
                          inputs={"kind": "synthetic-hidden", "n": a.n, "seed": a.seed}, result=cmp, ours={"reader": "loadLinearHead + linearScores"},
                          notes=("Head converted; package NOT runnable in s1s2 0.1: attention_mode noncausal_full_attention needs a full-attention build "
                                 "(spec 7.2 `attention`), which no s1s2 runtime provides. " if noncausal else "") +
                                "Option codes A, B, ... are rendered by the source's prompt (DECIDER_PROMPTS, in code): the manifest must carry that layout.")
    write_outputs(out, fragment, rec)


def from_jev(a) -> None:
    from safetensors.numpy import load_file
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jc = json.loads(hub_file(a.repo, a.revision, "judge_config.json").read_text())
    cal = json.loads(hub_file(a.repo, a.revision, "calibration.json").read_text())
    st = load_file(str(hub_file(a.repo, a.revision, "head.safetensors")))
    W, B = st["proj.weight"].astype(np.float32), st.get("proj.bias")
    B = None if B is None else B.astype(np.float32)
    K, H = W.shape
    ranges = jc["slots"]["ranges"]
    softcap = jc.get("softcap")
    temps = cal["per_kind"]
    write_safetensors(out / "head.safetensors", {"proj.weight": W, **({"proj.bias": B} if B is not None else {})})
    inp = Inputs()
    cases, rows, qt, no = _linear_cases(inp, H, a.n, a.seed, ranges, 1.0)
    offsets = {t: r[0] for t, r in ranges.items()}
    ref_raw = _jev_reference({"hidden_size": H, "num_verbalizers": K, "offsets": offsets, "softcap": softcap}, W, B, rows, qt, no)
    ref = [_temperature_ref(z, temps.get(t, 1.0)) for z, t in zip(ref_raw, qt)]
    spec = {"kind": "linear", "format": "safetensors", "tensors": {"weight": "proj.weight", "bias": "proj.bias" if B is not None else None},
            "rows": {QT[t]: {"start": r[0], "end": r[1]} for t, r in ranges.items()}, "softcap": softcap,
            "temperature": {QT[t]: v for t, v in temps.items() if t in QT}, "read": "decide"}
    ours = [r["scores"][0] for r in run_lib(out / "head.safetensors", spec, cases, inp)]
    cmp = compare(ref, ours, tol_p=TOL_FP32, tol_z=1e-4)
    fragment = {"heads": {"jev": {"file": pinned(out / "head.safetensors", "heads/head.safetensors"), "types": ["predicate", "choice", "score"], **spec, "source": "jev-judge-config"}},
                "x-jev": {"verbalizers": jc["slots"].get("verbalizers"), "template_version": jc["slots"].get("template_version"), "lm_head_included": jc.get("lm_head_included"), "lora": jc.get("lora")},
                "provenance": {"source": _source("jev-judge-config", a.repo, a.revision, ["judge_config.json", "head.safetensors", "calibration.json"])}}
    rec = fidelity_record(source_format="jev-judge-config", reference={"impl": "transformers JevForDecision.forward (judge_config branch) + TemperatureLogitsWarper(calibration.per_kind)", "repo": "huggingface/transformers", "commit": "31d7392997c13162c58f2e3c6555b5e060d6f60b", "file": "src/transformers/models/jev/modeling_jev.py"},
                          inputs={"kind": "synthetic-hidden", "n": a.n, "seed": a.seed}, result=cmp, ours={"reader": "loadLinearHead + linearScores"},
                          notes="noul rows in noul_options order [false, true] = s1s2 predicate order [no, yes]. Prompt: JEV_TEMPLATE bare-v1 (in transformers code), read at the token before <|mark|>.")
    write_outputs(out, fragment, rec)


# ---------------------------------------------------------------- label-token logits: transformers DecisionConfig / Standard One lm_rows
def from_transformers_decision(a) -> None:
    from transformers import AutoTokenizer
    from transformers.decision import DecisionConfig, Question
    from transformers.decision.template_utils import label_token_ids
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg_path = hub_file(a.repo, a.revision, "decision.json")
    cfg = DecisionConfig.from_file(cfg_path)
    tok = AutoTokenizer.from_pretrained(a.repo, revision=a.revision)
    types_ = [t for t in ("noul", "choice", "score")]
    labels, ids = {}, {}
    for t in types_:
        lab = cfg.labels.get(t) or (["Yes", "No"] if t == "noul" else [chr(65 + i) for i in range(26)])
        labels[t] = lab
        ids[t] = label_token_ids(tok, tuple(lab))
    V = len(tok)
    rng = np.random.default_rng(a.seed)
    inp, cases, ref = Inputs(), [], []
    for i in range(a.n):
        t = types_[i % 3]
        n = 2 if t == "noul" else 2 + i % 6
        L = (rng.standard_normal(V) * 3).astype(np.float32)
        lid = ids[t][:n]
        T = float(cfg.temperature.get(t, 1.0))
        # reference: option scores = logits of the label tokens (label_token_ids), TemperatureLogitsWarper(T)
        ref.append(_temperature_ref([float(L[j]) for j in lid], T))
        cases.append({"kind": "logits", "L": inp.add(L), "ids": lid, "T": T})
    ours = [r["scores"][0] for r in run_lib(None, {}, cases, inp)]
    cmp = compare(ref, ours, tol_p=TOL_FP32, tol_z=1e-6)
    noul = cfg.noul_options
    fragment = {"logits": {"temperature": {QT[t]: float(cfg.temperature[t]) for t in cfg.temperature if t in QT},
                           "choice": {"map": "tokens", "tokens": [{"label": l, "id": i, "text": tok.decode([i])} for l, i in zip(labels["choice"], ids["choice"])]},
                           "x-labels": labels, "x-noulOptions": noul},
                "x-decision-template": {"present": cfg.chat_template is not None, "sha256": None if cfg.chat_template is None else __import__("hashlib").sha256(cfg.chat_template.encode()).hexdigest()},
                "provenance": {"source": _source("transformers-decision-config", a.repo, a.revision, ["decision.json"] + (["decision_template.jinja"] if cfg.chat_template else []))}}
    rec = fidelity_record(source_format="transformers-decision-config", reference={"impl": "transformers decision: label_token_ids + DecisionConfig.get_logits_processor (TemperatureLogitsWarper)", "repo": "huggingface/transformers", "commit": "31d7392997c13162c58f2e3c6555b5e060d6f60b", "file": "src/transformers/decision/"},
                          inputs={"kind": "synthetic-logits", "n": a.n, "seed": a.seed, "vocab": V}, result=cmp, ours={"reader": "logits readout: L[ids] / T (src/decide.ts fromLogits)"},
                          notes="Covers the score arithmetic only. The Jinja decision template is not translated (spec 3.4: described, kept verbatim under x-decision-template; s1s2 0.1 renders tagged/kev layouts).")
    write_outputs(out, fragment, rec)


def from_lm_rows(a) -> None:
    """Standard One jev-gpu-readout-v2 (.npz): no public artifact was found (formatos-head.md 2.7); runs on a local file."""
    z = np.load(a.npz, allow_pickle=False)
    meta = json.loads(str(z["metadata"]))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mode = meta["mode"]
    if mode == "lm_rows":
        if meta.get("logit_scale") not in (None, 1, 1.0) or meta.get("softcap") not in (None,):
            note = "logit_scale/softcap declared: written as logits.x-logitScale/x-softcap (s1s2 0.1 logits readout has temperature only)"
        else:
            note = ""
        frag = {"logits": {"temperature": {"predicate": meta["temperature"], "choice": meta["temperature"], "score": meta["temperature"]},
                           "choice": {"map": "tokens", "tokens": [{"label": l, "id": int(i), "text": l} for l, i in zip(meta["labels"], meta["token_ids"])]},
                           **({"x-logitScale": meta.get("logit_scale"), "x-softcap": meta.get("softcap")} if note else {})}}
    else:
        W = z["weights"].astype(np.float32); B = z["bias"].astype(np.float32)
        write_safetensors(out / "head.safetensors", {"weight": W, "bias": B})
        frag = {"heads": {"classifier": {"file": pinned(out / "head.safetensors", "heads/head.safetensors"), "types": ["choice"], "kind": "linear", "format": "safetensors", "rows": {"choice": {"start": 0}}, "temperature": meta["temperature"], "read": "last", "source": "standard-one-readout-v2"}}}
        note = ""
    frag["provenance"] = {"source": {"format": "standard-one-readout-v2", "file": str(a.npz), "metadata": meta}}
    rec = fidelity_record(source_format="standard-one-readout-v2", reference={"impl": "none run (no public artifact)"}, inputs={"kind": "none"},
                          result={"n": 0, "pass": None}, ours={}, notes="structure-only conversion; " + note)
    write_outputs(out, frag, rec)


# ---------------------------------------------------------------- Clef joint head
def from_clef(a) -> None:
    import torch
    from safetensors.torch import load_file
    from . import refs
    sg = refs.sglang()
    jsh = refs.clef_cpu(sg.joint_schema_head)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(hub_file(a.repo, a.revision, "joint_head_config.json").read_text())
    st_path = hub_file(a.repo, a.revision, "joint_head.safetensors")
    sd = load_file(str(st_path))
    head = jsh.JointSchemaHead(cfg["hidden_size"], cfg["width"], cfg["routing_layers"], cfg["layers"], cfg["heads"], cfg["feedforward"])
    missing, unexpected = head.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    head = head.float().eval()
    # fp32 copy of the published bf16 weights, for s1s2 (bf16 -> fp32 is exact)
    write_safetensors(out / "joint_head.safetensors", {k: v.float().numpy() for k, v in sd.items()})
    H = cfg["hidden_size"]
    rng = np.random.default_rng(a.seed)
    inp, cases, ref = Inputs(), [], []
    for c in range(a.n):
        T = 40 + 10 * c
        nq = 1 + c % 3
        # a schema region of nq questions, each a span then 2..4 option spans, after a prefix
        pos, qs = 8, []
        for qi in range(nq):
            qspan = (pos, pos + 3); pos += 4
            k = 2 + (c + qi) % 3
            opts = []
            for _ in range(k):
                opts.append((pos, pos + 2)); pos += 3
            qs.append(jsh.LayoutQuestion(qi % 3, qspan, tuple(opts)))
        T = max(T, pos + 4)
        hidden = (rng.standard_normal((T, H)) * 2).astype(np.float32)
        lex_table = (rng.standard_normal((64, H)) * 0.05).astype(np.float32)
        input_ids = rng.integers(0, 64, T)
        with torch.no_grad():
            logits = head.forward_batch(torch.tensor(hidden), torch.tensor(input_ids), [T], [qs], torch.tensor(lex_table))
        flat = logits[0].double().tolist()
        per_q, o = [], 0
        for q in qs:
            per_q.append(flat[o:o + len(q.option_spans)]); o += len(q.option_spans)
        ref.extend(per_q)
        cases.append({"kind": "joint", "hidden": inp.add(hidden), "lexical": inp.add(lex_table[input_ids]),
                      "questions": [{"type": q.question_type, "span": list(q.question_span), "options": [list(s) for s in q.option_spans]} for q in qs]})
    res = run_lib(out / "joint_head.safetensors", {"kind": "joint", "format": "safetensors", "config": cfg}, cases, inp)
    ours = [z for r in res for z in r["scores"]]
    cmp = compare(ref, ours, tol_p=1e-4, tol_z=1e-3)
    fragment = {"heads": {"joint": {"file": pinned(out / "joint_head.safetensors", "heads/joint_head.safetensors"), "types": ["predicate", "choice", "score"], "kind": "joint", "format": "safetensors", "config": cfg, "source": "clef-joint-head"}},
                "encoding": {"kind": "joint-schema", "prompt": "CLEF_PROMPT (sglang systemone/joint_schema.py)", "noulOrder": ["true", "false"], "choiceOrder": "sorted by option id"},
                "provenance": {"source": _source("clef-joint-head", a.repo, a.revision, ["joint_head_config.json", "joint_head.safetensors"], loadMissing=list(missing), loadUnexpected=list(unexpected))}}
    rec = fidelity_record(source_format="clef-joint-head", reference={"impl": "sglang JointSchemaHead.forward_batch (CPU: flash-attention kernel replaced by per-request SDPA, s1s2_heads/refs)", "repo": "sgl-project/sglang", "commit": refs.SGLANG_COMMIT, "file": "python/sglang/srt/layers/joint_schema_head.py"},
                          inputs={"kind": "synthetic-hidden+synthetic-lexical-rows", "n": a.n, "questions": len(ref), "seed": a.seed}, result=cmp, ours={"reader": "loadJointHead + jointScores (fp64 JS)"},
                          notes="Reference in fp32 (weights bf16 -> fp32), ours fp64. The lexical prior needs the LM-head rows of the option tokens: a build must expose them (spec 6.4).")
    write_outputs(out, fragment, rec)


def register(sub: argparse._SubParsersAction) -> None:
    def common(p, repo, rev):
        p.add_argument("--repo", default=repo); p.add_argument("--revision", default=rev)
        p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=64); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("from-kev-headpt", help="Kev head.pt (pointer, fp32) -> head.safetensors + head.bin"); common(p, "jaredpalmer/kev-0.8b", "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"); p.set_defaults(fn=from_kev_headpt)
    p = sub.add_parser("from-gguf-kev", help="GGUF {arch}.decision.type=kev (classifier.out_proj = q|k) -> head.safetensors"); common(p, "ggml-org/Kev-4B-GGUF", "f4624b6bb7072f1bab726a622c1cad3f96030b9e")
    p.add_argument("--file", default="Kev-4B-BF16.gguf"); p.add_argument("--compare-headpt", default="jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"); p.set_defaults(fn=from_gguf_kev)
    p = sub.add_parser("from-alias3-web", help="alias3-web head.bin + norm.bin -> head.bin + build.norm (scale declared)"); common(p, "aliasrobotics/alias3-web", "2ab2e6016a91256fd074144a22ed97d7cea72f6d"); p.set_defaults(fn=from_alias3_web)
    p = sub.add_parser("from-decision-config", help="pplx-decider / d3 decision_config.json + readout.safetensors -> linear head"); common(p, "vllm-sr/d3-mini", "df99bf4ca4418bc6908ee68e9b7479af1f3b8800")
    p.add_argument("--readout-file", default="readout.safetensors"); p.set_defaults(fn=from_decision_config)
    p = sub.add_parser("from-jev", help="JEV judge_config.json + head.safetensors + calibration.json -> linear head"); common(p, "autotrust/AutoTrust-Decider-Mini-9B", "7ac669b8fb476ba926491e3d1c39f9330d68dbed"); p.set_defaults(fn=from_jev)
    p = sub.add_parser("from-transformers-decision", help="transformers decision.json (DecisionConfig) -> logits readout tokens + temperature"); common(p, "ArthurZ/pplx-decider-v1.1-27b", "df876c96295710f8fa8e89ac08fd19f6f63b90c1"); p.set_defaults(fn=from_transformers_decision)
    p = sub.add_parser("from-lm-rows", help="Standard One jev-gpu-readout-v2 .npz (local file) -> logits tokens or linear head")
    p.add_argument("--npz", required=True); p.add_argument("--out", required=True); p.set_defaults(fn=from_lm_rows)
    p = sub.add_parser("from-clef", help="Clef joint_head_config.json + joint_head.safetensors -> joint head"); common(p, "Cloudflare/clef-flash", "22b75507981345858dd152fbd9a6f8c0c4068b0a"); p.set_defaults(fn=from_clef)
    p.set_defaults(n=8)
