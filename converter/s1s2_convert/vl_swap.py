"""vl-swap: fuse a LoRA adapter into the int4 text decoder of a public Qwen3-VL ONNX build by swapping weight blocks.

Why. The text decoder of `onnx-community/Qwen3-VL-4B-Instruct-ONNX` (ONNX Runtime GenAI layout: embedding.onnx, vision.onnx, text.onnx) is a
graph whose weights are int4 `MatMulNBits` initializers (RTN, block 32, symmetric) in one external-data file. A LoRA on the language tower
(Visual Jev: r=16, alpha=32 on q,k,v,o,gate,up,down) changes 252 of its 253 matrices. Re-exporting the 4B needs 23 to 34 GiB of RAM (the GenAI builder
holds the fp16 graph in memory); this command instead keeps the public graph and rewrites, one tensor at a time, only the initializers the LoRA touches:

    W' = float32(W) + (alpha / r) * B @ A        (W from the base safetensors, read through mmap, in blocks of rows)
    -> round to the cast chain of the build -> ORT `DefaultWeightOnlyQuantizer.qbits_block_quant` (4 bits, block 32, symmetric)
    -> written in place at the same (offset, length) of the data file of a copy of the template.

The mechanism (tensor-by-tensor requantization written in place into the data file of an existing build of the same architecture) was validated byte for byte against the GenAI builder on a 0.8B model of another family; this file has the rules of this graph: q, k, v, gate and up are
separate matrices (no fused qkv); the final norm is `model.layers.36.final_norm_layernorm`; the RMSNorm weight is the plain weight (Qwen3 is not `1 + w`);
`lm_head` is the tied embedding (the LoRA does not touch it).

Oracle. The LoRA never touches `lm_head`, which is the int4 quantization of `embed_tokens`. `--verify` (no LoRA) recomputes every initializer the swap
would write, plus `lm_head` and the norms, and compares with the template byte for byte. Which cast chain the public build used is not documented: the
command tries the candidate chains on the first rows of `lm_head`, reports which one is byte-equal (or none), and uses it. If none is, the swap still runs
with the default chain and the report says `oracle: false`: the fidelity then rests on the logit parity against PyTorch, not on byte equality.

After the swap, `--split` rewrites the external-data locations into pieces of at most 1.9 GiB (onnxruntime-web reads up to 2 GB per file).

  s1s2-convert vl-swap --template DIR --base DIR --adapter adapter_model.safetensors --out DIR [--split] [--limit-layers N]
  s1s2-convert vl-swap --template DIR --base DIR --verify [--report r.json]
DIR(template) holds text.onnx and text.onnx.data (the public `cpu-int4-rtn-block-32` or `cuda-int4-rtn-block-32` folder); DIR(base) the
`Qwen/Qwen3-VL-4B-Instruct` snapshot (config + model.safetensors.index.json + shards).
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .util import log

SPLIT_MAX = int(1.9 * 1024 ** 3)
LM = "model.language_model."
ONNX_FLOAT, ONNX_FLOAT16 = 1, 10
Q_PROJ = {"attn.q_proj": "self_attn.q_proj", "attn.k_proj": "self_attn.k_proj", "attn.v_proj": "self_attn.v_proj", "attn.o_proj": "self_attn.o_proj",
          "mlp.gate_proj": "mlp.gate_proj", "mlp.up_proj": "mlp.up_proj", "mlp.down_proj": "mlp.down_proj"}
NORMS = {"input_layernorm.weight": "input_layernorm.weight", "post_attention_layernorm.weight": "post_attention_layernorm.weight",
         "attn.q_norm.layernorm.weight": "self_attn.q_norm.weight", "attn.k_norm.layernorm.weight": "self_attn.k_norm.weight"}


def rule(name, n_layers=36):
    """Initializer name of the public text.onnx -> (kind, base key).

    kind: "q4" / "q4s" (packed int4 / scales of a LoRA target), "lm4" / "lm4s" (lm_head, oracle only), "norm" (fp32 RMSNorm weight), "const" (rotary
    table), None (not recognised: the command stops). The final norm carries the index `n_layers` in the public graph."""
    if name == "model.inv_freq":
        return ("const", None)
    if name.startswith("lm_head.MatMul.weight_"):
        return ("lm4" if name.endswith("_Q4") else "lm4s", LM + "embed_tokens.weight") if name.endswith(("_Q4", "_scales")) else (None, None)
    if name == f"model.layers.{n_layers}.final_norm_layernorm.weight":
        return ("norm", LM + "norm.weight")
    m = re.match(r"^model\.layers\.(\d+)\.(.+)$", name)
    if not m or int(m.group(1)) >= n_layers:
        return (None, None)
    L, rest = int(m.group(1)), m.group(2)
    if rest in NORMS:
        return ("norm", f"{LM}layers.{L}.{NORMS[rest]}")
    mm = re.match(r"^(.+)\.MatMul\.weight_(Q4|scales)$", rest)
    if mm and mm.group(1) in Q_PROJ:
        return ("q4" if mm.group(2) == "Q4" else "q4s", f"{LM}layers.{L}.{Q_PROJ[mm.group(1)]}.weight")
    return (None, None)


def _quantizer():
    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, DefaultWeightOnlyQuantizer
    return DefaultWeightOnlyQuantizer(DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=4, quant_format=QuantFormat.QOperator))


def quantize_q4(w, chain, scale_dtype):
    """w: float32 torch tensor [N, K] (rows = output features). -> (packed uint8 [N, K/32, 16], scales [N, K/32] in scale_dtype).

    chain "f32": quantize the float32 values, scales in float32. chain "f16": round the weight to float16 first (the cast of an fp16 build), the
    scales come out in float16. The scales are then cast to the dtype the template stores (cast of a float16 scale to float32 is exact)."""
    import torch
    q = _quantizer()
    x = w.to(torch.float16) if chain == "f16" else w.to(torch.float32)
    packed, scales, _ = q.qbits_block_quant(np.ascontiguousarray(x.numpy().T))
    return packed, scales.astype(scale_dtype)


def reference_q4(w, block=32):
    """Independent numpy reading of the same quantizer (symmetric, zero point 8): per block of 32 along K, scale = (signed value of the largest
    magnitude) / -8, q = clip(floor(x / scale + 8.5), 0, 15); two values per byte, the first in the low nibble. Used by the tests as the reference
    of the ORT function. w: float32 [N, K] -> (packed [N, K/block, block/2], scales [N, K/block])."""
    n, k = w.shape
    b = w.reshape(n, k // block, block).astype(np.float32)
    idx = np.abs(b).argmax(-1)
    mx = np.take_along_axis(b, idx[..., None], -1)[..., 0]
    scale = mx / np.float32(-8.0)
    rec = np.where(scale != 0, np.float32(1.0) / np.where(scale != 0, scale, 1), 0).astype(np.float32)
    q = np.clip(np.floor(b * rec[..., None] + np.float32(8.5)), 0, 15).astype(np.uint8)
    return (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8), scale.astype(np.float32)


def dequant_q4(packed, scales):
    """Packed [N, K/32, 16] + scales [N, K/32] -> float32 [N, K] (what MatMulNBits multiplies with)."""
    lo, hi = packed & 15, packed >> 4
    q = np.stack([lo, hi], -1).reshape(packed.shape[0], packed.shape[1], 32).astype(np.float32)
    return ((q - 8.0) * scales.astype(np.float32)[..., None]).reshape(packed.shape[0], -1)


class Source:
    """Rows of base weights (safetensors, mmap) with the LoRA delta added in float32."""

    def __init__(self, base_dir, adapter=None):
        from safetensors import safe_open
        self._open = safe_open
        self.dir = Path(base_dir)
        idx = self.dir / "model.safetensors.index.json"
        self.map = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
        if self.map is None:
            with safe_open(str(self.dir / "model.safetensors"), "pt") as f:
                self.map = {k: "model.safetensors" for k in f.keys()}
        self.files, self.lora, self.scale, self.used = {}, {}, 1.0, set()
        if adapter:
            from .fuse_lora import read_adapter, target_keys
            pairs = read_adapter(adapter)
            cfg = json.loads((Path(adapter).parent / "adapter_config.json").read_text()) if (Path(adapter).parent / "adapter_config.json").exists() else None
            self.scale = float(cfg["lora_alpha"]) / float(cfg["r"]) if cfg else 2.0
            self.lora = target_keys(pairs, set(self.map))

    def _slice(self, key):
        f = self.map[key]
        if f not in self.files:
            self.files[f] = self._open(str(self.dir / f), "pt")
        return self.files[f].get_slice(key)

    def shape(self, key):
        return self._slice(key).get_shape()

    def rows(self, key, a, b):
        w = self._slice(key)[a:b].float()
        if key in self.lora:
            ab = self.lora[key]
            w = w + (ab["B"][a:b].float() @ ab["A"].float()) * self.scale
            self.used.add(key)
        return w


def chunked_q4(src, key, chain, scale_dtype, block_rows=16384, only_rows=None):
    n = src.shape(key)[0] if only_rows is None else only_rows
    pk, sc = [], []
    for a in range(0, n, block_rows):
        p, s = quantize_q4(src.rows(key, a, min(n, a + block_rows)), chain, scale_dtype)
        pk.append(p), sc.append(s)
    return np.concatenate(pk, 0), np.concatenate(sc, 0)


def load_template(tdir):
    import onnx
    m = onnx.load(str(Path(tdir) / "text.onnx"), load_external_data=False)
    ini = {}
    for i in m.graph.initializer:
        ex = {e.key: e.value for e in i.external_data}
        ini[i.name] = {"dtype": i.data_type, "dims": list(i.dims), "loc": ex["location"], "off": int(ex["offset"]), "len": int(ex["length"])}
    return m, ini


def probe_chain(src, template_data, ini, log_fn=log):
    """Which cast chain reproduces the template's lm_head? Tests the first 2048 rows (64 blocks of 32 rows of K are per row, so rows are independent)."""
    q4, qs = ini["lm_head.MatMul.weight_Q4"], ini["lm_head.MatMul.weight_scales"]
    n_probe = min(2048, q4["dims"][0])
    sdt = np.float32 if qs["dtype"] == ONNX_FLOAT else np.float16
    key = LM + "embed_tokens.weight"
    k_blocks = q4["dims"][1]
    with open(template_data, "rb") as f:
        f.seek(q4["off"]); ref_q = f.read(n_probe * k_blocks * 16)
        f.seek(qs["off"]); ref_s = f.read(n_probe * k_blocks * np.dtype(sdt).itemsize)
    out = {}
    for chain in ("f32", "f16"):
        p, s = chunked_q4(src, key, chain, sdt, only_rows=n_probe)
        out[chain] = {"q4_equal": p.tobytes() == ref_q, "scales_equal": s.tobytes() == ref_s,
                      "q4_bytes_different": int((np.frombuffer(ref_q, np.uint8) != p.reshape(-1)).sum())}
        log_fn("vl-swap", f"probe lm_head chain {chain}: {out[chain]}")
    ok = [c for c in out if out[c]["q4_equal"] and out[c]["scales_equal"]]
    return (ok[0] if ok else None), out


def norm_bytes(src, key):
    return src._slice(key)[:].float().numpy().astype(np.float32).tobytes()


def run(template, base, out=None, adapter=None, verify=False, limit_layers=None, chain=None, report=None, split=False, block_rows=16384):
    t0 = time.time()
    template, base = Path(template), Path(base)
    m, ini = load_template(template)
    n_layers = max(int(r.group(1)) for r in (re.match(r"^model\.layers\.(\d+)\.", n) for n in ini) if r)  # final norm has index n_layers
    src = Source(base, adapter)
    data_path = template / "text.onnx.data"
    rep = {"mode": "verify" if verify else "swap", "template": str(template), "n_initializers": len(ini), "layers": n_layers, "lora_scale": src.scale if adapter else None,
           "different": [], "equal": 0, "unrecognised": [], "swapped": 0}
    plan = []
    for name, d in sorted(ini.items(), key=lambda kv: kv[1]["off"]):
        kind, key = rule(name, n_layers)
        if kind is None:
            rep["unrecognised"].append(name)
            continue
        if limit_layers is not None:
            mm = re.match(r"^model\.layers\.(\d+)\.", name)
            if mm and int(mm.group(1)) >= limit_layers:
                continue
        plan.append((name, kind, key, d))
    if rep["unrecognised"]:
        log("vl-swap", f"{len(rep['unrecognised'])} initializers match no rule, e.g. {rep['unrecognised'][:4]}; stopping")
        if report:
            Path(report).write_text(json.dumps(rep, indent=1))
        raise SystemExit(3)
    if verify and adapter:
        raise SystemExit("--verify is the no-LoRA check against the template: drop --adapter")
    scale_dt = np.float32 if ini["model.layers.0.attn.q_proj.MatMul.weight_scales"]["dtype"] == ONNX_FLOAT else np.float16
    picked, probe = probe_chain(src, data_path, ini) if chain is None else (chain, {})
    rep["probe"], rep["oracle"] = probe, picked is not None
    chain = picked or "f32"
    rep["chain"] = chain
    log("vl-swap", f"cast chain {chain} (oracle {'yes' if picked else 'NO: no candidate reproduces lm_head byte for byte'}); scales {np.dtype(scale_dt).name}")
    if not verify:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for p in template.iterdir():
            if p.is_file() and p.name not in ("text.onnx.data", "text.onnx"):
                shutil.copyfile(p, out / p.name)
        shutil.copyfile(template / "text.onnx", out / "text.onnx")
        shutil.copyfile(data_path, out / "text.onnx.data")
        wf = open(out / "text.onnx.data", "r+b")
    rf = open(data_path, "rb")
    cache = {}
    for name, kind, key, d in plan:
        if kind == "const":
            continue
        if kind == "norm":
            buf = norm_bytes(src, key)
        else:
            if kind in ("lm4", "lm4s") and not verify:
                continue  # never swapped: the LoRA does not touch the tied embedding
            if key not in cache:
                cache.clear()
                cache[key] = chunked_q4(src, key, chain, scale_dt, block_rows)
            p, s = cache[key]
            buf = (p if kind in ("q4", "lm4") else s).tobytes()
        if len(buf) != d["len"]:
            raise SystemExit(f"{name}: computed {len(buf)} bytes, template has {d['len']} (dims {d['dims']})")
        if verify or kind == "norm":  # norms are not swapped; they are the second oracle (bf16 -> fp32 is exact)
            rf.seek(d["off"])
            if rf.read(d["len"]) == buf:
                rep["equal"] += 1
            else:
                rep["different"].append(name)
        if not verify and kind in ("q4", "q4s"):
            wf.seek(d["off"]); wf.write(buf); rep["swapped"] += 1
    rf.close()
    if not verify:
        wf.flush(); wf.close()
        if adapter and not limit_layers:
            if src.used != set(src.lora):
                raise SystemExit("LoRA deltas not applied to: " + str(sorted(set(src.lora) - src.used))[:300])
        rep["lora_modules_applied"] = len(src.used)
        if split:
            rep["split"] = split_external_data(out / "text.onnx", out / "text.onnx.data")
    rep["seconds"] = round(time.time() - t0, 1)
    rep["n_different"] = len(rep["different"])
    log("vl-swap", json.dumps({k: v for k, v in rep.items() if k not in ("probe", "different")}) + f" different={rep['different'][:4]}")
    if report:
        Path(report).write_text(json.dumps(rep, indent=1) + "\n")
    return rep


def split_external_data(model_path, data_path, max_bytes=SPLIT_MAX):
    """Rewrite the external-data entries of `model_path` so that no data file passes `max_bytes`: pieces `<data>`, `<data>_1`, ... Bytes are copied
    in streaming (no full load), initializers keep their relative alignment inside a piece. Returns the list of pieces."""
    import onnx
    model_path, data_path = Path(model_path), Path(data_path)
    m = onnx.load(str(model_path), load_external_data=False)
    ents = []
    for i in m.graph.initializer:
        ex = {e.key: e for e in i.external_data}
        if "location" in ex:
            ents.append((int(ex["offset"].value), int(ex["length"].value), ex, i.name))
    ents.sort(key=lambda t: t[0])
    pieces, cur = [], []
    for e in ents:
        if cur and e[0] + e[1] - cur[0][0] > max_bytes:
            pieces.append(cur)
            cur = []
        cur.append(e)
    if cur:
        pieces.append(cur)
    names = [data_path.name + ("" if k == 0 else f"_{k}") for k in range(len(pieces))]
    tmp = [data_path.with_name(n + ".part") for n in names]
    with open(data_path, "rb") as src:
        for k, grp in enumerate(pieces):
            start, end = grp[0][0], grp[-1][0] + grp[-1][1]
            with open(tmp[k], "wb") as dst:
                src.seek(start)
                left = end - start
                while left:
                    b = src.read(min(left, 1 << 24))
                    if not b:
                        raise SystemExit("data file shorter than the offsets of text.onnx")
                    dst.write(b); left -= len(b)
            for off, ln, ex, _ in grp:
                ex["location"].value = names[k]
                ex["offset"].value = str(off - start)
    data_path.unlink()
    for k, p in enumerate(tmp):
        p.rename(data_path.with_name(names[k]))
    onnx.save(m, str(model_path))
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(prog="s1s2-convert vl-swap", description=__doc__.split("\n")[0])
    ap.add_argument("--template", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--out")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--limit-layers", type=int)
    ap.add_argument("--chain", choices=["f32", "f16"], help="force the cast chain (default: probe on lm_head)")
    ap.add_argument("--report")
    ap.add_argument("--block-rows", type=int, default=16384)
    a = ap.parse_args(argv)
    if not a.verify and not (a.out and a.adapter):
        ap.error("a swap needs --adapter and --out (or use --verify)")
    run(a.template, a.base, a.out, a.adapter, a.verify, a.limit_layers, a.chain, a.report, a.split, a.block_rows)


if __name__ == "__main__":
    main(sys.argv[1:])
