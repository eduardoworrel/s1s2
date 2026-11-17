#!/usr/bin/env python3
"""Tiny com.microsoft::GatedDeltaNet graph exactly as the ONNX Runtime GenAI 0.19 builder emits it for Qwen3.5 linear
attention on WebGPU (onnxruntime_genai/models/builders/qwen3_5/text.py make_gated_delta_net_layer, dense layout):
packed QKV in input 0, inputs 1-3 (key, value, cu_seqlens) EMPTY, decay/beta cast to float, initial_state, a_log,
dt_bias; attributes update_rule=gated_delta, scale=0, gate_activation=qwen, beta_activation=sigmoid, qk_l2_norm=1,
chunk_size=64. Writes gdn.onnx (weights inline) and gdn_ref.json: random inputs plus the expected outputs from an
independent numpy implementation of the recurrence in docs/ContribOperators.md (v1.31.1):

    g = -exp(a_log) * softplus(decay + dt_bias); b = sigmoid(beta); q, k L2-normalized per head; scale = 1/sqrt(dk)
    S_t = exp(g_t) S_{t-1} + k_t (b_t (v_t - exp(g_t) S_{t-1}^T k_t))^T ;  o_t = scale * S_t^T q_t
    state V-major [B, Hv, dv, dk]; value head h uses query/key head h // (Hv / Hq).
Usage: python make_gdn_graph.py --out DIR [--hq 2 --hv 4 --dk 8 --dv 8 --seq 5 --seed 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def reference(qkv, decay, beta, s0, a_log, dt_bias, hq, hv, dk, dv):
    B, S, _ = qkv.shape
    q = qkv[..., : hq * dk].reshape(B, S, hq, dk).astype(np.float64)
    k = qkv[..., hq * dk: 2 * hq * dk].reshape(B, S, hq, dk).astype(np.float64)
    v = qkv[..., 2 * hq * dk:].reshape(B, S, hv, dv).astype(np.float64)
    l2 = lambda x: x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)  # noqa: E731
    q, k = l2(q), l2(k)
    g = -np.exp(a_log.astype(np.float64)) * np.log1p(np.exp(decay.astype(np.float64) + dt_bias))
    b = 1.0 / (1.0 + np.exp(-beta.astype(np.float64)))
    scale = 1.0 / np.sqrt(dk)
    St = s0.astype(np.float64).transpose(0, 1, 3, 2).copy()  # [B, Hv, dk, dv]
    out = np.zeros((B, S, hv, dv))
    r = hv // hq
    for bb in range(B):
        for t in range(S):
            for h in range(hv):
                kk, qq = k[bb, t, h // r], q[bb, t, h // r]
                e = np.exp(g[bb, t, h])
                Sp = e * St[bb, h]
                delta = b[bb, t, h] * (v[bb, t, h] - Sp.T @ kk)
                St[bb, h] = Sp + np.outer(kk, delta)
                out[bb, t, h] = scale * St[bb, h].T @ qq
    return out, St.transpose(0, 1, 3, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, d in (("hq", 2), ("hv", 4), ("dk", 8), ("dv", 8), ("seq", 5), ("seed", 0)):
        ap.add_argument(f"--{k}", type=int, default=d)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hq, hv, dk, dv, S = a.hq, a.hv, a.dk, a.dv, a.seq
    P = 2 * hq * dk + hv * dv
    rng = np.random.default_rng(a.seed)
    qkv = rng.standard_normal((1, S, P)).astype(np.float16)
    decay = rng.standard_normal((1, S, hv)).astype(np.float16)
    beta = rng.standard_normal((1, S, hv)).astype(np.float16)
    s0 = (0.1 * rng.standard_normal((1, hv, dv, dk))).astype(np.float32)
    a_log = (0.3 * rng.standard_normal(hv)).astype(np.float32)
    dt_bias = (0.3 * rng.standard_normal(hv)).astype(np.float32)

    name = "/model/layers.0/linear_attn/GatedDeltaNet"
    nodes = [
        helper.make_node("Cast", ["decay"], ["decay_f"], name=f"{name}/decay_fp32/Cast", to=TensorProto.FLOAT),
        helper.make_node("Cast", ["beta"], ["beta_f"], name=f"{name}/beta_fp32/Cast", to=TensorProto.FLOAT),
        helper.make_node("GatedDeltaNet", ["qkv", "", "", "", "decay_f", "beta_f", "past.0.recurrent", "a_log", "dt_bias"],
                         ["out", "present.0.recurrent"], name=name, domain="com.microsoft",
                         update_rule="gated_delta", scale=0.0, gate_activation="qwen", beta_activation="sigmoid",
                         qk_l2_norm=1, chunk_size=64),
    ]
    g = helper.make_graph(
        nodes, "gdn",
        [helper.make_tensor_value_info("qkv", TensorProto.FLOAT16, [1, "S", P]),
         helper.make_tensor_value_info("decay", TensorProto.FLOAT16, [1, "S", hv]),
         helper.make_tensor_value_info("beta", TensorProto.FLOAT16, [1, "S", hv]),
         helper.make_tensor_value_info("past.0.recurrent", TensorProto.FLOAT, [1, hv, dv, dk])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, "S", hv, dv]),
         helper.make_tensor_value_info("present.0.recurrent", TensorProto.FLOAT, [1, hv, dv, dk])],
        [helper.make_tensor("a_log", TensorProto.FLOAT, [hv], a_log.tolist()),
         helper.make_tensor("dt_bias", TensorProto.FLOAT, [hv], dt_bias.tolist())])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=10)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx.save_model(m, str(out / "gdn.onnx"))
    o, sf = reference(qkv, decay, beta, s0, a_log, dt_bias, hq, hv, dk, dv)
    ref = {"dims": {"hq": hq, "hv": hv, "dk": dk, "dv": dv, "seq": S, "packed": P},
           "inputs": {"qkv": qkv.astype(np.float32).ravel().tolist(), "decay": decay.astype(np.float32).ravel().tolist(),
                      "beta": beta.astype(np.float32).ravel().tolist(), "past.0.recurrent": s0.ravel().tolist()},
           "expected": {"out": o.ravel().tolist(), "present.0.recurrent": sf.ravel().tolist()}}
    (out / "gdn_ref.json").write_text(json.dumps(ref))
    print(f"wrote {out / 'gdn.onnx'} and gdn_ref.json (packed {P}, seq {S}, hq {hq}, hv {hv})")


if __name__ == "__main__":
    main()
