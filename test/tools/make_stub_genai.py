#!/usr/bin/env python3
"""Tiny synthetic GenAI-shaped graphs for s1s2 tests (onnx helper only; no model weights).

  stub   --hidden H --vocab V --out DIR [--head-size D]
         A weightless stand-in for a genai build of a real model (same hidden size and vocabulary), so S1S2.load can be
         exercised with the real tokenizer and a real manifest without the multi-GB graph. Inputs input_ids, attention_mask, a KV-like
         past_key_values.0.key (present = concat(past, id)); outputs logits[1,S,V] = (id+1) * w[v] and hidden_states[1,S,H] = (id+1) * u[h] (w, u in model.onnx.data).
         Writes model.onnx, model.onnx.data, genai_config.json.

  embeds --hidden H --vocab V --out DIR [--layers L] [--seed N]
         Two equivalent decoders over the same fp16 embedding table E[V,H]:
           ids/model.onnx     input_ids -> Gather(E) -> tail -> hidden_states, logits (+ a KV-like running state)
           embeds/model.onnx  inputs_embeds (fp16)    -> tail -> hidden_states, logits (same state)
         plus embed_tokens.fp16.bin (E as little-endian fp16, row-major [V,H], as the model cards' build scripts
         write it). The tail mixes tokens through a running-sum state (past_key_values.0.key / present.0.key), so the
         chunked prefill and the state hand-off are exercised too.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def save(model, d, data_name="model.onnx.data"):
    d.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(d / "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data_name, size_threshold=64)


def genai_config(d, hidden, vocab, head_size, inputs):
    cfg = {"model": {"decoder": {"filename": "model.onnx", "hidden_size": hidden, "head_size": head_size, "inputs": inputs,
                                 "outputs": {"logits": "logits"}, "num_hidden_layers": 1}, "vocab_size": vocab, "type": "stub"}}
    (d / "genai_config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def stub(a):
    H, V = a.hidden, a.vocab
    w = (np.arange(V, dtype=np.float32) % 97) / 97.0
    u = (np.arange(H, dtype=np.float32) % 13) / 13.0
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["idf"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["idf", "one"], ["idp"]),
        helper.make_node("Unsqueeze", ["idp", "ax2"], ["id3"]),          # [1,S,1]
        helper.make_node("Mul", ["id3", "w"], ["logits"]),              # [1,S,V]
        helper.make_node("Mul", ["id3", "u"], ["hidden_states"]),       # [1,S,H]
        helper.make_node("Unsqueeze", ["id3", "ax1"], ["id4"]),          # [1,1,S,1]
        helper.make_node("Concat", ["past_key_values.0.key", "id4"], ["present.0.key"], axis=2),  # KV-like state
    ]
    g = helper.make_graph(nodes, "stub",
                          [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "S"]),
                           helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "T"]),
                           helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, [1, 1, "P", 1])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "S", V]),
                           helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, "S", H]),
                           helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, [1, 1, "T", 1])],
                          [numpy_helper.from_array(w, "w"), numpy_helper.from_array(u, "u"),
                           numpy_helper.from_array(np.array(1.0, np.float32), "one"), numpy_helper.from_array(np.array([2], np.int64), "ax2"),
                           numpy_helper.from_array(np.array([1], np.int64), "ax1")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    d = Path(a.out)
    save(m, d)
    genai_config(d, H, V, a.head_size, {"input_ids": "input_ids", "attention_mask": "attention_mask"})


def embeds(a):
    H, V, L = a.hidden, a.vocab, a.layers
    rng = np.random.default_rng(a.seed)
    E = rng.standard_normal((V, H)).astype(np.float16)
    Ws = [(rng.standard_normal((H, H)) / np.sqrt(H)).astype(np.float32) for _ in range(L)]
    lm = (rng.standard_normal((H, V)) / np.sqrt(H)).astype(np.float32)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    E.astype("<f2").tofile(out / "embed_tokens.fp16.bin")

    def tail(x16):
        """x16: fp16 [1,S,H] -> nodes producing hidden_states, logits, present.0.key from past_key_values.0.key."""
        n = [helper.make_node("Cast", [x16], ["x0"], to=TensorProto.FLOAT)]
        x = "x0"
        for i in range(L):
            n += [helper.make_node("MatMul", [x, f"W{i}"], [f"m{i}"]), helper.make_node("Tanh", [f"m{i}"], [f"t{i}"]),
                  helper.make_node("Add", [x, f"t{i}"], [f"x{i + 1}"])]
            x = f"x{i + 1}"
        # KV-like state: present = concat(past, x) along the sequence axis; every position also sees the sum of the past
        n += [helper.make_node("Unsqueeze", [x, "ax1"], ["x4"]),                                  # [1,1,S,H]
              helper.make_node("Concat", ["past_key_values.0.key", "x4"], ["present.0.key"], axis=2),
              helper.make_node("ReduceSum", ["past_key_values.0.key", "ax2"], ["psum4"], keepdims=1),  # [1,1,1,H] (0 if empty)
              helper.make_node("Squeeze", ["psum4", "ax1"], ["past3"]),                           # [1,1,H]
              helper.make_node("CumSum", [x, "ax1s"], ["cum"]),                                   # causal mix inside the chunk
              helper.make_node("Add", ["cum", "past3"], ["mix"]),
              helper.make_node("Mul", ["mix", "scale"], ["mixs"]),
              helper.make_node("Add", [x, "mixs"], ["hpre"]),
              helper.make_node("LayerNormalization", ["hpre", "gamma", "beta"], ["hidden_states"], axis=-1),
              helper.make_node("MatMul", ["hidden_states", "lm"], ["logits"])]
        return n

    inits = [numpy_helper.from_array(W, f"W{i}") for i, W in enumerate(Ws)] + [
        numpy_helper.from_array(lm, "lm"), numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([2], np.int64), "ax2"),
        numpy_helper.from_array(np.array(1, np.int64), "ax1s"), numpy_helper.from_array(np.array(0.05, np.float32), "scale"),
        numpy_helper.from_array(np.ones(H, np.float32), "gamma"), numpy_helper.from_array(np.zeros(H, np.float32), "beta")]
    common_in = [helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]),
                 helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, ["batch_size", 1, "past_sequence_length", H])]
    outs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "sequence_length", V]),
            helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, ["batch_size", "sequence_length", H]),
            helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, ["batch_size", 1, "total_sequence_length", H])]

    g_ids = helper.make_graph([helper.make_node("Gather", ["E", "input_ids"], ["emb"], axis=0)] + tail("emb"), "ids",
                              [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"])] + common_in,
                              outs, [numpy_helper.from_array(E, "E")] + inits)
    g_emb = helper.make_graph(tail("inputs_embeds"), "embeds",
                              [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, ["batch_size", "sequence_length", H])] + common_in,
                              outs, inits)
    for name, g, inputs in (("ids", g_ids, {"input_ids": "input_ids"}), ("embeds", g_emb, {"inputs_embeds": "inputs_embeds"})):
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
        save(m, out / name)
        genai_config(out / name, H, V, H, {**inputs, "attention_mask": "attention_mask", "past_key_names": "past_key_values.%d.key"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["stub", "embeds"])
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--head-size", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (stub if a.kind == "stub" else embeds)(a)
