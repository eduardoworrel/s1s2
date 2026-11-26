"""vl-graph: graph edits that make the public Qwen3-VL ONNX build compute what the PyTorch model (and the Visual Jev LoRA) computes.

Finding that motivates this file (graph inspection of onnx-community/Qwen3-VL-4B-Instruct-ONNX, cpu and cuda, and of the 2B transformers.js layout):
Qwen3-VL "DeepStack" adds visual features taken from vision blocks 5, 11 and 17 (`deepstack_visual_indexes`), through three extra mergers, to the hidden
state after text layers 0, 1 and 2 at the image positions. The public `vision.onnx` has one merger (49 LayerNorm = 24 blocks x 2 + 1; 98 MatMulNBits =
24 x 4 + 2) and its only output is `image_features`; the public `text.onnx` takes `inputs_embeds`, `attention_mask`, `position_ids` and the cache, and no
Add of layers 0 to 2 reads anything but the layer outputs. DeepStack is therefore not computed by those graphs: they run the backbone without it, a
function different from the one the LoRA was trained on. Three edits, each small and testable without the 4B weights:

  text-deepstack    add inputs `deepstack_0..2` [batch, sequence, hidden] (zero except at image positions) added after layers 0, 1, 2
                    (in the fused graph: into the `mlp.down_proj` output that feeds the next layer's SkipSimplifiedLayerNormalization);
  text-hidden       replace the `logits` output by `hidden`, the final-norm output [batch, sequence, hidden]: the scores of a choice question are
                    hidden_last . embed_tokens[option token], read in JavaScript from the tied embedding (spec section "Image"); removes the lm_head
                    MatMulNBits (200 MB int4 of the int4 file are left unreferenced) and the [sequence, 151936] logits tensor of a 300-token prefix;
  vision-deepstack  add outputs `deepstack_0..2` [num_logical_patches, out_hidden]: the hidden state input of the LayerNorm that opens blocks 6, 12 and
                    18, through reshape(-1, 4 * hidden) -> LayerNorm -> Linear -> GELU -> Linear (weights of `visual.deepstack_merger_list.N`, float).

  s1s2-convert vl-graph text   --in text.onnx --out OUT/text.onnx [--deepstack] [--hidden-output]
  s1s2-convert vl-graph vision --in vision.onnx --base BASE_DIR --out OUT/vision.onnx
The data file of the input is referenced, not copied: put OUT next to a copy (or hard link) of it. New initializers are stored inside the model file.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

DEEPSTACK_TEXT_LAYERS = (0, 1, 2)
DEEPSTACK_VISION_BLOCKS = (5, 11, 17)


def _elem_type(model, name):
    for v in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if v.name == name:
            return v.type.tensor_type.elem_type
    return None


def _prune(model):
    """Drop nodes and initializers no output depends on."""
    need = {o.name for o in model.graph.output}
    keep = []
    for n in reversed(list(model.graph.node)):
        if any(o in need for o in n.output):
            keep.append(n)
            need.update(i for i in n.input if i)
    keep.reverse()
    del model.graph.node[:]
    model.graph.node.extend(keep)
    used = {i for n in keep for i in n.input}
    for ini in [i for i in model.graph.initializer if i.name not in used]:
        model.graph.initializer.remove(ini)
    return model


def text_deepstack(model, layers=DEEPSTACK_TEXT_LAYERS, hidden=None):
    """Add `deepstack_<i>` inputs. The fused decoder keeps the residual stream inside SkipSimplifiedLayerNormalization(residual, branch, gamma);
    the hidden state after layer i is residual + branch (the down_proj output) as read by layer i + 1's input_layernorm (or the final norm)."""
    import onnx
    from onnx import TensorProto, helper
    g = model.graph
    emb = next(i for i in g.input if i.name == "inputs_embeds")
    dt = emb.type.tensor_type.elem_type
    dims = [d.dim_param or d.dim_value for d in emb.type.tensor_type.shape.dim]
    hidden = hidden or dims[-1]
    by_name = {n.name: n for n in g.node}
    for i in layers:
        nxt = by_name.get(f"/model/layers.{i + 1}/input_layernorm/SkipLayerNorm") or by_name.get(f"/model/layers.{i + 1}/final_norm_layernorm/SkipLayerNorm")
        if nxt is None or nxt.op_type != "SkipSimplifiedLayerNormalization":
            raise SystemExit(f"no SkipSimplifiedLayerNormalization after layer {i}: this is not the public graph layout")
        if not re.search(rf"/layers\.{i}/mlp/down_proj/MatMul/output_0$", nxt.input[1]):
            raise SystemExit(f"layer {i + 1} skip input is {nxt.input[1]}, expected the down_proj output of layer {i}")
        name = f"deepstack_{i}"
        g.input.append(helper.make_tensor_value_info(name, dt, ["batch_size", "sequence_length", hidden]))
        out = f"/model/layers.{i}/deepstack/Add/output_0"
        add = helper.make_node("Add", [nxt.input[1], name], [out], name=f"/model/layers.{i}/deepstack/Add")
        pos = list(g.node).index(nxt)
        g.node.insert(pos, add)
        nxt.input[1] = out
    return model


def text_hidden_output(model):
    """Replace output `logits` by `hidden` (the final norm output) and prune the lm_head."""
    from onnx import helper
    g = model.graph
    fin = next((n for n in g.node if n.name.endswith("/final_norm_layernorm/SkipLayerNorm")), None)
    if fin is None:
        raise SystemExit("final_norm_layernorm not found")
    old = next(o for o in g.output if o.name == "logits")
    gname = fin.input[2]
    gam = next(i for i in g.initializer if i.name == gname)
    hid = helper.make_tensor_value_info("hidden", old.type.tensor_type.elem_type, ["batch_size", "sequence_length", int(gam.dims[0])])
    ident = helper.make_node("Identity", [fin.output[0]], ["hidden"], name="/model/hidden_output")
    g.node.append(ident)
    pos = list(g.output).index(old)
    g.output.remove(old)
    g.output.insert(pos, hid)
    return _prune(model)


def _merger_nodes(prefix, x, weights, dt_np, merged, out_name):
    """reshape(-1, merged) -> LayerNorm(merged, eps 1e-6) -> Linear -> GELU (erf, as nn.GELU()) -> Linear. Returns (nodes, initializers)."""
    from onnx import TensorProto, helper, numpy_helper
    inits, nodes = [], []

    def ini(name, arr):
        t = numpy_helper.from_array(np.ascontiguousarray(arr).astype(dt_np), f"{prefix}.{name}")
        inits.append(t)
        return t.name

    shp = numpy_helper.from_array(np.array([-1, merged], dtype=np.int64), f"{prefix}.shape")
    inits.append(shp)
    nodes += [
        helper.make_node("Reshape", [x, shp.name], [f"{prefix}/reshape"], name=f"{prefix}/Reshape"),
        helper.make_node("LayerNormalization", [f"{prefix}/reshape", ini("norm.weight", weights["norm.weight"]), ini("norm.bias", weights["norm.bias"])],
                         [f"{prefix}/norm"], name=f"{prefix}/LayerNorm", axis=-1, epsilon=1e-6),
        helper.make_node("MatMul", [f"{prefix}/norm", ini("fc1.weight_t", weights["linear_fc1.weight"].T)], [f"{prefix}/fc1_mm"], name=f"{prefix}/fc1_MatMul"),
        helper.make_node("Add", [f"{prefix}/fc1_mm", ini("fc1.bias", weights["linear_fc1.bias"])], [f"{prefix}/fc1"], name=f"{prefix}/fc1_Add"),
        helper.make_node("Gelu", [f"{prefix}/fc1"], [f"{prefix}/act"], name=f"{prefix}/Gelu", approximate="none", domain=""),
        helper.make_node("MatMul", [f"{prefix}/act", ini("fc2.weight_t", weights["linear_fc2.weight"].T)], [f"{prefix}/fc2_mm"], name=f"{prefix}/fc2_MatMul"),
        helper.make_node("Add", [f"{prefix}/fc2_mm", ini("fc2.bias", weights["linear_fc2.bias"])], [out_name], name=f"{prefix}/fc2_Add"),
    ]
    return nodes, inits


def vision_taps(model, blocks=DEEPSTACK_VISION_BLOCKS):
    """Names of the tensors holding the hidden state after each vision block in `blocks`: the input of the first LayerNormalization of the next block
    (LayerNormalization nodes in graph order: two per block, then the merger's)."""
    lns = [n for n in model.graph.node if n.op_type == "LayerNormalization"]
    if len(lns) % 2 != 1:
        raise SystemExit(f"{len(lns)} LayerNormalization in the vision graph: expected 2 per block + 1 for the merger")
    n_blocks = (len(lns) - 1) // 2
    if max(blocks) + 1 >= n_blocks:
        raise SystemExit(f"block {max(blocks)} is not followed by another block in a graph of {n_blocks}")
    return [lns[2 * (b + 1)].input[0] for b in blocks]


def vision_deepstack(model, weights, blocks=DEEPSTACK_VISION_BLOCKS):
    """weights[j] = dict of the six tensors of `visual.deepstack_merger_list.j` (norm.weight, norm.bias, linear_fc1.weight/bias, linear_fc2.weight/bias).
    The merger runs in the dtype of the tapped tensor (the cuda build normalises in float32) and its output is cast to the dtype of `image_features`."""
    from onnx import TensorProto, helper
    g = model.graph
    taps = vision_taps(model, blocks)
    out_dt = next(o for o in g.output if o.name == "image_features").type.tensor_type.elem_type
    out_hidden = int(weights[0]["linear_fc2.weight"].shape[0])
    merged = int(weights[0]["linear_fc1.weight"].shape[1])
    for j, tap in enumerate(taps):
        dt = _elem_type(model, tap) or out_dt
        dt_np = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}[dt]
        last = f"deepstack_{j}" if dt == out_dt else f"/deepstack_merger.{j}/out_f"
        nodes, inits = _merger_nodes(f"/deepstack_merger.{j}", tap, weights[j], dt_np, merged, last)
        if dt != out_dt:
            nodes.append(helper.make_node("Cast", [last], [f"deepstack_{j}"], name=f"/deepstack_merger.{j}/Cast", to=out_dt))
        g.node.extend(nodes)
        g.initializer.extend(inits)
        g.output.append(helper.make_tensor_value_info(f"deepstack_{j}", out_dt, ["num_logical_patches", out_hidden]))
    return model, taps


def read_merger_weights(base_dir, n=3):
    """The deepstack mergers of a base checkpoint (safetensors dir), as float32 numpy: about 55 MB each."""
    from safetensors import safe_open
    d = Path(base_dir)
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    for j in range(n):
        w = {}
        for k in ("norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight", "linear_fc2.bias"):
            key = f"model.visual.deepstack_merger_list.{j}.{k}"
            with safe_open(str(d / idx[key]), "pt") as f:
                w[k] = f.get_tensor(key).float().numpy()
        out.append(w)
    return out


def main(argv=None):
    import onnx
    ap = argparse.ArgumentParser(prog="s1s2-convert vl-graph", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="what", required=True)
    t = sub.add_parser("text")
    t.add_argument("--in", dest="inp", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--deepstack", action="store_true")
    t.add_argument("--hidden-output", action="store_true")
    v = sub.add_parser("vision")
    v.add_argument("--in", dest="inp", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--base", required=True)
    a = ap.parse_args(argv)
    m = onnx.load(a.inp, load_external_data=False)
    if a.what == "text":
        if a.deepstack:
            text_deepstack(m)
        if a.hidden_output:
            text_hidden_output(m)
    else:
        m, taps = vision_deepstack(m, read_merger_weights(a.base))
        print("taps", taps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, a.out)
    print(json.dumps({"out": a.out, "inputs": [i.name for i in m.graph.input if not i.name.startswith("past")], "outputs": [o.name for o in m.graph.output if not o.name.startswith("present")]}))


if __name__ == "__main__":
    main(sys.argv[1:])
