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


def _subgraph_uses(node):
    """Every name read inside the subgraphs of `node` (Loop / If bodies read tensors of the enclosing graph without listing them as inputs)."""
    used = set()
    for a in node.attribute:
        graphs = [a.g] if a.HasField("g") else list(a.graphs)
        for sg in graphs:
            for n in sg.node:
                used.update(i for i in n.input if i)
                used.update(_subgraph_uses(n))
    return used


def _prune(model):
    """Drop nodes and initializers no output depends on (subgraph reads count as uses)."""
    need = {o.name for o in model.graph.output}
    keep = []
    for n in reversed(list(model.graph.node)):
        if any(o in need for o in n.output):
            keep.append(n)
            need.update(i for i in n.input if i)
            need.update(_subgraph_uses(n))
    keep.reverse()
    del model.graph.node[:]
    model.graph.node.extend(keep)
    used = {i for n in keep for i in n.input} | {u for n in keep for u in _subgraph_uses(n)}
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
        prod = {o: n for n in g.node for o in n.output}
        tgt, tgt_node = nxt.input[1], nxt
        slot = 1
        if tgt in prod and prod[tgt].op_type == "Cast":   # fp16 build: the norm runs in float32 behind Casts; add before the Cast, in the io dtype
            tgt_node, slot = prod[tgt], 0
            tgt = tgt_node.input[0]
        if not re.search(rf"/layers\.{i}/mlp/down_proj/MatMul/output_0$", tgt):
            raise SystemExit(f"layer {i + 1} skip input is {tgt}, expected the down_proj output of layer {i}")
        name = f"deepstack_{i}"
        g.input.append(helper.make_tensor_value_info(name, dt, ["batch_size", "sequence_length", hidden]))
        out = f"/model/layers.{i}/deepstack/Add/output_0"
        add = helper.make_node("Add", [tgt, name], [out], name=f"/model/layers.{i}/deepstack/Add")
        g.node.insert(list(g.node).index(tgt_node), add)
        tgt_node.input[slot] = out
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
    lm = next((n for n in g.node if n.name == "/lm_head/MatMul_Q4"), None)
    src = lm.input[0] if lm is not None else fin.output[0]   # the tensor the lm_head reads: the final norm output in the io dtype
    ident = helper.make_node("Identity", [src], ["hidden"], name="/model/hidden_output")
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


def check_merger_layout(w, j=0):
    """DeepStack mergers normalise AFTER the 2x2 regrouping (use_postshuffle_norm=True: LayerNorm over 4 * hidden, = the input width of fc1), unlike the main
    merger (LayerNorm over hidden, then the regrouping): checked on the checkpoint header (norm.weight [4096], fc1 [4096, 4096]) and in Qwen3VLVisionModel.__init__.
    The edit builds the post-shuffle layout, so a checkpoint that says otherwise fails here instead of producing a wrong graph."""
    if w["norm.weight"].shape[0] != w["linear_fc1.weight"].shape[1]:
        raise SystemExit(f"deepstack_merger_list.{j}: norm {tuple(w['norm.weight'].shape)} does not match fc1 {tuple(w['linear_fc1.weight'].shape)}: not the post-shuffle layout this edit builds")


def vision_single_posembed_gather(model, table="visual.pos_embed.weight"):
    """Replace the four Gather nodes that read the interpolated position embedding (same table, four index vectors) by ONE Gather over the concatenated indices and a Split.
    onnxruntime-web 1.31.1 on WebGPU returns the rows of another Gather for two of the four (reproduced against CPU on the same indices); one Gather has no other to be confused with."""
    from onnx import helper, numpy_helper
    import numpy as np
    g = model.graph
    gs = [n for n in g.node if n.op_type == "Gather" and n.input[0] == table]
    if len(gs) < 2:
        return model, 0
    pos = list(g.node).index(gs[0])
    cat = helper.make_node("Concat", [n.input[1] for n in gs], ["posembed/indices"], axis=0, name="posembed/Concat")
    one = helper.make_node("Gather", [table, "posembed/indices"], ["posembed/rows"], axis=0, name="posembed/Gather")
    outs = [n.output[0] for n in gs]
    sp = helper.make_node("Split", ["posembed/rows"], outs, axis=0, num_outputs=len(gs), name="posembed/Split")
    for n in gs:
        g.node.remove(n)
    for k, n in enumerate([cat, one, sp]):
        g.node.insert(pos + k, n)
    return model, len(gs)


def vision_external_posembed(model, name="pos_embeds"):
    """Make the interpolated position embedding a graph INPUT [num_patches, hidden]: the Add that follows the patch Conv (through Reshapes) reads `name` instead of the
    graph's own Gather/interpolation, and the nodes only that path used are left dead for the runtime to drop. The caller computes the rows (src/media/imagePrep.ts `positionEmbeddings`);
    onnxruntime-web 1.31.1 on WebGPU gets that path wrong (spec 10.9). Returns (model, tensor that was replaced)."""
    from onnx import helper
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    conv = next((n for n in g.node if n.op_type == "Conv"), None)
    if conv is None:
        raise SystemExit("no Conv (patch embedding) in the vision graph")
    cur = conv.output[0]
    while True:
        nxt = cons.get(cur, [])
        if len(nxt) != 1:
            raise SystemExit(f"patch embedding output {cur} has {len(nxt)} consumers: not the public layout")
        n = nxt[0]
        if n.op_type == "Add":
            break
        if n.op_type not in ("Reshape", "Cast"):
            raise SystemExit(f"unexpected {n.op_type} after the patch convolution")
        cur = n.output[0]
    other = [i for i in n.input if i != cur]
    if len(other) != 1:
        raise SystemExit("the Add after the patch embedding does not have two different inputs")
    old = other[0]
    elem = next((v.type.tensor_type.elem_type for v in g.value_info if v.name == old), None)
    if elem is None:
        elem = next((v.type.tensor_type.elem_type for v in g.value_info if v.name == cur), 1)
    hid = None
    for v in g.value_info:
        if v.name == old and len(v.type.tensor_type.shape.dim) == 2:
            hid = v.type.tensor_type.shape.dim[1].dim_value or None
    g.input.append(helper.make_tensor_value_info(name, elem, ["num_patches", hid or "hidden"]))
    n.input[list(n.input).index(old)] = name
    return model, old   # the dead interpolation nodes stay: onnxruntime drops unused nodes itself, and a pruner that misses what the Loop bodies read breaks the graph


def vision_break_aliases(model):
    """Copy every output of a `Split` that feeds a `Squeeze` (the q/k/v unbind of each vision block): Squeeze(Split_out * 1).
    On onnxruntime-web 1.31.1 / WebGPU, the Squeeze of the first output of that Split reads the contents of the third (isolated on the real graph: the Split outputs are right,
    the Squeeze result is another output's data); a multiplication by one in between gives the Squeeze its own buffer. Returns (model, count)."""
    import numpy as np
    from onnx import helper, numpy_helper, TensorProto
    g = model.graph
    vt = {v.name: v.type.tensor_type.elem_type for v in g.value_info}
    split_out = {o: vt.get(n.input[0]) for n in g.node if n.op_type == "Split" for o in n.output}   # output -> element type of the Split's input
    done = 0
    for n in list(g.node):
        if n.op_type == "Squeeze" and split_out.get(n.input[0]) in (TensorProto.FLOAT, TensorProto.FLOAT16):
            src = n.input[0]
            dt = split_out[src]
            one = numpy_helper.from_array(np.array(1.0, dtype={TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}[dt]), f"{src}/one")
            g.initializer.append(one)
            g.node.insert(list(g.node).index(n), helper.make_node("Mul", [src, one.name], [src + "/copy"], name=n.name + "/copy"))
            n.input[0] = src + "/copy"
            done += 1
    return model, done


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
        check_merger_layout(w, j)
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
    v.add_argument("--external-posembed", action="store_true", help="the position embedding as a graph input computed by the caller (WebGPU workaround)")
    v.add_argument("--break-aliases", action="store_true", help="copy the unbound q/k/v tensors (WebGPU workaround)")
    v.add_argument("--single-gather", action="store_true", help="one Gather for the position embedding (WebGPU workaround)")
    a = ap.parse_args(argv)
    m = onnx.load(a.inp, load_external_data=False)
    if a.what == "text":
        if a.deepstack:
            text_deepstack(m)
        if a.hidden_output:
            text_hidden_output(m)
        # The public graph declares a rank-0 shape for rank-1 intermediates (".../TotalLen/Mul/output_0": declared {} , inferred {1}): onnxruntime on CPU merges
        # leniently, onnxruntime-web refuses ("Mismatch between number of inferred and declared dimensions"). Dropping the declared intermediate shapes lets the runtime infer them.
        del m.graph.value_info[:]
    else:
        m, taps = vision_deepstack(m, read_merger_weights(a.base))
        if a.external_posembed:
            m, old = vision_external_posembed(m)
            print("position embedding is now the input pos_embeds (replaced", old + ")")
        if a.break_aliases:
            m, n_a = vision_break_aliases(m)
            print('copies after', n_a, 'unbind Squeeze nodes')
        if a.single_gather:
            m, n_g = vision_single_posembed_gather(m)
            print('single posembed gather over', n_g, 'nodes')
        print("taps", taps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, a.out)
    print(json.dumps({"out": a.out, "inputs": [i.name for i in m.graph.input if not i.name.startswith("past")], "outputs": [o.name for o in m.graph.output if not o.name.startswith("present")]}))


if __name__ == "__main__":
    main(sys.argv[1:])
