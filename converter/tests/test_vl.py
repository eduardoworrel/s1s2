"""Tests of vl-swap and vl-graph with synthetic weights (no model download, a few MB): python -m unittest tests.test_vl (from converter/).

Tolerances, written before the first run of these tests:
  * Vectorised RTN (`rtn_q4`) vs the element-by-element loop reference (`reference_q4`): packed bytes and scales identical; the ONNX Runtime quantizer identical off ties;
    exact ties go up and equal magnitudes of both signs take the negative value (the public build's rules; byte-equal on its lm_head rows and on three LoRA-target matrices of both builds, 0 differing bytes; the ONNX Runtime function differs on 0.03% of the bytes).
  * Dequantised error of one tensor: |w - dequant| <= |scale| element-wise (the clipped side of the symmetric range costs at most one step).
  * LoRA fusion vs PEFT's own forward (one layer, float32): max |difference| <= 1e-5.
  * Swapped graph run in ONNX Runtime CPU vs x @ dequant(reference_q4(W + alpha/r B A)).T: max |difference| <= 1e-4 (same int4 values, float32 accumulation order);
    vs x @ (W + alpha/r B A).T: relative Frobenius error <= 0.20 (int4, block 32).
  * Initializers the LoRA does not touch (lm_head, norms, rotary table): byte-identical to the template. Touched ones: different bytes.
  * Split external data: outputs bit-identical to the unsplit graph.
  * DeepStack text patch: zero inputs give bit-identical outputs; non-zero inputs match the numpy model within 1e-5. Vision patch within 1e-4.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from s1s2_convert import vl_graph as G  # noqa: E402
from s1s2_convert import vl_swap as S  # noqa: E402

H, FF, KV, VOCAB, LAYERS, R, ALPHA = 64, 128, 32, 192, 2, 4, 8


def tri(seed):
    return np.random.default_rng(seed)


class Quantizer(unittest.TestCase):
    def test_rtn_equals_the_loop_reference_and_ort_off_ties(self):
        w = tri(1).standard_normal((12, 128)).astype(np.float32)
        rp, rs = S.reference_q4(w)
        p, s = S.quantize_q4(torch.from_numpy(w), "f32", np.float32)
        self.assertEqual(p.shape, (12, 4, 16))
        self.assertTrue(np.array_equal(p, rp) and np.array_equal(s, rs))
        op, os_ = S.quantize_q4(torch.from_numpy(w), "f32", np.float32, quantizer="ort")   # random floats have no exact ties
        self.assertTrue(np.array_equal(op, rp) and np.array_equal(os_, rs))

    def test_ties_go_up_as_in_the_public_build(self):
        # one block: the largest magnitude is -8 -> scale 1; 0.5 is exactly halfway between steps 8 and 9 (x/scale + 8 = 8.5): the public build gives 9
        w = np.zeros((1, 32), np.float32)
        w[0, 0], w[0, 1], w[0, 2], w[0, 3] = -8.0, 0.5, -0.5, 2.5
        p, s = S.quantize_q4(torch.from_numpy(w), "f32", np.float32)
        self.assertEqual(float(s[0, 0]), 1.0)
        q = np.stack([p[0, 0] & 15, p[0, 0] >> 4], -1).reshape(-1)
        self.assertEqual([int(x) for x in q[:4]], [0, 9, 8, 11])   # -0.5 + 8.5 = 8.0 -> 8; 2.5 + 8.5 = 11 -> 11
        self.assertTrue(np.array_equal(p, S.reference_q4(w)[0]))

    def test_equal_magnitudes_of_both_signs_pick_the_negative_value(self):
        w = np.zeros((1, 32), np.float32)
        w[0, 0], w[0, 5], w[0, 9] = 1.0, -1.0, 0.25   # +1 comes first, -1 later: the public build's scale is -1 / -8 = +0.125, not +1 / -8
        p, s = S.quantize_q4(torch.from_numpy(w), "f32", np.float32)
        self.assertEqual(float(s[0, 0]), 0.125)
        self.assertTrue(np.array_equal(p, S.reference_q4(w)[0]) and np.array_equal(s, S.reference_q4(w)[1]))

    def test_dequant_error_bound(self):
        w = tri(2).standard_normal((16, 96)).astype(np.float32)
        p, s = S.quantize_q4(torch.from_numpy(w), "f32", np.float32)
        d = S.dequant_q4(p, s)
        step = np.repeat(np.abs(s), 32, axis=1)
        self.assertTrue(np.all(np.abs(w - d) <= step * (1 + 1e-5)))

    def test_float16_scales_are_the_ones_q_was_computed_with(self):
        w = (tri(3).standard_normal((8, 64)) * 0.37).astype(np.float32)   # scales not exactly representable in float16
        p, s = S.quantize_q4(torch.from_numpy(w), "f32", np.float16)
        self.assertEqual(s.dtype, np.float16)
        d = S.dequant_q4(p, s)   # dequantised with the stored float16 scale
        self.assertTrue(np.all(np.abs(w - d) <= np.repeat(np.abs(s.astype(np.float32)), 32, axis=1) * (1 + 1e-5)))
        # for bf16-exact weights the f32 and f16 chains give the same bytes
        wb = torch.from_numpy(w).bfloat16().float()
        a, sa = S.quantize_q4(wb, "f32", np.float32)
        b, sb = S.quantize_q4(wb, "f16", np.float32)
        self.assertTrue(np.array_equal(a, b) and np.array_equal(sa, sb))


class Rules(unittest.TestCase):
    def test_names_of_the_public_graph(self):
        names = ["model.inv_freq", "lm_head.MatMul.weight_Q4", "lm_head.MatMul.weight_scales", "model.layers.36.final_norm_layernorm.weight"]
        for L in range(36):
            names += [f"model.layers.{L}.{x}" for x in ("input_layernorm.weight", "post_attention_layernorm.weight", "attn.q_norm.layernorm.weight", "attn.k_norm.layernorm.weight")]
            for pj in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                names += [f"model.layers.{L}.{pj}.MatMul.weight_Q4", f"model.layers.{L}.{pj}.MatMul.weight_scales"]
        self.assertEqual(len(names), 652)
        kinds = [S.rule(n)[0] for n in names]
        self.assertNotIn(None, kinds)
        self.assertEqual(kinds.count("q4"), 252)
        self.assertEqual(S.rule("model.layers.7.mlp.up_proj.MatMul.weight_Q4"), ("q4", "model.language_model.layers.7.mlp.up_proj.weight"))
        self.assertEqual(S.rule("model.layers.7.attn.k_norm.layernorm.weight"), ("norm", "model.language_model.layers.7.self_attn.k_norm.weight"))
        self.assertEqual(S.rule("model.layers.36.final_norm_layernorm.weight"), ("norm", "model.language_model.norm.weight"))
        self.assertEqual(S.rule("model.layers.0.attn.qkv_proj.MatMul.weight_Q4")[0], None)  # a fused layout is not this graph

    def test_real_public_graph_if_present(self):
        root = os.environ.get("S1S2_IMAGE_DIR")
        if not root:
            self.skipTest("needs the graph files of onnx-community/Qwen3-VL-4B-Instruct-ONNX (text.onnx only, no weights): download onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-32/text.onnx "
                          "to $S1S2_IMAGE_DIR/onnx-cpu/ and onnxruntime/cuda/cuda-int4-rtn-block-32/text.onnx to $S1S2_IMAGE_DIR/onnx-cuda/, then run with S1S2_IMAGE_DIR set")
        for d in ("onnx-cpu", "onnx-cuda"):
            p = Path(root) / d / "text.onnx"
            if not p.exists():
                self.skipTest(f"{p} is missing (see S1S2_IMAGE_DIR above)")
            m = onnx.load(str(p), load_external_data=False)
            self.assertEqual([i.name for i in m.graph.initializer if S.rule(i.name)[0] is None], [])


class LoraFusion(unittest.TestCase):
    def test_fused_weight_equals_peft_forward(self):
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(0)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(H, 48, bias=False)

            def forward(self, x):
                return self.q_proj(x)

        base = M()
        w0 = base.q_proj.weight.detach().clone()
        pm = get_peft_model(base, LoraConfig(r=R, lora_alpha=ALPHA, target_modules=["q_proj"], lora_dropout=0.0))
        with torch.no_grad():
            pm.base_model.model.q_proj.lora_B["default"].weight.normal_(0, 0.3)
        A = pm.base_model.model.q_proj.lora_A["default"].weight.detach()
        B = pm.base_model.model.q_proj.lora_B["default"].weight.detach()
        x = torch.randn(5, H)
        want = pm.eval()(x)
        got = x @ (w0 + (ALPHA / R) * B @ A).T
        self.assertLess(float((want - got).abs().max()), 1e-5)


def build_world(tmp, lora=True, seed=0):
    """base safetensors (bf16, HF names), adapter (+config), template ONNX with MatMulNBits made from the base (no LoRA) in one data file."""
    g = tri(seed)
    base, shapes = {}, {}
    for L in range(LAYERS):
        pre = f"model.language_model.layers.{L}."
        for pj, (n, k) in {"self_attn.q_proj": (H, H), "self_attn.k_proj": (KV, H), "self_attn.v_proj": (KV, H), "self_attn.o_proj": (H, H),
                           "mlp.gate_proj": (FF, H), "mlp.up_proj": (FF, H), "mlp.down_proj": (H, FF)}.items():
            base[pre + pj + ".weight"] = torch.from_numpy(g.standard_normal((n, k)).astype(np.float32) * 0.5).bfloat16()
        for nm, dim in (("input_layernorm", H), ("post_attention_layernorm", H), ("self_attn.q_norm", 16), ("self_attn.k_norm", 16)):
            base[pre + nm + ".weight"] = torch.from_numpy(g.uniform(0.5, 1.5, dim).astype(np.float32)).bfloat16()
    base["model.language_model.norm.weight"] = torch.from_numpy(g.uniform(0.5, 1.5, H).astype(np.float32)).bfloat16()
    base["model.language_model.embed_tokens.weight"] = torch.from_numpy(g.standard_normal((VOCAB, H)).astype(np.float32)).bfloat16()
    bdir = Path(tmp) / "base"
    bdir.mkdir()
    save_file(base, str(bdir / "model.safetensors"), metadata={"format": "pt"})
    ad = {}
    for k, w in base.items():
        if k.endswith("_proj.weight"):
            mod = k[: -len(".weight")]
            ad[f"base_model.model.{mod}.lora_A.weight"] = torch.from_numpy(g.standard_normal((R, w.shape[1])).astype(np.float32) * 0.2)
            ad[f"base_model.model.{mod}.lora_B.weight"] = torch.from_numpy(g.standard_normal((w.shape[0], R)).astype(np.float32) * 0.2)
    adir = Path(tmp) / "adapter"
    adir.mkdir()
    save_file(ad, str(adir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (adir / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": ALPHA}))
    # template: initializers named as the public graph
    inits, nodes = [], []

    def q(name, key, n, k):
        p, s = S.quantize_q4(base[key].float(), "f32", np.float32)
        inits.append(numpy_helper.from_array(p, name + "_Q4"))
        inits.append(numpy_helper.from_array(s, name + "_scales"))
        return name + "_Q4", name + "_scales"

    outs = []
    for L in range(LAYERS):
        pre = f"model.language_model.layers.{L}."
        for pj, hf, (n, k), inp in (("attn.q_proj", "self_attn.q_proj", (H, H), "x"), ("attn.k_proj", "self_attn.k_proj", (KV, H), "x"), ("attn.v_proj", "self_attn.v_proj", (KV, H), "x"),
                                    ("attn.o_proj", "self_attn.o_proj", (H, H), "x"), ("mlp.gate_proj", "mlp.gate_proj", (FF, H), "x"), ("mlp.up_proj", "mlp.up_proj", (FF, H), "x"),
                                    ("mlp.down_proj", "mlp.down_proj", (H, FF), "h")):
            a, b = q(f"model.layers.{L}.{pj}.MatMul.weight", pre + hf + ".weight", n, k)
            o = f"out_{L}_{pj}"
            nodes.append(helper.make_node("MatMulNBits", [inp, a, b], [o], domain="com.microsoft", K=k, N=n, bits=4, block_size=32))
            outs.append(helper.make_tensor_value_info(o, TensorProto.FLOAT, [1, "s", n]))
        for nm, hf, dim in (("input_layernorm.weight", "input_layernorm.weight", H), ("post_attention_layernorm.weight", "post_attention_layernorm.weight", H),
                            ("attn.q_norm.layernorm.weight", "self_attn.q_norm.weight", 16), ("attn.k_norm.layernorm.weight", "self_attn.k_norm.weight", 16)):
            inits.append(numpy_helper.from_array(base[pre + hf].float().numpy(), f"model.layers.{L}.{nm}"))
    inits.append(numpy_helper.from_array(base["model.language_model.norm.weight"].float().numpy(), f"model.layers.{LAYERS}.final_norm_layernorm.weight"))
    a, b = q("lm_head.MatMul.weight", "model.language_model.embed_tokens.weight", VOCAB, H)
    nodes.append(helper.make_node("MatMulNBits", ["x", a, b], ["logits"], domain="com.microsoft", K=H, N=VOCAB, bits=4, block_size=32))
    outs.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "s", VOCAB]))
    inits.append(numpy_helper.from_array(g.standard_normal(8).astype(np.float32), "model.inv_freq"))
    graph = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "s", H]), helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, "s", FF])],
                              outs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    tdir = Path(tmp) / "template"
    tdir.mkdir()
    onnx.save_model(model, str(tdir / "text.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, location="text.onnx.data", size_threshold=0)
    (tdir / "genai_config.json").write_text("{}")
    return base, ad, bdir, adir, tdir


def run_graph(path, feeds):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    return dict(zip(names, s.run(None, feeds)))


class Swap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base, cls.ad, cls.bdir, cls.adir, cls.tdir = build_world(cls.tmp.name)
        cls.out = Path(cls.tmp.name) / "out"
        cls.rep = S.run(cls.tdir, cls.bdir, cls.out, str(cls.adir / "adapter_model.safetensors"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_verify_no_lora_is_byte_equal_and_probe_finds_the_chain(self):
        rep = S.run(self.tdir, self.bdir, verify=True)
        self.assertTrue(rep["oracle"])
        self.assertEqual(rep["chain"], "f32")
        self.assertEqual(rep["different"], [])
        self.assertGreater(rep["equal"], 0)

    def test_only_lora_targets_change(self):
        _, ti = S.load_template(self.tdir)
        _, oi = S.load_template(self.out)
        a, b = (self.tdir / "text.onnx.data").read_bytes(), (self.out / "text.onnx.data").read_bytes()
        self.assertEqual(len(a), len(b))
        changed = set()
        for n, d in ti.items():
            if a[d["off"]: d["off"] + d["len"]] != b[d["off"]: d["off"] + d["len"]]:
                changed.add(n)
        for n in ti:
            kind = S.rule(n, LAYERS)[0]
            self.assertEqual(n in changed, kind in ("q4", "q4s"), n)  # lm_head, norms and the rotary table are untouched
        self.assertEqual(self.rep["swapped"], 2 * 7 * LAYERS)
        self.assertEqual(self.rep["lora_modules_applied"], 7 * LAYERS)

    def test_outputs_follow_the_fused_weights(self):
        g = tri(9)
        x = g.standard_normal((1, 5, H)).astype(np.float32)
        h = g.standard_normal((1, 5, FF)).astype(np.float32)
        got = run_graph(self.out / "text.onnx", {"x": x, "h": h})
        old = run_graph(self.tdir / "text.onnx", {"x": x, "h": h})
        self.assertTrue(np.array_equal(got["logits"], old["logits"]))
        for L in range(LAYERS):
            for pj, hf, inp in (("attn.q_proj", "self_attn.q_proj", x), ("attn.k_proj", "self_attn.k_proj", x), ("attn.v_proj", "self_attn.v_proj", x), ("attn.o_proj", "self_attn.o_proj", x),
                                ("mlp.gate_proj", "mlp.gate_proj", x), ("mlp.up_proj", "mlp.up_proj", x), ("mlp.down_proj", "mlp.down_proj", h)):
                key = f"model.language_model.layers.{L}.{hf}.weight"
                A, B = self.ad[f"base_model.model.{key[:-7]}.lora_A.weight"], self.ad[f"base_model.model.{key[:-7]}.lora_B.weight"]
                wf = self.base[key].float().numpy() + (ALPHA / R) * (B.numpy() @ A.numpy())
                rp, rs = S.reference_q4(wf)
                exp_q = inp[0] @ S.dequant_q4(rp, rs).T
                o = got[f"out_{L}_{pj}"][0]
                self.assertLess(float(np.abs(o - exp_q).max()), 1e-4, (L, pj))
                exact = inp[0] @ wf.T
                self.assertLess(float(np.linalg.norm(o - exact) / np.linalg.norm(exact)), 0.20, (L, pj))
                self.assertGreater(float(np.abs(o - old[f"out_{L}_{pj}"][0]).max()), 1e-3, (L, pj))  # the LoRA did change the answer

    def test_split_is_transparent(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            for f in self.out.iterdir():
                (d / f.name).write_bytes(f.read_bytes())
            names = S.split_external_data(d / "text.onnx", d / "text.onnx.data", max_bytes=8192)
            self.assertGreater(len(names), 3)
            self.assertTrue(all((d / n).exists() for n in names))
            self.assertTrue(all((d / n).stat().st_size <= 8192 + 4096 for n in names))
            g = tri(5)
            feeds = {"x": g.standard_normal((1, 3, H)).astype(np.float32), "h": g.standard_normal((1, 3, FF)).astype(np.float32)}
            a, b = run_graph(self.out / "text.onnx", feeds), run_graph(d / "text.onnx", feeds)
            for k in a:
                self.assertTrue(np.array_equal(a[k], b[k]), k)

    def test_unknown_initializer_stops(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "tpl"
            d.mkdir()
            m = onnx.load(str(self.tdir / "text.onnx"), load_external_data=False)
            m.graph.initializer[0].name = "model.layers.0.attn.qkv_proj.MatMul.weight_Q4"
            onnx.save(m, str(d / "text.onnx"))
            (d / "text.onnx.data").write_bytes((self.tdir / "text.onnx.data").read_bytes())
            with self.assertRaises(SystemExit):
                S.run(d, self.bdir, verify=True)


# ---- graph edits -------------------------------------------------------------------------------------------------------------------------

def rms(x, g, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def toy_text(path, n_layers=3, hid=16, seed=0):
    """Same node names and fused structure as the public text graph: SkipSimplifiedLayerNormalization(residual, branch, gamma)."""
    g = tri(seed)
    I, N, ini = [], [], []

    def w(name, shape, scale=0.3):
        a = (g.standard_normal(shape) * scale).astype(np.float32)
        ini.append(numpy_helper.from_array(a, name))
        return name, a

    gam = {}
    for L in range(n_layers + 1):
        gam[L] = {k: g.uniform(0.5, 1.5, hid).astype(np.float32) for k in ("in", "post")}
        for k in ("in", "post"):
            ini.append(numpy_helper.from_array(gam[L][k], f"gamma.{L}.{k}"))
    W = {}
    for L in range(n_layers):
        W[L] = {"a": w(f"Wa.{L}", (hid, hid))[1], "d": w(f"Wd.{L}", (hid, hid))[1]}
    sk = "SkipSimplifiedLayerNormalization"
    prev_res, prev_mlp = None, None
    for L in range(n_layers):
        pre = f"/model/layers.{L}"
        if L == 0:
            N.append(helper.make_node("SimplifiedLayerNormalization", ["inputs_embeds", f"gamma.0.in"], [f"{pre}/input_layernorm/output_0"], name=f"{pre}/input_layernorm/LayerNorm", axis=-1, epsilon=1e-6))
            res_in = "inputs_embeds"
        else:
            N.append(helper.make_node(sk, [prev_res, prev_mlp, f"gamma.{L}.in"], [f"{pre}/input_layernorm/output_0", "", "", f"{pre}/input_layernorm/output_3"],
                                      name=f"{pre}/input_layernorm/SkipLayerNorm", domain="com.microsoft", epsilon=1e-6))
            res_in = f"{pre}/input_layernorm/output_3"
        N.append(helper.make_node("MatMul", [f"{pre}/input_layernorm/output_0", f"Wa.{L}"], [f"{pre}/attn/o_proj/MatMul/output_0"], name=f"{pre}/attn/o_proj/MatMul"))
        N.append(helper.make_node(sk, [res_in, f"{pre}/attn/o_proj/MatMul/output_0", f"gamma.{L}.post"], [f"{pre}/post_attention_layernorm/output_0", "", "", f"{pre}/post_attention_layernorm/output_3"],
                                  name=f"{pre}/post_attention_layernorm/SkipLayerNorm", domain="com.microsoft", epsilon=1e-6))
        N.append(helper.make_node("MatMul", [f"{pre}/post_attention_layernorm/output_0", f"Wd.{L}"], [f"{pre}/mlp/down_proj/MatMul/output_0"], name=f"{pre}/mlp/down_proj/MatMul"))
        prev_res, prev_mlp = f"{pre}/post_attention_layernorm/output_3", f"{pre}/mlp/down_proj/MatMul/output_0"
    fin = f"/model/layers.{n_layers}/final_norm_layernorm"
    ini.append(numpy_helper.from_array(gam[n_layers]["in"], f"model.layers.{n_layers}.final_norm_layernorm.weight"))
    N.append(helper.make_node(sk, [prev_res, prev_mlp, f"model.layers.{n_layers}.final_norm_layernorm.weight"], [f"{fin}/output_0"], name=f"{fin}/SkipLayerNorm", domain="com.microsoft", epsilon=1e-6))
    ini.append(numpy_helper.from_array(g.standard_normal((hid, 50)).astype(np.float32), "lm_head.w_t"))
    N.append(helper.make_node("MatMul", [f"{fin}/output_0", "lm_head.w_t"], ["logits"], name="/lm_head/MatMul"))
    graph = helper.make_graph(N, "toy", [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", hid])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "sequence_length", 50])], initializer=ini)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)])
    m.ir_version = 10
    onnx.save(m, str(path))
    return gam, W


def toy_text_numpy(x, gam, W, ds=None, n_layers=3):
    """Reference of the toy: the hidden state after layer L gets `ds[L]` added (DeepStack) before the next layer reads it."""
    res, mlp = x, None
    for L in range(n_layers):
        r = x if L == 0 else res + mlp
        att = rms(r, gam[L]["in"]) @ W[L]["a"]
        res = r + att
        mlp = rms(res, gam[L]["post"]) @ W[L]["d"]
        if ds is not None and L in ds:
            mlp = mlp + ds[L]
    return rms(res + mlp, gam[n_layers]["in"])


class GraphEdits(unittest.TestCase):
    def test_text_deepstack_and_hidden_output(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "toy.onnx"
            gam, W = toy_text(p)
            x = tri(4).standard_normal((2, 6, 16)).astype(np.float32)
            base = run_graph(p, {"inputs_embeds": x})["logits"]
            m = onnx.load(str(p))
            G.text_deepstack(m, layers=(0, 1, 2), hidden=16)
            G.text_hidden_output(m)
            q = Path(t) / "patched.onnx"
            onnx.save(m, str(q))
            self.assertEqual([i.name for i in m.graph.input], ["inputs_embeds", "deepstack_0", "deepstack_1", "deepstack_2"])
            self.assertEqual([o.name for o in m.graph.output], ["hidden"])
            self.assertNotIn("lm_head.w_t", [i.name for i in m.graph.initializer])
            z = {f"deepstack_{i}": np.zeros_like(x) for i in range(3)}
            hid0 = run_graph(q, {"inputs_embeds": x, **z})["hidden"]
            ref0 = toy_text_numpy(x, gam, W)
            self.assertLess(float(np.abs(hid0 - ref0).max()), 1e-5)
            self.assertTrue(np.array_equal(base, base))  # logits graph still runs (sanity of the original)
            ds = {i: (tri(10 + i).standard_normal(x.shape) * (np.arange(6)[None, :, None] % 2)).astype(np.float32) for i in range(3)}
            hid = run_graph(q, {"inputs_embeds": x, **{f"deepstack_{i}": ds[i] for i in range(3)}})["hidden"]
            self.assertLess(float(np.abs(hid - toy_text_numpy(x, gam, W, ds)).max()), 1e-5)
            self.assertGreater(float(np.abs(hid - hid0).max()), 1e-2)  # the injection does something

    def test_external_posembed(self):
        g = tri(21)
        P, hid, side = 12, 6, 4
        table = g.standard_normal((side * side, hid)).astype(np.float32)
        W = g.standard_normal((hid, hid)).astype(np.float32)
        ini = [numpy_helper.from_array(table, "table"), numpy_helper.from_array(np.array([P, hid], np.int64), "shp"), numpy_helper.from_array(W, "W"),
               numpy_helper.from_array(g.standard_normal((hid, 3, 1, 1)).astype(np.float32), "cw")]
        nodes = [helper.make_node("Reshape", ["pix", "shp"], ["x2"], name="r0"),
                 helper.make_node("Conv", ["img", "cw"], ["c"], name="conv"), helper.make_node("Reshape", ["c", "shp"], ["view_1"], name="r1"),
                 helper.make_node("Gather", ["table", "idx"], ["emb"], axis=0, name="gather"), helper.make_node("Tile", ["emb", "reps"], ["repeat_1"], name="tile"),
                 helper.make_node("Add", ["view_1", "repeat_1"], ["sum"], name="add"), helper.make_node("MatMul", ["sum", "W"], ["out"], name="mm")]
        ini.append(numpy_helper.from_array(np.array([1, 1], np.int64), "reps"))
        graph = helper.make_graph([n for n in nodes if n.name != "r0"], "v", [helper.make_tensor_value_info("img", TensorProto.FLOAT, [1, 3, P, 1]), helper.make_tensor_value_info("idx", TensorProto.INT64, [P])],
                                  [helper.make_tensor_value_info("out", TensorProto.FLOAT, [P, hid])], initializer=ini[1:])
        graph.initializer.append(ini[0])
        m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
        m.ir_version = 10
        img = g.standard_normal((1, 3, P, 1)).astype(np.float32)
        idx = g.integers(0, side * side, P).astype(np.int64)
        ref = run_graph_model(m, {"img": img, "idx": idx})["out"]
        m2, old = G.vision_external_posembed(m, "pos_embeds")
        self.assertEqual(old, "repeat_1")
        self.assertEqual([i.name for i in m2.graph.input], ["img", "idx", "pos_embeds"])
        got = run_graph_model(m2, {"img": img, "idx": idx, "pos_embeds": table[idx]})["out"]
        self.assertLess(float(np.abs(got - ref).max()), 1e-6)

    def test_merger_layout_check(self):
        ok = {"norm.weight": np.zeros(4096), "linear_fc1.weight": np.zeros((4096, 4096))}
        G.check_merger_layout(ok)
        with self.assertRaises(SystemExit):
            G.check_merger_layout({"norm.weight": np.zeros(1024), "linear_fc1.weight": np.zeros((4096, 4096))})   # the main merger's layout

    def test_text_deepstack_refuses_other_layouts(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "toy.onnx"
            toy_text(p)
            m = onnx.load(str(p))
            with self.assertRaises(SystemExit):
                G.text_deepstack(m, layers=(0, 1, 2, 3), hidden=16)  # layer 4 does not exist

    def test_vision_deepstack(self):
        hid, merge, out = 8, 4, 12
        g = tri(7)
        P = 16
        ini, nodes = [], []

        def w(name, shape, s=0.3):
            a = (g.standard_normal(shape) * s).astype(np.float32)
            ini.append(numpy_helper.from_array(a, name))
            return a

        n_blocks = 4
        Ws = [w(f"W{b}", (hid, hid)) for b in range(n_blocks)]
        lw = []
        for i in range(2 * n_blocks + 1):
            gw = g.uniform(0.5, 1.5, hid).astype(np.float32)
            ini.append(numpy_helper.from_array(gw, f"ln{i}.w"))
            lw.append((gw, w(f"ln{i}.b", (hid,), 0.1)))
        cur = "x"
        for b in range(n_blocks):
            nodes.append(helper.make_node("LayerNormalization", [cur, f"ln{2 * b}.w", f"ln{2 * b}.b"], [f"a{b}"], name=f"ln{2 * b}", axis=-1, epsilon=1e-6))
            nodes.append(helper.make_node("Add", [cur, f"a{b}"], [f"s{b}"], name=f"s{b}"))
            nodes.append(helper.make_node("LayerNormalization", [f"s{b}", f"ln{2 * b + 1}.w", f"ln{2 * b + 1}.b"], [f"c{b}"], name=f"ln{2 * b + 1}", axis=-1, epsilon=1e-6))
            nodes.append(helper.make_node("MatMul", [f"c{b}", f"W{b}"], [f"d{b}"], name=f"mm{b}"))
            nodes.append(helper.make_node("Add", [cur, f"d{b}"], [f"h{b}"], name=f"h{b}"))
            cur = f"h{b}"
        nodes.append(helper.make_node("LayerNormalization", [cur, f"ln{2 * n_blocks}.w", f"ln{2 * n_blocks}.b"], ["mn"], name="merger_ln", axis=-1, epsilon=1e-6))
        ini.append(numpy_helper.from_array(np.array([-1, hid * merge], dtype=np.int64), "shp"))
        nodes.append(helper.make_node("Reshape", ["mn", "shp"], ["image_features"], name="merger_reshape"))
        graph = helper.make_graph(nodes, "v", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["num_patches", hid])],
                                  [helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["num_logical_patches", hid * merge])], initializer=ini)
        m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
        m.ir_version = 10
        x = g.standard_normal((P, hid)).astype(np.float32)
        base_feat = run_graph_model(m, {"x": x})["image_features"]
        weights = [{"norm.weight": g.standard_normal(hid * merge).astype(np.float32) * 0.2 + 1, "norm.bias": g.standard_normal(hid * merge).astype(np.float32) * 0.1,
                    "linear_fc1.weight": g.standard_normal((hid * merge, hid * merge)).astype(np.float32) * 0.1, "linear_fc1.bias": g.standard_normal(hid * merge).astype(np.float32) * 0.1,
                    "linear_fc2.weight": g.standard_normal((out, hid * merge)).astype(np.float32) * 0.1, "linear_fc2.bias": g.standard_normal(out).astype(np.float32) * 0.1} for _ in range(3)]
        m2, taps = G.vision_deepstack(m, weights, blocks=(0, 1, 2))
        self.assertEqual(taps, ["h0", "h1", "h2"])  # the input of the first LayerNorm of blocks 1, 2, 3
        res = run_graph_model(m2, {"x": x})
        self.assertTrue(np.array_equal(res["image_features"], base_feat))  # the original output is untouched
        cur = x
        hs = []
        for b in range(n_blocks):
            a = ln(cur, *lw[2 * b])
            s = cur + a
            cur = cur + ln(s, *lw[2 * b + 1]) @ Ws[b]
            hs.append(cur)
        from math import erf
        gelu = np.vectorize(lambda v: 0.5 * v * (1 + erf(v / np.sqrt(2))))
        for j in range(3):
            wj = weights[j]
            y = hs[j].reshape(-1, hid * merge)
            y = ln(y, wj["norm.weight"], wj["norm.bias"])
            y = gelu(y @ wj["linear_fc1.weight"].T + wj["linear_fc1.bias"]).astype(np.float32)
            y = y @ wj["linear_fc2.weight"].T + wj["linear_fc2.bias"]
            self.assertLess(float(np.abs(res[f"deepstack_{j}"] - y).max()), 1e-4)


def ln(x, w, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * w + b


def run_graph_model(m, feeds):
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    return dict(zip([o.name for o in s.get_outputs()], s.run(None, feeds)))


if __name__ == "__main__":
    unittest.main()
