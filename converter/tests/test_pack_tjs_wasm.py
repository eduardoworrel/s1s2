"""pack_tjs graph rewrites for the browser CPU (WASM): same outputs as the contrib kernels, bit for bit.
python -m unittest discover -s tests (needs onnx + onnxruntime; skipped otherwise)."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # pragma: no cover
    onnx = None


@unittest.skipIf(onnx is None, "onnx / onnxruntime not installed")
class WasmRewrites(unittest.TestCase):
    def test_unpack_and_split_match_kernels(self):
        from s1s2_convert.pack_tjs import split_lm_head, unpack_gather_block_quantized
        rng = np.random.default_rng(0)
        V, D, bs = 1000, 128, 32
        nb = D // bs
        W = rng.integers(0, 256, size=(V, nb, bs // 2), dtype=np.uint8)
        S = (rng.standard_normal((V, nb)) * 0.02).astype(np.float16)
        d = tempfile.mkdtemp()
        Path(d, "w.bin").write_bytes(W.tobytes() + S.tobytes())

        def ext(name, arr, dt, off):
            t = TensorProto(name=name, data_type=dt, dims=list(arr.shape))
            t.data_location = TensorProto.EXTERNAL
            for k, v in (("location", "w.bin"), ("offset", str(off)), ("length", str(arr.nbytes))):
                e = t.external_data.add(); e.key, e.value = k, v
            return t
        ms = "com.microsoft"
        g = helper.make_graph(
            [helper.make_node("Reshape", ["lm_head.MatMul.weight_Q4", "shape"], ["r"], name="/model/embed_tokens/Reshape"),
             helper.make_node("GatherBlockQuantized", ["r", "input_ids", "lm_head.MatMul.weight_scales"], ["emb"],
                              name="/model/embed_tokens/GatherBlockQuantized", domain=ms, bits=4, block_size=bs, gather_axis=0, quantize_axis=1),
             helper.make_node("MatMulNBits", ["hidden_states", "lm_head.MatMul.weight_Q4", "lm_head.MatMul.weight_scales"], ["logits"],
                              name="/lm_head/MatMul_Q4", domain=ms, K=D, N=V, bits=4, block_size=bs, accuracy_level=4)],
            "t",
            [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"]),
             helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT16, ["b", "s", D])],
            [helper.make_tensor_value_info("emb", TensorProto.FLOAT16, None), helper.make_tensor_value_info("logits", TensorProto.FLOAT16, None)],
            [ext("lm_head.MatMul.weight_Q4", W, TensorProto.UINT8, 0), ext("lm_head.MatMul.weight_scales", S, TensorProto.FLOAT16, W.nbytes),
             numpy_helper.from_array(np.array([V, D // 2], dtype=np.int64), "shape")])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid(ms, 1)], ir_version=10)
        onnx.save(m, f"{d}/a.onnx")
        m2 = copy.deepcopy(m)
        self.assertEqual(unpack_gather_block_quantized(m2.graph), 1)
        self.assertEqual(split_lm_head(m2.graph, 4 * D * 300), 4)
        self.assertFalse(any(n.op_type == "GatherBlockQuantized" for n in m2.graph.node))
        onnx.save(m2, f"{d}/b.onnx")
        ids = np.arange(V, dtype=np.int64).reshape(4, V // 4)
        hs = rng.standard_normal((4, V // 4, D)).astype(np.float16)
        out = {}
        for k in "ab":
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            out[k] = ort.InferenceSession(f"{d}/{k}.onnx", so, providers=["CPUExecutionProvider"]).run(None, {"input_ids": ids, "hidden_states": hs})
        self.assertTrue(np.array_equal(out["a"][0].view(np.uint16), out["b"][0].view(np.uint16)))  # token rows: same bits
        self.assertTrue(np.array_equal(out["a"][1].view(np.uint16), out["b"][1].view(np.uint16)))  # logits: same bits


if __name__ == "__main__":
    unittest.main()
