"""pack_mlc / pack_litert: recovery matrix, q4f16_1 dequantization, manifests, adapter fusing, CLI dispatch.
python -m unittest discover -s tests (numpy needed; torch + safetensors for the fuse test; skipped otherwise)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import torch
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover
    torch = None


def quantize_q4f16_1(w):
    """Reference encoder for the layout MLC writes: group 32 along K, scale = max|w| / 7, nibble = round(w / scale) + 7,
    8 nibbles per uint32 (lowest nibble = first element)."""
    V, K = w.shape
    g = w.reshape(V, K // 32, 32)
    scale = (np.abs(g).max(-1) / 7.0).astype(np.float16)
    s32 = np.where(scale == 0, 1, scale).astype(np.float32)
    nib = np.clip(np.round(g / s32[:, :, None]) + 7, 0, 14).astype(np.uint32).reshape(V, K)
    words = (nib.reshape(V, K // 8, 8) << (np.arange(8, dtype=np.uint32) * 4)).sum(-1, dtype=np.uint64).astype(np.uint32)
    return words, scale


def write_mlc_build(d, params, f32_bf16=()):
    """tensor-cache.json + one shard, the way mlc_llm convert_weight lays them out."""
    d = Path(d)
    blob, recs = b"", []
    for name, a in params.items():
        raw = a.tobytes()
        dt = {"uint32": "uint32", "float16": "float16", "float32": "float32"}[str(a.dtype)]
        fmt = "f32-to-bf16"
        if name in f32_bf16:  # float32 values stored as bf16 (q4f32_1 scales)
            raw = (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        recs.append({"name": name, "shape": list(a.shape), "dtype": dt, "format": fmt, "nbytes": len(raw), "byteOffset": len(blob)})
        blob += raw
    (d / "params_shard_0.bin").write_bytes(blob)
    (d / "tensor-cache.json").write_text(json.dumps({"metadata": {}, "records": [{"dataPath": "params_shard_0.bin", "format": "raw-shard", "nbytes": len(blob), "records": recs}]}))


@unittest.skipIf(np is None, "numpy not installed")
class Recovery(unittest.TestCase):
    def setUp(self):
        from s1s2_convert import pack_mlc as M
        self.M = M
        rng = np.random.default_rng(1)
        self.V, self.D = 600, 64
        self.W = (rng.standard_normal((self.V, self.D)) * 0.05).astype(np.float32)
        self.qw, self.qs = quantize_q4f16_1(self.W)
        self.rng = rng

    def test_dequant_layout(self):
        rows = np.array([0, 5, 599])
        E = self.M.dequant_rows(self.qw, self.qs, rows)
        err = np.abs(E - self.W[rows]).max()
        step = self.qs[rows].astype(np.float32).max()
        self.assertLessEqual(err, step / 2 + 1e-3)          # quantization error is half a step at most

    def test_recover_hidden_from_logits(self):
        rows, P, cond = self.M.recovery_matrix(self.qw, self.qs, 300, seed=0)
        self.assertEqual(P.shape, (self.D, 300))
        self.assertTrue(np.all(np.diff(rows) > 0))
        E_all = self.M.dequant_rows(self.qw, self.qs, np.arange(self.V))
        h = self.rng.standard_normal(self.D).astype(np.float32)
        logits = (E_all @ h).astype(np.float16).astype(np.float32)     # the runtime hands over fp16-rounded logits
        got = P @ logits[rows]
        cos = float(got @ h / (np.linalg.norm(got) * np.linalg.norm(h)))
        self.assertGreater(cos, 0.9999)
        # same indexing as src/backends/webllm.ts: h[d] = sum_n P[d * N + n] * logits[rows[n]]
        flat = P.reshape(-1)
        h2 = np.array([sum(flat[d * 300 + n] * logits[rows[n]] for n in range(300)) for d in range(3)])
        np.testing.assert_allclose(h2, got[:3], rtol=1e-4, atol=1e-5)

    def test_too_few_rows_refused(self):
        with self.assertRaises(SystemExit):
            self.M.recovery_matrix(self.qw, self.qs, 32)

    def test_read_build_and_write_recovery(self):
        d = tempfile.mkdtemp()
        write_mlc_build(d, {"model.embed_tokens.q_weight": self.qw, "model.embed_tokens.q_scale": self.qs,
                            "model.norm.weight": np.ones(4, np.float16)})
        info = self.M.write_recovery(d, d, 300, "f16")
        self.assertEqual((info["rows"], info["hidden"], info["source"]), (300, self.D, "model.embed_tokens.q_weight"))
        rows = np.frombuffer((Path(d) / "recovery.rows.i32").read_bytes(), dtype="<i4")
        P = np.frombuffer((Path(d) / "recovery.P.f16").read_bytes(), dtype="<f2")
        self.assertEqual((len(rows), P.size), (300, self.D * 300))

    def test_untied_lm_head_preferred(self):
        names = {"model.embed_tokens.q_weight": 0, "lm_head.q_weight": 0}
        self.assertEqual(self.M.head_param_names(names), ("lm_head.q_weight", "lm_head.q_scale"))
        self.assertEqual(self.M.head_param_names({"model.embed_tokens.q_weight": 0}), ("model.embed_tokens.q_weight", "model.embed_tokens.q_scale"))

    def test_bf16_scales(self):
        d = tempfile.mkdtemp()
        s32 = self.qs.astype(np.float32)
        write_mlc_build(d, {"model.embed_tokens.q_weight": self.qw, "model.embed_tokens.q_scale": s32}, f32_bf16={"model.embed_tokens.q_scale"})
        got = self.M.read_mlc_params(d)["model.embed_tokens.q_scale"]
        np.testing.assert_allclose(got, s32, rtol=1e-2)


SIB = {"schema": "s12.manifest/1", "id": "x.cnli", "base": {"repo": "o/base", "revision": "a" * 40}, "hiddenSize": 64,
       "tokenizer": {"repo": "o/src", "revision": "b" * 40, "files": {}},
       "defaults": {"method": {"predicate": "logits", "choice": "head", "score": "logits"}},
       "lab": {"note": "internal bookkeeping", "task": "cnli", "cue": "Answer:", "model": "internal"},
       "builds": {"genai-int4-webgpu": {"runtime": "genai", "repo": "o/src", "revision": "b" * 40, "files": {}, "hiddenOutput": "hidden_states",
                                        "heads": {"cnli": {"file": {"path": "heads/cnli/head.bin", "sha256": "0" * 64, "size": 1}, "types": ["choice"]}}}}}


class Manifests(unittest.TestCase):
    def test_mlc_head_readout(self):
        from s1s2_convert import pack_mlc as M
        files = {"config": {"path": "mlc-chat-config.json"}, "lib": {"path": "model-lib.wasm"}}
        m = M.new_manifest(SIB, "o/new-MLC-q4f16_1-S1S2", "cnli", files, lambda h: h, "head", {"repo": "o/src", "revision": "b" * 40},
                           {"mlc-ai-nightly-cpu": "0.26.dev246"}, {"rows": 8192, "seed": 0, "format": "f16"}, "n", "prebuilt x")
        b = m["builds"]["webllm-q4f16_1"]
        self.assertEqual((m["schema"], b["runtime"], b["hiddenOutput"]), ("s1s2.manifest/1", "webllm", "recovered-from-logits"))
        self.assertIn("heads", b)
        self.assertEqual(m["defaults"]["method"]["choice"], "head")
        self.assertNotIn("model", m["lab"])                  # only the keys s1s2 reads are kept
        self.assertEqual(m["tokenizer"]["repo"], "o/new-MLC-q4f16_1-S1S2")

    def test_mlc_logits_readout_has_no_heads(self):
        from s1s2_convert import pack_mlc as M
        m = M.new_manifest(SIB, "o/n", "cnli", {"config": {}, "lib": {}}, lambda h: h, "logits", {"repo": "o/src", "revision": "b" * 40}, {}, None, "n", "x")
        b = m["builds"]["webllm-q4f16_1"]
        self.assertIsNone(b["hiddenOutput"])
        self.assertNotIn("heads", b)
        self.assertEqual(set(m["defaults"]["method"].values()), {"logits"})

    def test_litert_manifest(self):
        from s1s2_convert import pack_litert as L
        m = L.new_manifest(SIB, "o/n-LiteRT-S1S2", "cnli", {"manifest": {}, "graph": {}, "weights": {}, "embed": {}}, {"repo": "o/src", "revision": "b" * 40}, {}, "n", 2048)
        b = m["builds"]["litert-int4"]
        self.assertEqual((b["runtime"], b["hiddenOutput"], b["maxSafeTokens"]), ("litert", None, 2040))
        self.assertNotIn("heads", b)
        self.assertEqual(set(m["defaults"]["method"].values()), {"logits"})

    def test_cli_dispatch(self):
        from s1s2_convert import cli
        with self.assertRaises(SystemExit) as c:
            cli.main(["pack-mlc", "--help"])
        self.assertEqual(c.exception.code, 0)
        with self.assertRaises(SystemExit) as c:
            cli.main(["pack-litert", "--help"])
        self.assertEqual(c.exception.code, 0)

    def test_recipe_is_int4_block32(self):
        from s1s2_convert import pack_litert as L
        r = json.loads(L.RECIPE.read_text())[0]["op_config"]["weight_tensor_config"]
        self.assertEqual((r["num_bits"], r["granularity"]), (4, "BLOCKWISE_32"))


@unittest.skipIf(torch is None, "torch / safetensors not installed")
class Fuse(unittest.TestCase):
    def test_fuse_applies_lora_delta(self):
        from s1s2_convert import fuse_lora as F
        d = Path(tempfile.mkdtemp())
        base, ad = d / "base", d / "ad.safetensors"
        base.mkdir()
        torch.manual_seed(0)
        W = torch.randn(8, 4).bfloat16()
        save_file({"model.layers.0.self_attn.q_proj.weight": W, "model.norm.weight": torch.ones(8).bfloat16()}, str(base / "model.safetensors"), metadata={"format": "pt"})
        (base / "config.json").write_text("{}")
        A, B = torch.randn(2, 4), torch.randn(8, 2)
        save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A,
                   "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B}, str(ad))
        info = F.fuse(base, ad, 2.0, d / "out")
        from safetensors import safe_open
        with safe_open(str(d / "out" / "model.safetensors"), "pt") as f:
            got = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            same = f.get_tensor("model.norm.weight")
        want = (W.float() + 2.0 * (B @ A)).bfloat16()
        self.assertTrue(torch.equal(got, want))
        self.assertTrue(torch.equal(same, torch.ones(8).bfloat16()))
        self.assertEqual(info["modules"], 1)
        self.assertTrue((d / "out" / "config.json").exists())

    def test_unmatched_module_refused(self):
        from s1s2_convert import fuse_lora as F
        with self.assertRaises(SystemExit):
            F.target_keys({"model.layers.9.q_proj": {}}, {"model.layers.0.q_proj.weight"})


if __name__ == "__main__":
    unittest.main()
