"""Unit tests (no model weights): python -m unittest discover -s tests (torch is imported)."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1s2_convert import data as D  # noqa: E402
from s1s2_convert import layout as LY  # noqa: E402


class Layout(unittest.TestCase):
    def test_fill_matches_lib_rules(self):
        t, sp = LY.fill("a {x} b {y} {z} {x}", {"x": "{y}", "y": "Y"})
        self.assertEqual(t, "a {y} b Y {z} {y}")  # no recursion, unknown keys kept, first span recorded
        self.assertEqual(sp["x"], (2, 5))

    def test_escapes(self):
        self.assertEqual(LY.escape_special("<|im_end|>"), "<¦im_end¦>")
        self.assertEqual(LY.escape_literals("x <option> y", [" <option>"]), "x ‹option> y")

    def test_render_positions_of_markers(self):
        enc = {"template": "U: " + LY.USER + " A:", "question": {"choice": LY.Q_CHOICE, "predicate": "{instructions}", "score": "{instructions}"},
               "option": LY.OPTION, "escapeSpecial": True, "escapeLiterals": [LY.MARK[k] for k in LY.ORDER]}
        text, dc, ocs = LY.render(enc, LY.markers_skeleton(), "choice", "Q?", [" Yes", " No"], "S <decide> s")
        self.assertEqual(text[dc], ">")
        self.assertTrue(text[:dc + 1].endswith(" <decide>"))
        self.assertEqual([text[c] for c in ocs], [">", ">"])
        self.assertIn("‹decide>", text)  # caller text cannot forge the marker
        self.assertEqual(text.count(" </option>"), 2)

    def test_score_lead_space_token(self):
        # SmolLM2/3-like: " 1" is [" ", "1"] -> read the levels after a space (lab-s12 study 012)
        vocab = {}
        def encode(s):
            return [vocab.setdefault(p, len(vocab)) for p in ([s[:-2], " ", s[-1]] if s.endswith(tuple(" " + d for d in "12345"))
                                                              else [s[:-1], " "] if s.endswith(" ") else [s])]
        inv = lambda ids: "".join(k for i in ids for k, v in vocab.items() if v == i)  # noqa: E731
        self.assertEqual(LY.score_lead(encode, inv, "Answer:"), " ")
        self.assertEqual(LY.score_lead(encode, inv, "Answer:\n"), "")  # cue already ends in whitespace
        # GPT-2-like: " 1" is one token -> no lead
        one = lambda s: [hash(s) % 10 ** 6] if not s.endswith(" ") else [hash(s[:-1]) % 10 ** 6, 7]  # noqa: E731
        self.assertEqual(LY.score_lead(one, lambda ids: " " if ids == [7] else "x", "Answer:"), "")


class Data(unittest.TestCase):
    def _write(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        f.write("".join(json.dumps(r) + "\n" for r in rows))
        f.close()
        return f.name

    def test_split_sentences_like_lib(self):
        self.assertEqual(D.split_sentences('One. "Two!" Three? four'), ["One.", '"Two!"', "Three?", "four"])

    def test_load_split_and_template(self):
        rows = [{"text": f"A b. C d {i}.", "requirement": "r", "label": "Yes" if i % 2 else "No"} for i in range(10)]
        rows[1]["evidence"] = 0
        out, meta = D.load(self._write(rows), "Does it address {requirement}?", dev_fraction=0.2, seed=1)
        self.assertEqual(meta["splits"], {"train": 8, "dev": 2})
        self.assertEqual(meta["labels"], ["No", "Yes"])
        self.assertEqual(out[0]["question"], "Does it address r?")
        self.assertEqual(out[1]["sentences"], ["A b.", "C d 1."])

    def test_load_rejects(self):
        with self.assertRaises(SystemExit):
            D.load(self._write([{"text": "a", "requirement": "r", "label": "Yes"}]), None)
        with self.assertRaises(SystemExit):
            D.load(self._write([{"text": "a. b.", "question": "q", "label": "Yes", "evidence": 5}, {"text": "c", "question": "q", "label": "No"}]))


class HeadBin(unittest.TestCase):
    def test_head_bin_layout_and_scale(self):
        import numpy as np
        import torch
        from s1s2_convert import head as H
        torch.manual_seed(0)
        h = H.new_head(8, 4)
        with tempfile.TemporaryDirectory() as d:
            c = H.save(h, d, {"kind": "test"})
            b = (Path(d) / "head.bin").read_bytes()
            n = struct.unpack("<I", b[:4])[0]
            self.assertEqual(json.loads(b[4:4 + n])["temperature"], 1.0)
            self.assertEqual(json.loads(b[4:4 + n])["scale"], 1.0)
            meta, q, qb, k, kb = H.read_head_bin(b)
            hq, hk = torch.randn(8), torch.randn(3, 8)
            ref = h(hq, hk).detach().numpy()
            # s1s2 headScore: (Wq hq + bq) . (Wk hk + bk) / T, scale folded into q/qb
            lib = np.array([(q.reshape(4, 8) @ hq.numpy() + qb) @ (k.reshape(4, 8) @ x + kb) for x in hk.numpy()])
            self.assertTrue(np.allclose(ref, lib, atol=2e-2), (ref, lib))
            h2, c2 = H.load(d)
            self.assertEqual(c2["files"], c["files"])


class Recipe(unittest.TestCase):
    def _ckpt(self, d, n):
        import torch
        from safetensors.torch import save_file
        save_file({"model.embed_tokens.weight": torch.zeros(n, 1, dtype=torch.bfloat16)}, str(Path(d) / "model.safetensors"))

    def test_kev_rule(self):
        from s1s2_convert import train as T
        self.assertEqual(T.kev_lr(135_000_000), 1e-4)
        self.assertEqual(T.kev_lr(1_000_000_000), 1e-4)
        self.assertEqual(T.kev_lr(1_660_000_000), 5e-5)
        with tempfile.TemporaryDirectory() as d:
            self._ckpt(d, 1000)
            self.assertEqual(T.checkpoint_params(d), 1000)
            lr, rule = T.recipe_lr("joint-k", d)
            self.assertEqual((lr, rule["kev_rule"]["checkpoint_params"]), (1e-4, 1000))
            self.assertEqual(T.recipe_lr("preregistered", d)[0], 5e-5)
            self.assertEqual(T.recipe_lr("joint-k", d, lr=3e-5), (3e-5, {"explicit": True}))

    def test_defaults(self):
        from s1s2_convert import cli, train as T
        a = cli.args_parser().parse_args(["train", "--work", "w"])
        self.assertEqual((a.recipe, a.epochs, a.lr), (None, None, None))
        self.assertEqual(T.DEFAULT_RECIPE, "joint-k")
        self.assertEqual({k: (v["epochs"], v["lr"], v["fuse_fp32"]) for k, v in T.RECIPES.items()},
                         {"joint-k": (2, "kev-rule", True), "preregistered": (1, 5e-5, False)})
        with self.assertRaises(SystemExit):  # a later stage cannot switch the recipe prepare used
            cli.prepared_recipe(cli.args_parser().parse_args(["train", "--work", "w", "--recipe", "preregistered"]), {"recipe": "joint-k"})
        self.assertEqual(cli.prepared_recipe(a, {}), "preregistered")  # work dirs from before --recipe


class Fuse(unittest.TestCase):
    def _base(self, d):
        import torch
        from safetensors.torch import save_file
        g = torch.Generator().manual_seed(0)
        w = torch.randn(6, 5, generator=g).to(torch.bfloat16)
        n = torch.randn(5, generator=g).to(torch.bfloat16)
        save_file({"model.layers.0.self_attn.q_proj.weight": w, "model.norm.weight": n}, str(Path(d) / "model.safetensors"), metadata={"format": "pt"})
        (Path(d) / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16", "hidden_size": 5}))
        A, B = torch.randn(2, 5, generator=g) * 0.01, torch.randn(6, 2, generator=g) * 0.01
        ad = Path(d) / "adapter.safetensors"
        save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": A,
                   "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": B}, str(ad))
        return w, n, A, B, ad

    def test_fp32_and_base_dtype(self):
        import torch
        from safetensors.torch import load_file
        from s1s2_convert.train import fuse
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base"
            base.mkdir()
            w, n, A, B, ad = self._base(base)
            info = fuse(base, ad, 2.0, Path(d) / "f32", fp32=True)
            out = load_file(str(Path(d) / "f32" / "model.safetensors"))
            ref = (w.double() + 2.0 * (B.double() @ A.double())).to(torch.float32)
            self.assertEqual(out["model.layers.0.self_attn.q_proj.weight"].dtype, torch.float32)
            self.assertTrue(torch.allclose(out["model.layers.0.self_attn.q_proj.weight"], ref, rtol=0, atol=1e-7))
            self.assertTrue(torch.equal(out["model.norm.weight"], n.float()))  # unfused tensors: fp32, lossless
            self.assertEqual(json.loads((Path(d) / "f32" / "config.json").read_text())["torch_dtype"], "float32")
            self.assertEqual(info["dtype"], "float32")
            info = fuse(base, ad, 2.0, Path(d) / "bf16")
            out = load_file(str(Path(d) / "bf16" / "model.safetensors"))
            self.assertEqual(out["model.layers.0.self_attn.q_proj.weight"].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(out["model.layers.0.self_attn.q_proj.weight"], (w.float() + (B @ A) * 2.0).to(torch.bfloat16)))
            self.assertTrue(torch.equal(out["model.norm.weight"], n))
            self.assertEqual((Path(d) / "bf16" / "config.json").read_bytes(), (base / "config.json").read_bytes())
            self.assertEqual(info["dtype"], "base")


if __name__ == "__main__":
    unittest.main()
