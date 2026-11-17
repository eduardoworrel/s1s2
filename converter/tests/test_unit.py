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


if __name__ == "__main__":
    unittest.main()
