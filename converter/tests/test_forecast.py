"""Forecast reader tooling (no model weights unless S1S2_TEST_FP32_GRAPH / S1S2_TEST_MODEL are set): python -m unittest discover -s tests"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from s1s2_convert import forecast as F  # noqa: E402

LIB = Path(__file__).resolve().parents[2]


class Layers(unittest.TestCase):
    def test_builder_option_is_one_more_than_the_block(self):
        # entry i of aux_hidden_state_layers is the residual stream ENTERING block i = the output of block i-1
        self.assertEqual(F.builder_option(14), "aux_hidden_state_layers=15")
        self.assertEqual(F.builder_option(0), "aux_hidden_state_layers=1")

    def test_layer_range(self):
        self.assertEqual(F.check_layer(0, 30), 0)
        self.assertEqual(F.check_layer(28, 30), 28)
        for bad in (-1, 29, 30, "3", True):
            with self.assertRaises(SystemExit):
                F.check_layer(bad, 30)

    def test_build_patch(self):
        self.assertEqual(F.build_patch(14), {"forecastOutput": "aux_hidden_states", "forecastLayer": 14})


class Pooling(unittest.TestCase):
    def test_same_arithmetic_as_the_library(self):
        rows = torch.tensor([[float(i), 2.0 * i] for i in range(64)])
        self.assertEqual(F.pool(rows, "last").tolist(), [63.0, 126.0])
        self.assertEqual(F.pool(rows, "mean").tolist(), [31.5, 63.0])
        b = F.pool(rows, "bins32")
        self.assertEqual(b.shape[0], 64)
        self.assertEqual([b[0].item(), b[1].item(), b[2].item(), b[3].item(), b[62].item(), b[63].item()], [0.5, 1.0, 2.5, 5.0, 62.5, 125.0])
        odd = F.pool(torch.tensor([[float(i)] for i in range(33)]), "bins32")
        self.assertEqual((odd[0].item(), odd[31].item()), (0.0, 31.5))
        with self.assertRaises(ValueError):
            F.pool(rows[:5], "bins32")


class Labels(unittest.TestCase):
    def test_join_keeps_labelled_rows_and_reports_the_rest(self):
        rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        labels = [{"id": "a", "labels": {"list": 1, "refuse": 0}}, {"id": "c", "labels": {"list": 0, "refuse": 1}}]
        kept, names, y, dropped = F.join_labels(rows, labels)
        self.assertEqual([r["id"] for r in kept], ["a", "c"])
        self.assertEqual(names, ["list", "refuse"])
        self.assertEqual(y, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(dropped, ["b"])
        with self.assertRaises(SystemExit):
            F.join_labels(rows, [{"id": "a", "labels": {"x": 1}}, {"id": "a", "labels": {"x": 0}}])
        with self.assertRaises(SystemExit):
            F.join_labels(rows, [{"id": "a", "labels": {"x": 2}}])

    def test_fill_turn(self):
        self.assertEqual(F.fill_turn("<u>{system}{prompt}<a>{think}", "hi"), "<u>hi<a>")


class Fitting(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        n, d = 400, 6
        self.X = torch.randn(n, d)
        w = torch.tensor([[2.0, -1.0, 0, 0, 0, 0], [0, 0, 1.5, 0, 0, -2.0]])
        logit = self.X @ w.T
        self.y = (torch.rand(n, 2) < torch.sigmoid(logit)).float()

    def test_temperature_recovers_a_known_scale(self):
        z = torch.randn(4000, 2) * 3
        y = (torch.rand(4000, 2) < torch.sigmoid(z / 2.0)).float()  # true T = 2
        self.assertAlmostEqual(F.fit_temperature(z, y), 2.0, delta=0.3)

    def test_heads_learn_a_linear_signal_and_have_the_names_the_lib_reads(self):
        for kind, names in (("logistic", {"weight", "bias"}), ("ridge", {"weight", "bias"}), ("mlp", {"w1", "b1", "w2", "b2"})):
            t = F.fit_head(kind, self.X, self.y, epochs=200)
            self.assertEqual(set(t), names, kind)
            z = F.head_scores(t, self.X)
            self.assertEqual(tuple(z.shape), (400, 2))
            agree = ((z > 0).float() == self.y).float().mean().item()
            self.assertGreater(agree, 0.65, kind)  # sanity that the fit moves; not a result

    def test_platt_is_identity_on_calibrated_scores(self):
        z = torch.randn(4000, 2)
        y = (torch.rand(4000, 2) < torch.sigmoid(z)).float()
        a, b = F.fit_platt(z, y)
        self.assertTrue(all(abs(x - 1) < 0.2 for x in a) and all(abs(x) < 0.2 for x in b), (a, b))


NODE_CHECK = """
import { ForecastReader } from %s;
import fs from "node:fs";
const [file, specFile, rowsFile] = process.argv.slice(2);
const spec = JSON.parse(fs.readFileSync(specFile, "utf8")), rows = JSON.parse(fs.readFileSync(rowsFile, "utf8"));
const r = ForecastReader.load(spec, new Uint8Array(fs.readFileSync(file)), spec.hiddenSize);
console.log(JSON.stringify(rows.map((row) => { const o = r.read([Float32Array.from(row)], "q", 1); return Object.values(o.behaviors).map((b) => b.logit); })));
"""


@unittest.skipUnless((LIB / "dist" / "index.js").exists(), "dist/ missing: run npm run build")
class AgainstTheLibrary(unittest.TestCase):
    """The file this module writes, read by the TypeScript reader: same numbers as head_scores + temperature + Platt."""

    def check(self, kind, platt):
        torch.manual_seed(1)
        d, k = 8, 3
        X = torch.randn(50, d) * 3 + 1
        y = (torch.rand(50, k) < 0.5).float()
        mean, std = F.standardize(X)
        t = F.fit_head(kind, (X - mean) / std, y, epochs=20, hidden=5)
        T = 1.7
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            F.write_head(td / "h.safetensors", t, mean, std)
            cal = {"temperature": T}
            if platt:
                cal["platt"] = {"a": [0.5, 1.5, 2.0], "b": [0.1, -0.2, 0.3]}
            spec = {"id": "t", "layer": 1, "pooling": "last", "hiddenSize": d,
                    "head": {"type": "mlp" if "w1" in t else "linear", "file": {"path": "h", "sha256": "0" * 64}, "dims": {"in": d, "out": k, **({"hidden": 5} if "w1" in t else {})}},
                    "behaviors": [{"name": f"b{i}"} for i in range(k)], "calibration": cal, "trainedOn": {"baseId": "x", "weightsSha256": "0" * 64}}
            (td / "spec.json").write_text(json.dumps(spec))
            rows = X[:4]
            (td / "rows.json").write_text(json.dumps(rows.tolist()))
            (td / "check.mjs").write_text(NODE_CHECK % json.dumps((LIB / "dist" / "index.js").as_uri()))
            out = subprocess.run(["node", str(td / "check.mjs"), str(td / "h.safetensors"), str(td / "spec.json"), str(td / "rows.json")], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            got = torch.tensor(json.loads(out.stdout))
        z = F.head_scores(t, (rows - mean) / std) / T
        if platt:
            z = torch.tensor(cal["platt"]["a"]) * z + torch.tensor(cal["platt"]["b"])
        self.assertLess((got - z).abs().max().item(), 1e-3)

    def test_linear(self):
        self.check("logistic", False)

    def test_linear_platt(self):
        self.check("ridge", True)

    def test_mlp(self):
        self.check("mlp", True)


@unittest.skipUnless(os.environ.get("S1S2_TEST_FP32_GRAPH") and os.environ.get("S1S2_TEST_MODEL"),
                     "set S1S2_TEST_FP32_GRAPH (a build exported with --forecast-layer) and S1S2_TEST_MODEL (its checkpoint) and S1S2_TEST_LAYER")
class ExportedGraph(unittest.TestCase):
    def test_aux_output_is_the_output_of_block_L(self):
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoModelForCausalLM, AutoTokenizer
        graph, model_dir, L = Path(os.environ["S1S2_TEST_FP32_GRAPH"]), os.environ["S1S2_TEST_MODEL"], int(os.environ.get("S1S2_TEST_LAYER", "14"))
        tok = AutoTokenizer.from_pretrained(model_dir)
        ids = tok("What is the capital of France? Answer briefly.", return_tensors="pt")["input_ids"]
        s = ort.InferenceSession(str(graph / "model.onnx"), providers=["CPUExecutionProvider"])
        feed = {"input_ids": ids.numpy().astype(np.int64), "attention_mask": np.ones(ids.shape, np.int64)}
        for i in s.get_inputs():
            if i.name.startswith("past_key_values"):
                feed[i.name] = np.zeros((1, i.shape[1], 0, 64), np.float32)
        aux = s.run(["aux_hidden_states"], feed)[0][0]
        m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        with torch.no_grad():
            ref = m(ids, output_hidden_states=True).hidden_states[L + 1][0].numpy()
        self.assertLess(np.abs(aux - ref).max() / max(1.0, np.abs(ref).max()), 1e-4)
        self.assertGreater(np.abs(aux - m(ids, output_hidden_states=True).hidden_states[L][0].detach().numpy()).max(), 1e-2)  # not the neighbour layer

    def test_truncated_graph_gives_the_same_output_with_fewer_nodes(self):
        import numpy as np
        import onnxruntime as ort
        graph = Path(os.environ["S1S2_TEST_FP32_GRAPH"])
        with tempfile.TemporaryDirectory() as td:
            info = F.truncate_graph(graph / "model.onnx", Path(td) / "gate.onnx")
            self.assertLess(info["nodes"], info["nodesFull"])
            full = ort.InferenceSession(str(graph / "model.onnx"), providers=["CPUExecutionProvider"])
            cut = ort.InferenceSession(str(Path(td) / "gate.onnx"), providers=["CPUExecutionProvider"])
            ids = np.array([[1, 5, 9, 12, 7]], np.int64)
            feed = {"input_ids": ids, "attention_mask": np.ones(ids.shape, np.int64)}
            for i in full.get_inputs():
                if i.name.startswith("past_key_values"):
                    feed[i.name] = np.zeros((1, i.shape[1], 0, 64), np.float32)
            a = full.run(["aux_hidden_states"], feed)[0]
            b = cut.run(["aux_hidden_states"], {k: v for k, v in feed.items() if k in {i.name for i in cut.get_inputs()}})[0]
            self.assertLess(np.abs(a - b).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
