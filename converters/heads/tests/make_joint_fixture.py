"""Tiny random Clef joint head + the reference's logits, for s1s2 test/joint.test.mjs (run once, output committed)."""
import json, sys
from pathlib import Path
import numpy as np, torch
from s1s2_heads import refs
from s1s2_heads.common import write_safetensors

out = Path(sys.argv[1])
sg = refs.sglang(); jsh = refs.clef_cpu(sg.joint_schema_head)
cfg = {"hidden_size": 16, "width": 8, "routing_layers": 1, "layers": 2, "heads": 2, "feedforward": 12}
torch.manual_seed(0)
head = jsh.JointSchemaHead(cfg["hidden_size"], cfg["width"], cfg["routing_layers"], cfg["layers"], cfg["heads"], cfg["feedforward"]).double().eval()
with torch.no_grad():
    for n, p in head.named_parameters():
        p.copy_(torch.randn_like(p) * (0.5 if p.dim() > 1 else 0.3))
sd = {k: v.float().numpy() for k, v in head.state_dict().items()}
head = head.float()
head.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})
write_safetensors(out / "joint_head.safetensors", sd)
rng = np.random.default_rng(1)
T = 30
hidden = rng.standard_normal((T, 16)).astype(np.float32)
lex_table = (rng.standard_normal((10, 16)) * 0.3).astype(np.float32)
ids = rng.integers(0, 10, T)
qs = [jsh.LayoutQuestion(1, (3, 6), ((7, 9), (10, 12), (13, 14))), jsh.LayoutQuestion(0, (15, 17), ((18, 20), (21, 23))), jsh.LayoutQuestion(2, (24, 25), ((25, 26), (26, 28)))]
with torch.no_grad():
    logits = head.forward_batch(torch.tensor(hidden), torch.tensor(ids), [T], [qs], torch.tensor(lex_table))[0].double().tolist()
json.dump({"config": cfg, "hidden": hidden.tolist(), "lexical": lex_table[ids].tolist(),
           "questions": [{"type": q.question_type, "span": list(q.question_span), "options": [list(s) for s in q.option_spans]} for q in qs],
           "expected": logits, "reference": "sglang@da614ba JointSchemaHead.forward_batch, fp32, CPU SDPA in place of flash attention"},
          open(out / "case.json", "w"))
print(logits)
