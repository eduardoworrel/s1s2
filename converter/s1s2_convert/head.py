"""Pointer head and head.bin (spec section 6.1).

Port of the Kev PointerHead (github.com/jaredpalmer/kev, kev/model.py:205-218 @2074cc44, Apache-2.0; see NOTICE), same
contract: h = final post-norm hidden state (decoder last_hidden_state), fp32; query = h at the decide marker's read
position, key k = h at the k-th option marker; z_k = ((Wq h_q + bq) . (Wk h_k + bk)) / sqrt(dp), dp = 256; p = softmax.
head.bin folds 1/sqrt(dp) into q and qb and states temperature 1.0 (s1s2 computes (Wq hq + bq).(Wk hk + bk)/T).
"""
import hashlib
import json
import math
import struct
from pathlib import Path

HEAD_DIM = 256  # kev/train.py:397 --head_dim
SCHEMA = "s1s2.head/1"


def new_head(d, dp=HEAD_DIM):
    import torch
    nn = torch.nn

    class PointerHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
            self.scale = 1 / math.sqrt(dp)

        def forward(self, h_decide, h_opts):  # [d], [K, d] -> logits [K]
            return (self.k(h_opts) @ self.q(h_decide)) * self.scale

    return PointerHead()


def scores(head, h, pos_decide, pos_options):
    """h [L, d] (one row, fp32) -> z [K] at the contract positions."""
    import torch
    return head(h[pos_decide], h[torch.tensor(pos_options, device=h.device)])


def save(head, folder, contract):
    """head.safetensors (fp32) + head.bin (s1s2 format) + HEAD_CONTRACT.json. Returns the contract with sha256s."""
    import torch
    from safetensors.torch import save_file
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().float().cpu().contiguous() for k, v in head.state_dict().items()}
    save_file(sd, str(folder / "head.safetensors"), metadata={"schema": SCHEMA})
    dp, d = sd["q.weight"].shape
    s = 1 / math.sqrt(dp)
    # explicit reader scale (spec 6.1): 1, because 1/sqrt(dp) is folded into q and qb
    meta = json.dumps({"headDim": dp, "hiddenSize": d, "temperature": 1.0, "scale": 1.0, "scale_folded_into_q": s}).encode()
    parts = [struct.pack("<I", len(meta)), meta]
    for t in (sd["q.weight"] * s, sd["q.bias"] * s, sd["k.weight"], sd["k.bias"]):
        parts.append(t.to(torch.float16).numpy().tobytes())
    (folder / "head.bin").write_bytes(b"".join(parts))
    sha = lambda p: hashlib.sha256((folder / p).read_bytes()).hexdigest()  # noqa: E731
    c = {**contract, "schema": SCHEMA, "headDim": dp, "hiddenSize": d, "scale": s, "temperature": 1.0,
         "files": {"head.safetensors": sha("head.safetensors"), "head.bin": sha("head.bin")}}
    (folder / "HEAD_CONTRACT.json").write_text(json.dumps(c, ensure_ascii=False, sort_keys=True, indent=1) + "\n")
    return c


def load(folder, device="cpu"):
    """(head fp32 in eval mode, contract); checks head.safetensors against the contract's sha256."""
    from safetensors.torch import load_file
    folder = Path(folder)
    c = json.loads((folder / "HEAD_CONTRACT.json").read_text())
    if hashlib.sha256((folder / "head.safetensors").read_bytes()).hexdigest() != c["files"]["head.safetensors"]:
        raise SystemExit(f"{folder}: head.safetensors does not match HEAD_CONTRACT.json")
    h = new_head(c["hiddenSize"], c["headDim"])
    h.load_state_dict(load_file(str(folder / "head.safetensors")))
    return h.to(device).eval(), c


def read_head_bin(data):
    """Reader with the same rule as s1s2 src/readout.ts loadHead (tests). -> (meta, q, qb, k, kb) numpy fp32."""
    import numpy as np
    n = struct.unpack("<I", data[:4])[0]
    meta = json.loads(data[4:4 + n])
    off, out = 4 + n, []
    for size in (meta["headDim"] * meta["hiddenSize"], meta["headDim"], meta["headDim"] * meta["hiddenSize"], meta["headDim"]):
        out.append(np.frombuffer(data[off:off + 2 * size], dtype="<f2").astype("float32"))
        off += 2 * size
    if off != len(data):
        raise ValueError(f"head.bin has {len(data) - off} trailing bytes")
    return (meta, *out)
