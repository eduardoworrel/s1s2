"""Parity of two result files (torch_reference.py / ort_reader.py output) on the same questions: |dp| of the chosen option of the oracle, decision flips, accuracy."""
import json, sys
import numpy as np
a = [json.loads(l) for l in open(sys.argv[1])]; b = {(r["image"], r["instruction"]): r for r in map(json.loads, open(sys.argv[2]))}
dp, flips, ca, cb, n = [], 0, 0, 0, 0
for r in a:
    o = b.get((r["image"], r["instruction"]))
    if o is None: continue
    n += 1; pa, pb = np.array(r["probabilities"]), np.array(o["probabilities"])
    i = int(pa.argmax()); dp.append(abs(pa[i] - pb[i])); flips += int(pa.argmax() != pb.argmax())
    if r.get("label") is not None: ca += int(pa.argmax() == r["label"]); cb += int(pb.argmax() == r["label"])
dp = np.array(dp)
print(json.dumps({"n": n, "median_dp": float(np.median(dp)), "p95_dp": float(np.percentile(dp, 95)), "max_dp": float(dp.max()), "flips": flips, "flip_rate": flips / n, "acc_oracle": ca / n, "acc_other": cb / n}))
