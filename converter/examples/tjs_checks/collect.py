import json, glob, os
SC = os.environ.get("TJS_CHECKS_WORK", ".")  # folder with res/<repo name>/{tok,snip,pr,wasm}.json
out = {"note": "Checks behind the transformers.js cards, Node CPU unless stated; prompts from examples/pack_tjs_items.py "
               "(written for the parity check, not a test set).",
       "plain": {"verified": []}, "repos": {}}
for d in sorted(glob.glob(f"{SC}/res/*")):
    name = "eduardoworrel/" + os.path.basename(d)
    r = {}
    if os.path.exists(f"{d}/tok.json"):
        t = json.load(open(f"{d}/tok.json")); r["positions"] = {"items": t["items"], "equal": t["positionsEqual"]}
    if os.path.exists(f"{d}/snip.json"):
        s = json.load(open(f"{d}/snip.json"))
        out["plain"]["verified"].append({"model": os.path.basename(d).split("-")[0] + "-" + os.path.basename(d).split("-")[1],
                                         "items": s["items"], "positionsEqual": s["positionsEqual"],
                                         "decisionsEqual": s["decisionsEqual"], "probMaxAbsDiff": s["probMaxAbsDiff"]})
    if os.path.exists(f"{d}/pr.json"):
        p = json.load(open(f"{d}/pr.json")); r["pipeline"] = {"items": p["items"], "agree": p["pipelineAgreesWithHead"],
                                                               "pr": "huggingface/transformers.js#1943@2c39cfae"}
    if os.path.exists(f"{d}/wasm.json"):
        w = json.load(open(f"{d}/wasm.json"))
        r["wasm"] = {"measured": w["allWasm"], "chrome": w["chrome"], "items": w["items"], "equal": w["sameAsNodeCpu"],
                     "probMaxAbsDiff": w["probMaxAbsDiff"], "secondsPerPrompt": w["secondsPerPrompt"]}
    out["repos"][name] = r
print(json.dumps(out, indent=1))
