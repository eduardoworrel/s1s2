"""Package assembly: files into the package directory, manifest with builds, heads, evidence, provenance, license."""
import json
import shutil
from pathlib import Path

from .util import sha256_file, write_json


def pinned(path, rel):
    return {"path": rel, "sha256": sha256_file(path), "size": Path(path).stat().st_size}


def assemble(draft, pkg, *, tokenizer_dir, builds, heads, provenance, license_id, evidence=None):
    """builds: [{name_prefix, dir, dtype, devices: [[name_suffix, [devices]]...]}]; heads: {name: (dir, types)}.
    Copies everything into `pkg` and returns the manifest (also written to pkg/manifest.json)."""
    pkg = Path(pkg)
    pkg.mkdir(parents=True, exist_ok=True)
    m = json.loads(json.dumps(draft))
    repo = m["tokenizer"]["repo"]
    rev = m["base"]["revision"]
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(Path(tokenizer_dir) / f, pkg / f)
    m["tokenizer"]["files"] = {"tokenizer": pinned(pkg / "tokenizer.json", "tokenizer.json"),
                               "config": pinned(pkg / "tokenizer_config.json", "tokenizer_config.json")}
    head_specs = {}
    for name, (d, types) in heads.items():
        dst = pkg / "heads" / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("head.bin", "head.safetensors", "HEAD_CONTRACT.json"):
            shutil.copyfile(Path(d) / f, dst / f)
        c = json.loads((dst / "HEAD_CONTRACT.json").read_text())
        if sha256_file(dst / "head.bin") != c["files"]["head.bin"] or c["hiddenSize"] != m["hiddenSize"]:
            raise SystemExit(f"head {name}: head.bin sha256 or hiddenSize differs from HEAD_CONTRACT.json / manifest")
        head_specs[name] = {"file": pinned(dst / "head.bin", f"heads/{name}/head.bin"), "types": types}
    m["builds"] = {}
    for b in builds:
        src, sub = Path(b["dir"]), b["subdir"]
        dst = pkg / sub
        dst.mkdir(parents=True, exist_ok=True)
        g = json.loads((src / "genai_config.json").read_text())
        gd = g["model"]["decoder"]
        if gd["hidden_size"] != m["hiddenSize"]:
            raise SystemExit(f"{src}: genai_config hidden_size {gd['hidden_size']} != {m['hiddenSize']}")
        for f in (gd["filename"], gd["filename"] + ".data", "genai_config.json"):
            shutil.copyfile(src / f, dst / f)
        files = {"model": pinned(dst / gd["filename"], f"{sub}/{gd['filename']}"),
                 "data": pinned(dst / (gd["filename"] + ".data"), f"{sub}/{gd['filename']}.data"),
                 "genaiConfig": pinned(dst / "genai_config.json", f"{sub}/genai_config.json")}
        for suffix, devices in b["devices"]:
            m["builds"][f"genai-{b['dtype']}-{suffix}"] = {
                "runtime": "genai", "repo": repo, "revision": rev, "files": files, "hiddenOutput": "hidden_states",
                **({"heads": head_specs} if head_specs else {}), "devices": devices, "dtype": b["dtype"],
                "kvHeadDim": gd["head_size"], "maxSafeTokens": m["defaults"].get("maxSafeTokens", 2048), "fidelity": None}
    if evidence:
        m["evidence"] = evidence
    m["license"] = license_id
    m["provenance"] = provenance
    write_json(pkg / "manifest.json", m)
    return m
