"""Small helpers shared by the converter stages."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, p)


def read_json(p):
    return json.loads(Path(p).read_text())


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    os.replace(tmp, p)


def log(stage, msg):
    print(f"[s1s2-convert {stage}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def lib_dir(arg=None):
    """s1s2 checkout (bin/s1s2.mjs + dist/). Default: the parent of the converter directory."""
    d = Path(arg) if arg else Path(__file__).resolve().parents[2]
    if not (d / "bin" / "s1s2.mjs").exists() or not (d / "dist" / "index.js").exists():
        raise SystemExit(f"s1s2 not found at {d} (needs bin/s1s2.mjs and dist/; run `npm ci && npm run build` there, or pass --lib)")
    return d


def run(cmd, stage, env=None, cwd=None):
    log(stage, "$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], env={**os.environ, **(env or {})}, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"{stage}: command failed with exit code {r.returncode}")


def done(work, stage):
    return (Path(work) / f".done-{stage}").exists()


def mark_done(work, stage, info=None):
    write_json(Path(work) / f".done-{stage}", {"at": now_iso(), **(info or {})})


def python_exe():
    return sys.executable
