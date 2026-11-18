"""Repackage a published ONNX Runtime GenAI S1S2 repo into a Transformers.js-loadable folder (same weights).

The GenAI graph runs unchanged in transformers.js (`AutoModelForCausalLM`) once it is laid out the way transformers.js
expects. This tool changes only what the layout needs and never touches a weight byte:

  * graph: `onnx/model_q4.onnx` = the GenAI `model.onnx` proto with (a) the symbolic head dimension of every
    `past_key_values.*` / `present.*` input/output pinned to the KV head size from `genai_config.json` (transformers.js
    builds its empty cache from the graph metadata and resolves unknown symbols to 0, which GroupQueryAttention
    rejects), (b) every initializer's external-data `location` rewritten to `model_q4.onnx_data` and, so the graph also
    runs on the browser CPU (onnxruntime-web WASM), (c) com.microsoft GatherBlockQuantized (int4 tied token table; no
    WASM kernel) replaced by standard ops on the same initializers, and (d) lm_head split along the vocabulary into
    parts whose initializers point at consecutive byte ranges of the same data file (`--lm-head-max-mb`; WASM unpacks
    MatMulNBits weights to fp32 at every run, which for a large vocabulary passes wasm32's 4 GB). Both are exact;
  * weights: `onnx/model_q4.onnx_data` = the GenAI `model.onnx.data`, byte for byte (hard link when possible);
    builds made with exclude_embeds (graph input `inputs_embeds`, token table in a separate fp16 file) get the table
    back as `onnx/model_q4.onnx_data_1` (the same file, byte for byte) and one Gather node from `input_ids`;
  * `config.json` = the base model's config at the pinned revision + `transformers.js_config`
    (`use_external_data_format`), checked against `genai_config.json` (layers, heads, KV heads, head size, vocab);
    `generation_config.json` from the base when it has one;
  * tokenizer files, chat template, heads (`heads/<name>/head.bin` + `head.safetensors`), LICENSE*, NOTICE copied;
  * one manifest per source manifest (`manifest*.json`, same names), with a single `transformers.js` build.

Stages:
  pack    download the source repo at a pinned revision and write the folder (manifests pin a placeholder revision)
  parity  run the source GenAI graph and the repackaged graph on the same prompts (Node, onnxruntime CPU, one model
          at a time) and compare the hidden state at every read position and the head decisions; writes
          parity.json and the build's `fidelity` into the manifests
  pin     after the weights/files commit is on the Hub: write that commit sha into the manifests
  card    README.md: front matter and License/Credits from the source card, parity numbers from parity.json
  update  manifests + card only (same files commit), e.g. after a browser run: --devices cpu,webgpu --browser "..."
  publish create the new public repo, upload the files (commit 1), pin the manifests to that commit, write the card,
          upload manifests + card (commit 2), verify every remote file against the local sha256 / git blob id

Example:
  python -m s1s2_convert.pack_tjs pack --repo eduardoworrel/Qwen3-0.6B-ContractNLI-CLINC-SIB-ONNX-genai-int4-S1S2 \\
      --out /tmp/tjs --name eduardoworrel/Qwen3-0.6B-ContractNLI-CLINC-SIB-ONNX-transformersjs-S1S2
  python -m s1s2_convert.pack_tjs parity --out /tmp/tjs --items items.jsonl
  python -m s1s2_convert.pack_tjs pin --out /tmp/tjs --revision <40-hex sha of the files commit>
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .util import log, write_json, read_json, sha256_file, lib_dir, now_iso

STAGE = "pack-tjs"
GRAPH = "model_q4.onnx"
DATA = GRAPH + "_data"
BUILD = "transformersjs-q4"
PLACEHOLDER = "0" * 40
COPY_ROOT = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "NOTICE"]
SOURCE_PATTERNS = ["model.onnx", "model.onnx.data", "embed_tokens*.bin", "genai_config.json", "manifest*.json", "tokenizer.json",
                   "tokenizer_config.json", "tokenizer/*", "chat_template.jinja", "heads/*/head.bin",
                   "heads/*/head.safetensors", "LICENSE*", "NOTICE", "README.md"]
# Manifest `lab` keys kept (the quick start reads them); everything else in `lab` is source bookkeeping.
LAB_KEEP = ["task", "language", "cue", "state", "question", "options", "labels", "e1", "e2"]


def pfile(root, rel):
    p = Path(root) / rel
    return {"path": rel, "sha256": sha256_file(p), "size": p.stat().st_size}


def link_or_copy(src, dst):
    src = Path(os.path.realpath(src))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def add_embedding_gather(g, size, hidden):
    """exclude_embeds builds: the graph takes inputs_embeds and the token table ships as a separate fp16 file
    ([vocab, hidden], row-major). Put it back in the graph as an external initializer stored in the second data chunk
    (the same file, byte for byte) and compute inputs_embeds = Gather(table, input_ids)."""
    import onnx
    from onnx import TensorProto, helper
    emb_in = next(i for i in g.input if i.name == "inputs_embeds")
    tt = emb_in.type.tensor_type
    if tt.elem_type != TensorProto.FLOAT16 or tt.shape.dim[2].dim_value != hidden or size % (hidden * 2):
        raise SystemExit(f"inputs_embeds {tt.elem_type}/{tt.shape.dim[2].dim_value} vs table {size} bytes, hidden {hidden}")
    vocab = size // (hidden * 2)
    t = TensorProto(name="model.embed_tokens.weight", data_type=TensorProto.FLOAT16, dims=[vocab, hidden])
    t.data_location = TensorProto.EXTERNAL
    for k, v in (("location", DATA + "_1"), ("offset", "0"), ("length", str(size))):
        e = t.external_data.add(); e.key, e.value = k, v
    g.initializer.append(t)
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [tt.shape.dim[0].dim_param or 1, tt.shape.dim[1].dim_param or None])
    pos = list(g.input).index(emb_in)
    g.input.remove(emb_in)
    g.input.insert(pos, ids)
    g.node.insert(0, helper.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], ["inputs_embeds"], name="/model/embed_tokens/Gather", axis=0))
    return vocab


def unpack_gather_block_quantized(g):
    """Browser CPU (onnxruntime-web WASM) has no kernel for com.microsoft GatherBlockQuantized, which GenAI builds with
    tied embeddings use to read token rows from the int4 lm_head table. Replace it by standard ONNX ops on the same
    initializers: Gather the packed uint8 rows and their fp16 scales, split each byte into its two 4-bit values (low
    nibble first), (q - 8) * scale in fp32, cast to fp16. Products of a 4-bit integer and an fp16 scale are exact in fp32,
    so the rows equal the kernel's. Returns the number of nodes replaced (0 or 1)."""
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np
    nodes = [n for n in g.node if n.op_type == "GatherBlockQuantized"]
    if not nodes:
        return 0
    if len(nodes) != 1:
        raise SystemExit(f"{len(nodes)} GatherBlockQuantized nodes; expected one (token table)")
    n = nodes[0]
    at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    if len(n.input) != 3 or at.get("bits") != 4 or at.get("gather_axis", 0) != 0 or at.get("quantize_axis") != 1:
        raise SystemExit(f"GatherBlockQuantized {n.name}: unsupported form {list(n.input)} {at}")
    scales = next(t for t in g.initializer if t.name == n.input[2])
    vocab, nb = scales.dims
    bs = at["block_size"]
    data, ids, sc = n.input
    p = n.name.rsplit("/", 1)[0] + "/unpack"
    consts = {"sh_bytes": [0, 0, nb, bs // 2], "sh_q": [0, 0, nb, bs], "sh_out": [0, 0, nb * bs], "ax": [-1]}
    inits = [numpy_helper.from_array(np.array(v, dtype=np.int64), f"{p}/{k}") for k, v in consts.items()]
    inits += [numpy_helper.from_array(np.array(v, dtype=np.float32), f"{p}/{k}") for k, v in (("c16", 16.0), ("c8", 8.0))]
    o = lambda k: f"{p}/{k}"  # noqa: E731
    mk = helper.make_node
    new = [
        mk("Gather", [data, ids], [o("rows")], name=o("Gather"), axis=0),
        mk("Reshape", [o("rows"), o("sh_bytes")], [o("bytes")], name=o("Reshape")),
        mk("Cast", [o("bytes")], [o("bytes_f")], name=o("Cast"), to=TensorProto.FLOAT),
        mk("Div", [o("bytes_f"), o("c16")], [o("div")], name=o("Div")),
        mk("Floor", [o("div")], [o("hi")], name=o("Floor")),
        mk("Mul", [o("hi"), o("c16")], [o("hi16")], name=o("Mul_hi")),
        mk("Sub", [o("bytes_f"), o("hi16")], [o("lo")], name=o("Sub_lo")),
        mk("Unsqueeze", [o("lo"), o("ax")], [o("lo1")], name=o("Unsqueeze_lo")),
        mk("Unsqueeze", [o("hi"), o("ax")], [o("hi1")], name=o("Unsqueeze_hi")),
        mk("Concat", [o("lo1"), o("hi1")], [o("pairs")], name=o("Concat"), axis=-1),
        mk("Reshape", [o("pairs"), o("sh_q")], [o("q")], name=o("Reshape_q")),
        mk("Sub", [o("q"), o("c8")], [o("qc")], name=o("Sub_zp")),
        mk("Gather", [sc, ids], [o("s16")], name=o("Gather_scales"), axis=0),
        mk("Cast", [o("s16")], [o("s32")], name=o("Cast_scales"), to=TensorProto.FLOAT),
        mk("Unsqueeze", [o("s32"), o("ax")], [o("s")], name=o("Unsqueeze_scales")),
        mk("Mul", [o("qc"), o("s")], [o("deq")], name=o("Mul_scales")),
        mk("Reshape", [o("deq"), o("sh_out")], [o("emb32")], name=o("Reshape_out")),
        mk("Cast", [o("emb32")], [n.output[0]], name=o("Cast_out"), to=TensorProto.FLOAT16),
    ]
    i = list(g.node).index(n)
    g.node.remove(n)
    for k, m in enumerate(new):
        g.node.insert(i + k, m)
    g.initializer.extend(inits)
    return 1


def split_lm_head(g, max_bytes):
    """Browser CPU (onnxruntime-web WASM) runs MatMulNBits by unpacking the whole int4 weight to fp32 at every run
    (K * N * 4 bytes): for lm_head over a large vocabulary that one buffer (e.g. 2.1 GB at 256k x 2048) plus the
    weights passes the 4 GB of wasm32 memory. Split lm_head along the vocabulary into MatMulNBits nodes of at most
    `max_bytes` of fp32 each, whose weights and scales are external initializers pointing at consecutive byte ranges
    of the same data file (no byte copied or changed), and Concat the logits. Each output column is computed exactly
    as before. Returns the number of parts (1 = unchanged)."""
    import onnx
    from onnx import TensorProto, helper
    n = next((x for x in g.node if x.op_type == "MatMulNBits" and "logits" in x.output), None)
    if n is None or not max_bytes:
        return 1
    at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    K, N = at["K"], at["N"]
    rows = max(1, max_bytes // (4 * K))
    if N <= rows:
        return 1
    if len(n.input) != 3:
        raise SystemExit(f"{n.name}: lm_head with zero points / g_idx / bias not supported for splitting")
    inits = {t.name: t for t in g.initializer}
    w, s = inits[n.input[1]], inits[n.input[2]]
    for t in (w, s):
        if t.data_location != TensorProto.EXTERNAL or t.dims[0] != N:
            raise SystemExit(f"{t.name}: expected an external initializer with {N} rows")
    ext = lambda t: {e.key: e.value for e in t.external_data}  # noqa: E731
    elem = {TensorProto.UINT8: 1, TensorProto.FLOAT16: 2, TensorProto.FLOAT: 4}

    def part(t, r0, r1, name):
        e = ext(t)
        row = elem[t.data_type]
        for d in t.dims[1:]:
            row *= d
        if int(e["length"]) != row * N:
            raise SystemExit(f"{t.name}: external length {e['length']} != {row * N}")
        c = TensorProto(name=name, data_type=t.data_type, dims=[r1 - r0, *t.dims[1:]])
        c.data_location = TensorProto.EXTERNAL
        for k, v in (("location", e["location"]), ("offset", str(int(e.get("offset", 0)) + r0 * row)), ("length", str((r1 - r0) * row))):
            x = c.external_data.add(); x.key, x.value = k, v
        return c

    pre = n.name.rsplit("/", 1)[0]
    parts, outs, new_inits = [], [], []
    for k, r0 in enumerate(range(0, N, rows)):
        r1 = min(N, r0 + rows)
        wn, sn = f"{n.input[1]}.part{k}", f"{n.input[2]}.part{k}"
        new_inits += [part(w, r0, r1, wn), part(s, r0, r1, sn)]
        out = f"{pre}/logits_part{k}"
        a = {**at, "N": r1 - r0}
        parts.append(helper.make_node("MatMulNBits", [n.input[0], wn, sn], [out], name=f"{n.name}_part{k}", domain=n.domain, **a))
        outs.append(out)
    parts.append(helper.make_node("Concat", outs, ["logits"], name=f"{pre}/logits_Concat", axis=-1))
    i = list(g.node).index(n)
    g.node.remove(n)
    for k, m in enumerate(parts):
        g.node.insert(i + k, m)
    g.initializer.extend(new_inits)
    used = {x for m in g.node for x in m.input}
    for t in (w, s):
        if t.name not in used:  # untied lm_head: the full tensors are no longer read
            g.initializer.remove(t)
    return len(parts) - 1


def rewrite_graph(src_onnx, dst_onnx, head_size, embeddings=None, lm_head_max_bytes=None):
    """Pin the KV head dim and point every external initializer at DATA (+ the embedding table at DATA_1 when the
    build excluded it). Returns a summary dict."""
    import onnx
    m = onnx.load(str(src_onnx), load_external_data=False)
    g = m.graph
    in_names = [i.name for i in g.input]
    out_names = [o.name for o in g.output]
    vocab = None
    if "input_ids" not in in_names:
        if "inputs_embeds" not in in_names or not embeddings:
            raise SystemExit(f"{src_onnx}: graph has neither input_ids nor inputs_embeds + an embeddings table")
    if "logits" not in out_names or "hidden_states" not in out_names:
        raise SystemExit(f"{src_onnx}: expected outputs logits and hidden_states, got {out_names[:4]}...")
    pinned, symbols = 0, set()
    for vi in list(g.input) + list(g.output):
        if not re.match(r"^(past_key_values|present)\.", vi.name):
            continue
        dims = vi.type.tensor_type.shape.dim
        if len(dims) != 4:
            raise SystemExit(f"{vi.name}: expected 4 dims, got {len(dims)}")
        d = dims[3]
        if d.HasField("dim_value"):
            if d.dim_value != head_size:
                raise SystemExit(f"{vi.name}: head dim {d.dim_value} != genai_config head_size {head_size}")
            continue
        symbols.add(d.dim_param)
        d.Clear()
        d.dim_value = head_size
        pinned += 1
    if "input_ids" not in in_names:
        vocab = add_embedding_gather(g, *embeddings)
    unpacked = unpack_gather_block_quantized(g)
    lm_parts = split_lm_head(g, lm_head_max_bytes)
    locs, n_ext = set(), 0
    for t in g.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            continue
        if t.name == "model.embed_tokens.weight" and vocab:
            continue
        n_ext += 1
        for kv in t.external_data:
            if kv.key == "location":
                locs.add(kv.value)
                kv.value = DATA
    for node in g.node:  # constants stored externally (not expected in GenAI graphs, refused if present)
        for a in node.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_location == onnx.TensorProto.EXTERNAL:
                raise SystemExit(f"node {node.name}: external tensor attribute; not supported")
    if len(locs) != 1:
        raise SystemExit(f"{src_onnx}: expected one external data file, found {sorted(locs)}")
    Path(dst_onnx).parent.mkdir(parents=True, exist_ok=True)
    Path(dst_onnx).write_bytes(m.SerializeToString())
    return {"kvDimsPinned": pinned, "kvDimSymbols": sorted(symbols), "externalInitializers": n_ext, "embeddingGather": vocab,
            "embeddingUnpacked": unpacked or None, "lmHeadParts": lm_parts if lm_parts > 1 else None,
            "sourceDataLocation": locs.pop(), "inputs": len(in_names), "outputs": out_names[:2] + ["..."]}


def check_config(cfg, gc):
    """Base config.json vs genai_config.json decoder: the graph must match the architecture transformers.js reads."""
    dec = gc["model"]["decoder"]
    tc = cfg.get("text_config", cfg)
    heads = tc["num_attention_heads"]
    want = {
        "num_hidden_layers": dec["num_hidden_layers"],
        "num_attention_heads": dec["num_attention_heads"],
        "num_key_value_heads": dec["num_key_value_heads"],
        "hidden_size": dec["hidden_size"],
        "vocab_size": gc["model"]["vocab_size"],
    }
    bad = {k: (tc.get(k), v) for k, v in want.items() if tc.get(k) != v}
    hd = tc.get("head_dim") or tc["hidden_size"] // heads
    if hd != dec["head_size"]:
        bad["head_dim"] = (hd, dec["head_size"])
    if bad:
        raise SystemExit(f"base config.json does not match genai_config.json: {bad}")
    return dec["head_size"]


def graph_changes(g):
    """Compute changes to the GenAI graph beyond metadata, in words (manifest provenance and card)."""
    r = []
    if g.get("embeddingUnpacked"):
        r.append("token rows read from the int4 table with standard ONNX ops instead of com.microsoft GatherBlockQuantized "
                 "(no browser CPU kernel), same values")
    if g.get("lmHeadParts"):
        r.append(f"lm_head split into {g['lmHeadParts']} parts along the vocabulary (same weight bytes, logits concatenated) "
                 "so the browser CPU fits in wasm32 memory")
    return r


def new_manifest(src, name, suffix, heads, files, sibling, note, g=None):
    m = json.loads(json.dumps(src))
    m["schema"] = "s1s2.manifest/1"
    short = name.split("/")[-1].lower()
    m["id"] = f"{short}.{suffix}" if suffix else short
    m["tokenizer"]["repo"] = name
    m["tokenizer"]["revision"] = PLACEHOLDER
    sb = next(iter(src["builds"].values()))
    m["builds"] = {BUILD: {
        "runtime": "transformers.js", "repo": name, "revision": PLACEHOLDER, "files": files,
        "hiddenOutput": sb.get("hiddenOutput", "hidden_states"), "heads": heads(sb.get("heads", {})),
        "devices": ["cpu"], "dtype": "q4", "fidelity": None}}
    lab = src.get("lab") or {}
    keep = {k: lab[k] for k in LAB_KEEP if k in lab}
    if keep:
        m["lab"] = {"note": note, **keep}
    else:
        m.pop("lab", None)
    m["provenance"] = {
        "generator": {"name": "s1s2-convert pack_tjs", "version": __version__},
        "createdAt": now_iso(),
        "base": dict(src["base"]),
        "source": {"format": "onnxruntime-genai", "repo": sibling["repo"], "revision": sibling["revision"],
                   "files": ["model.onnx", "model.onnx.data", "genai_config.json"]},
        "builds": {BUILD: {"tool": "s1s2-convert pack_tjs", "version": __version__,
                           "args": ["pin past_key_values/present head dim", "external data location -> " + DATA, *graph_changes(g or {})]}},
        "notes": "Same weights as the source ONNX Runtime GenAI build; " + (
            "the graph changes are listed in args." if graph_changes(g or {}) else "only the graph metadata and file layout changed."),
    }
    return m


def stage_pack(a):
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    api = HfApi()
    rev = a.revision or api.model_info(a.repo).sha
    snap = Path(snapshot_download(a.repo, revision=rev, allow_patterns=SOURCE_PATTERNS))
    out = Path(a.out)
    # Hard-linked mirror of the snapshot: onnxruntime refuses external data reached through the Hub cache's symlinks.
    src = out.parent / f".{out.name}.source"
    shutil.rmtree(src, ignore_errors=True)
    for p in snap.rglob("*"):
        if p.is_file():
            link_or_copy(p, src / p.relative_to(snap))
    if out.exists() and any(out.iterdir()) and not a.force:
        raise SystemExit(f"{out} is not empty (pass --force)")
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or a.repo.replace("-ONNX-genai-int4-S1S2", "-ONNX-transformersjs-S1S2")
    man_paths = sorted(src.glob("manifest*.json"))
    if not man_paths:
        raise SystemExit(f"{a.repo}: no manifest*.json at the root")
    m0 = read_json(src / "manifest.json")
    gc = read_json(src / "genai_config.json")
    base = m0["base"]
    cfg = json.loads(Path(hf_hub_download(base["repo"], "config.json", revision=base["revision"])).read_text())
    head_size = check_config(cfg, gc)
    log(STAGE, f"{a.repo}@{rev[:8]} -> {out} ({name}); base {base['repo']}@{base['revision'][:8]}, KV head size {head_size}")

    ef = next(iter(m0["builds"].values()))["files"].get("embeddings")
    emb = (ef["size"], m0["hiddenSize"]) if ef else None
    g = rewrite_graph(src / "model.onnx", out / "onnx" / GRAPH, head_size, emb, a.lm_head_max_mb * 2**20)
    if g["sourceDataLocation"] != "model.onnx.data":
        raise SystemExit(f"graph points at {g['sourceDataLocation']}, expected model.onnx.data")
    link_or_copy(src / "model.onnx.data", out / "onnx" / DATA)
    if ef:
        link_or_copy(src / ef["path"], out / "onnx" / (DATA + "_1"))
    cfg["transformers.js_config"] = {"use_external_data_format": {GRAPH: 2 if ef else 1}}
    write_json(out / "config.json", cfg)
    files = {"model": pfile(out, f"onnx/{GRAPH}"), "data": pfile(out, f"onnx/{DATA}"), "config": pfile(out, "config.json")}
    if ef:
        files["data_1"] = pfile(out, f"onnx/{DATA}_1")
        if files["data_1"]["sha256"] != ef["sha256"]:
            raise SystemExit("embeddings table changed while copying")
    try:
        gen = Path(hf_hub_download(base["repo"], "generation_config.json", revision=base["revision"]))
        shutil.copyfile(gen, out / "generation_config.json")
        files["generationConfig"] = pfile(out, "generation_config.json")
    except Exception as e:  # noqa: BLE001 - optional file
        log(STAGE, f"no generation_config.json in the base ({type(e).__name__})")
    if files["data"]["sha256"] != sha256_file(src / "model.onnx.data"):
        raise SystemExit("weights changed while copying")

    for rel in COPY_ROOT:
        if (src / rel).exists():
            shutil.copyfile(src / rel, out / rel)
    for p in sorted(src.glob("LICENSE*")) + sorted((src / "tokenizer").glob("*")) + sorted(src.glob("heads/*/head.*")):
        rel = p.relative_to(src)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out / rel)
    if (out / "NOTICE").exists():
        with open(out / "NOTICE", "a") as f:
            f.write(f"\nTransformers.js layout: the ONNX Runtime GenAI build of {a.repo} (revision {rev}) repackaged "
                    "without changing the weights (graph metadata and file names only), by Eduardo Worrel, 2026.\n")

    def heads(hs):
        r = {}
        for k, h in hs.items():
            if not (out / h["file"]["path"]).exists():
                raise SystemExit(f"head {k}: {h['file']['path']} missing in the source repo")
            r[k] = {**h, "file": pfile(out, h["file"]["path"])}
            if r[k]["file"]["sha256"] != h["file"]["sha256"]:
                raise SystemExit(f"head {k}: sha256 differs from the source manifest")
        return r

    note = (f"Alternative Transformers.js layout of {a.repo}; prompts, heads and option words are the same. "
            "Ask with type 'choice' and the option words in `options` (or `labels`).")
    sibling = {"repo": a.repo, "revision": rev}
    written = []
    for mp in man_paths:
        sm = read_json(mp)
        suffix = mp.name[len("manifest."):-len(".json")] if mp.name != "manifest.json" else (sm.get("lab", {}).get("task") or sm.get("lab", {}).get("language") or "")
        write_json(out / mp.name, new_manifest(sm, name, suffix, heads, files, sibling, note, g))
        written.append(mp.name)
    write_json(out / ".pack-tjs.json", {"source": sibling, "sourceDir": str(src), "name": name, "graph": g,
                                        "headSize": head_size, "manifests": written, "createdAt": now_iso()})
    log(STAGE, f"graph: {g}; manifests: {written}")


def stage_parity(a):
    """Two Node processes, one model at a time: source GenAI graph, then the repackaged graph; then compare."""
    out = Path(a.out)
    meta = read_json(out / ".pack-tjs.json")
    lib = lib_dir(a.lib)
    node = Path(__file__).resolve().parent / "node" / "tjs_parity.mjs"
    res = {}
    for side in ("genai", "tjs"):
        dst = out.parent / f".{out.name}.parity-{side}.json"
        cmd = ["node", str(node), side, str(lib), meta["sourceDir"], str(out), str(a.items), str(dst)]
        log(STAGE, "$ " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        res[side] = read_json(dst)
    ref, got = res["genai"], res["tjs"]
    rows, maxd, same, n = [], 0.0, 0, 0
    for r in ref["rows"]:
        t = next(x for x in got["rows"] if x["id"] == r["id"])
        if r["ids"] != t["ids"] or r["positions"] != t["positions"]:
            raise SystemExit(f"{r['id']}: token ids / read positions differ between the two loads")
        d = max(max(abs(x - y) for x, y in zip(h1, h2)) for h1, h2 in zip(r["hiddens"], t["hiddens"])) if r["hiddens"] else 0.0
        dp = max(abs(x - y) for x, y in zip(r["probabilities"], t["probabilities"]))
        maxd = max(maxd, d)
        n += 1
        same += r["choice"] == t["choice"]
        rows.append({"id": r["id"], "manifest": r["manifest"], "tokens": len(r["ids"]), "readPositions": len(r["positions"]),
                     "hiddenMaxAbsDiff": d, "probMaxAbsDiff": dp, "genai": r["choice"], "transformersjs": t["choice"]})
    summary = {"reference": "source ONNX Runtime GenAI graph, onnxruntime-node CPU (single prefill)",
               "candidate": "repackaged graph through transformers.js AutoModelForCausalLM, Node CPU",
               "items": n, "decisionsEqual": same, "hiddenMaxAbsDiff": maxd,
               "probMaxAbsDiff": max(r["probMaxAbsDiff"] for r in rows), "versions": got.get("versions"),
               "measuredAt": now_iso(), "rows": rows}
    write_json(out.parent / f".{out.name}.parity.json", summary)
    log(STAGE, f"parity: {same}/{n} decisions equal, hidden max |diff| {maxd:g}, prob max |diff| {summary['probMaxAbsDiff']:g}")
    by = {}
    for r in rows:
        by.setdefault(r["manifest"], []).append(r)
    for mn in meta["manifests"]:
        mp = out / mn
        m = read_json(mp)
        rs = by.get(mn)
        if not rs:  # e.g. manifest.json = a copy of one of the per-task manifests: use the rows of that twin
            key = lambda x: (x.get("lab", {}).get("task"), x.get("lab", {}).get("language"), json.dumps(x["builds"][BUILD]["heads"], sort_keys=True))  # noqa: E731
            twin = next((o for o in by if key(read_json(out / o)) == key(m)), None)
            if not twin:
                raise SystemExit(f"parity: no rows for {mn} and no twin manifest")
            rs = by[twin]
        b = m["builds"][BUILD]
        b["fidelity"] = {"reference": "onnxruntime-genai-int4-source-graph", "n": len(rs),
                         "decisionChanged": {"head": sum(r["genai"] != r["transformersjs"] for r in rs) / len(rs)},
                         "measuredAt": summary["measuredAt"],
                         "notes": (f"Same weights as {meta['source']['repo']}. Node CPU, {len(rs)} prompts written for this check "
                                   f"(not a test set): hidden state at the read positions max |diff| "
                                   f"{max(r['hiddenMaxAbsDiff'] for r in rs):g}, head probability max |diff| "
                                   f"{max(r['probMaxAbsDiff'] for r in rs):g}.")}
        m.setdefault("provenance", {})["parity"] = {BUILD: {k: summary[k] for k in ("reference", "candidate", "items", "decisionsEqual", "hiddenMaxAbsDiff", "probMaxAbsDiff", "measuredAt")}}
        write_json(mp, m)
    if same != n:
        raise SystemExit("parity: decisions differ")


def stage_pin(a):
    out = Path(a.out)
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision or ""):
        raise SystemExit("--revision: 40-hex commit sha of the files commit")
    meta = read_json(out / ".pack-tjs.json")
    for mn in meta["manifests"]:
        m = read_json(out / mn)
        m["tokenizer"]["revision"] = a.revision
        b = m["builds"][BUILD]
        b["revision"] = a.revision
        if a.devices:
            b["devices"] = a.devices.split(",")
        if a.browser and b.get("fidelity") and a.browser not in b["fidelity"]["notes"]:
            b["fidelity"]["notes"] += " " + a.browser
        write_json(out / mn, m)
    lib = lib_dir(a.lib)
    only = ["--manifest-only"] if meta.get("hydratedAt") else []  # hydrated folders hold manifests + card only, no weights
    subprocess.run(["node", str(lib / "bin" / "s1s2.mjs"), "validate", "--strict", *only, *[str(out / mn) for mn in meta["manifests"]]], check=True)
    log(STAGE, f"pinned {meta['manifests']} to {a.revision}")


def _front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    return (m.group(1), text[m.end():]) if m else ("", text)


def _section(body, title):
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", body, re.S | re.M)
    return m.group(1).strip() if m else None


LIB_URL = "https://github.com/eduardoworrel/s1s2/blob/1ecbd80d380a78d43056fa10e0c95745261194a5/"
PLAIN_SNIPPET = Path(__file__).resolve().parent / "node" / "tjs_plain_decide.js"


def _qs_manifest(mans):
    return mans[0] if len(mans) == 1 else "manifest.json"


def _twin(out, mans, mn):
    """The per-task manifest that manifest.json copies (same lab and heads), whose check rows are named after it."""
    if len(mans) == 1:
        return mn
    m = read_json(out / mn)
    key = lambda x: (json.dumps(x.get("lab", {}), sort_keys=True), json.dumps(x["builds"][BUILD]["heads"], sort_keys=True))  # noqa: E731
    return next(o for o in mans if o != mn and key(read_json(out / o)) == key(m))


def _example_item(lab):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pack_tjs_items", Path(__file__).resolve().parents[1] / "examples" / "pack_tjs_items.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.items_for("x", lab, 1)[0]


def _quick_start(name, mans, out, chk):
    """s1s2 on a real input (the first check prompt of the quick-start manifest): decision, probability, margin, explanation."""
    mn = _qs_manifest(mans)
    lab = read_json(out / mn).get("lab") or {}
    it = _example_item(lab)
    ex = chk["repo"].get("examples", {}).get(_twin(out, mans, mn))
    js = lambda x: json.dumps(x, ensure_ascii=False)  # noqa: E731
    choices = "model.manifest.lab.options" if lab.get("options") else "[model.manifest.lab.labels.yes, model.manifest.lab.labels.no]"
    res = (f" // {ex['choice'].strip()} {ex['confidence']:.2f} {ex['margin']:.2f} (measured, Node CPU"
           + ("; below the threshold, so it goes to review)" if ex["margin"] < 0.2 else ")")) if ex else ""
    others = [x for x in mans if x not in (mn, _twin(out, mans, mn))]
    return [
        "## Quick start", "",
        "```js",
        'import { S1S2 } from "@s1s2/web"; // github.com/eduardoworrel/s1s2 (needs @huggingface/transformers installed)',
        f'const model = await S1S2.load({{ manifest: "https://huggingface.co/{name}/resolve/main/{mn}" }});',
        "const s = await model.decide({",
        f"  input: {js(it['input'])},",
        f'  questions: [{{ name: "q", type: "choice", choices: {choices},',
        f"    instructions: {js(it['question']['instructions'])} }}],",
        "});",
        "const { choice, confidence, margin } = s.answers[0];",
        f"console.log(choice, confidence.toFixed(2), margin.toFixed(2));{res}",
        "if (margin < 0.2) { // low margin = don't trust the one-pass decision alone (0.2 is an example; measure yours)",
        '  const why = await s.generate("Explain the decision in two sentences.", { maxNewTokens: 96 }); // same weights',
        '  console.log("needs review:", why.text);',
        "}",
        "await s.dispose(); await model.dispose();",
        "```", "",
        "Use the option words and question wording of the manifest (`lab`): the head was trained on them."
        + (f" Other manifests (tasks or languages): {', '.join('`' + x + '`' for x in others)}." if others else ""), "",
    ]


def _plain_section(name, mans, man, chk, tjs_v):
    """'Without s1s2': the card example through transformers.js pipeline("decision") (PR #1943, draft), then where the
    head readout lives in s1s2. Agreement of that pipeline with the head is stated when it is not n/n."""
    lab = man.get("lab") or {}
    it = _example_item(lab)
    js = lambda x: json.dumps(x, ensure_ascii=False)  # noqa: E731
    opts = [o.strip() for o in it["question"]["choices"]]
    crit = ", ".join(f"{o if re.fullmatch(r'[A-Za-z_]\w*', o) else js(o)}: null" for o in opts)
    code = [
        'import { pipeline } from "@huggingface/transformers";',
        f'const decide = await pipeline("decision", "{name}", {{ dtype: "q4" }});',
        f'decide.tokenizer.chat_template ??= await (await fetch("https://huggingface.co/{name}/resolve/main/chat_template.jinja")).text(); // shipped as a file',
        "const { q } = await decide({",
        f"  state: {js(it['input'])},",
        f"  questions: {{ q: {{ type: \"choice\", instructions: {js(it['question']['instructions'])}, criteria: {{ {crit} }} }} }},",
        '}); // pipeline("decision"): transformers.js PR #1943 (draft) — https://github.com/huggingface/transformers.js/pull/1943',
        "// Without the PR, this step needs the s1s2 code (prompt, read positions, head on hidden_states, softmax):",
        f"// {LIB_URL}src/decide.ts#L85-L139",
        "console.log(q.choice, q.probabilities[q.choice]);",
    ]
    pr = chk["repo"].get("pipeline") or {}
    note = []
    if pr and pr["agree"] < pr["items"]:
        note = [f"`pipeline(\"decision\")` reads option-label logits, not the trained head: on this model it agrees with the head "
                f"on {pr['agree']} of {pr['items']} check prompts (Node CPU), so use s1s2 here.", ""]
    return ["## Without s1s2", "", "```js", *code, "```", "", *note]


def _wasm_line(chk):
    """Card line for the browser CPU (onnxruntime-web WASM, no WebGPU): yes/no first, then time and memory."""
    w = chk["repo"].get("wasm")
    if w and w.get("measured"):
        mem = (f", page memory about {w['memoryGB']:.1f} GB" if w.get("memoryGB") else "") + (
            f" (tab peak {w['tabPeakGB']:.1f} GB while loading)" if w.get("tabPeakGB") else "")
        return (f"- Runs on the browser CPU (no WebGPU): yes. Chrome {w['chrome'].split('.')[0]} on an Apple M2 Pro with s1s2: "
                f"about {w['secondsPerPrompt']:.1f} s per decision{mem}, {w['equal']}/{w['items']} decisions equal to Node CPU"
                + (f" ({w['equalWebgpu']}/{w['items']} to WebGPU: the same low-margin flips)" if w["equal"] < w["items"] and w.get("equalWebgpu") else "") + "; "
                "the manifests list `wasm`.")
    return "- Runs on the browser CPU (no WebGPU): " + ((w or {}).get("note") or "not measured.")


def _graph_limits(meta):
    """Limits line on what changed in the GenAI graph."""
    g = meta["graph"]
    ch = ["KV cache head dimension pinned in the graph (transformers.js needs it)"]
    if g.get("embeddingGather"):
        ch.append(f"one Gather node added that reads the token table (`onnx/{DATA}_1`)")
    ch += graph_changes(g)
    return "- " + "; ".join(ch) + "; everything else is the GenAI graph."


def stage_card(a):
    import yaml  # PyYAML ships with huggingface_hub
    out = Path(a.out)
    meta = read_json(out / ".pack-tjs.json")
    par = read_json(out.parent / f".{out.name}.parity.json")
    sib, name = meta["source"]["repo"], meta["name"]
    fm, body = _front_matter((Path(meta["sourceDir"]) / "README.md").read_text())
    front = yaml.safe_load(fm) or {}
    on_gpu = "webgpu" in read_json(out / meta["manifests"][0])["builds"][BUILD]["devices"]
    tags = [t for t in front.get("tags", []) if t not in ("onnxruntime-genai",) and (on_gpu or t != "webgpu")]
    for t in ("transformers.js", "onnx"):
        if t not in tags:
            tags.insert(0, t)
    front["tags"] = tags
    front["library_name"] = "s1s2"
    h1 = re.search(r"^# (.+)$", body, re.M).group(1)
    title = re.sub(r"\s*\([^()]*\)\s*$", "", h1) + " (Transformers.js layout, ONNX int4, S1S2)"
    lic, cred = _section(body, "License"), _section(body, "Credits")
    if not lic or not cred:
        raise SystemExit("source card has no License / Credits section")
    mans = meta["manifests"]
    sib_url = f"https://huggingface.co/{sib}"
    browser = a.browser or meta.get("browser") or "Browser (WebGPU) not run for this layout yet; the reference GenAI build is the one measured in the browser."
    v = par.get("versions") or {}
    tjs_v = v.get("@huggingface/transformers") or "4.x"
    checks = read_json(Path(a.checks)) if a.checks else {"repos": {}}
    chk = {**checks, "repo": checks["repos"].get(name, {})}
    lines = [
        "---", yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip(), "---", "",
        f"# {title}", "",
        f"**Alternative format.** The same specialist as [{sib.split('/')[-1]}]({sib_url}) in the Transformers.js layout "
        f"(`config.json` + `onnx/{GRAPH}`), for apps that already use transformers.js. The reference build is the ONNX Runtime "
        "GenAI one in that repo: this copy carries the same int4 weights byte for byte (" + (
            f"the token table `{meta['graph'].get('embeddingFile', 'embed_tokens.fp16.bin')}` of that repo is `onnx/{DATA}_1` here, read by one added "
            "Gather node; otherwise only graph metadata and file names changed" if meta["graph"].get("embeddingGather")
            else "graph changes listed under Limits" if graph_changes(meta["graph"])
            else "only graph metadata and file names changed") + "), the same heads and the same prompts, so accuracy is the one reported there.", "",
        f"**Parity with the GenAI build:** {par['items']} prompts written for this check (not a test set), Node CPU "
        f"(onnxruntime-node {v.get('onnxruntime-node', '?')}, transformers.js {tjs_v}): hidden state at every read position "
        f"max |diff| {par['hiddenMaxAbsDiff']:g}, head probabilities max |diff| {par['probMaxAbsDiff']:g}, decisions "
        f"{par['decisionsEqual']}/{par['items']} equal. {browser}", "",
        *_quick_start(name, mans, out, chk),
        *_plain_section(name, mans, read_json(out / _qs_manifest(mans)), chk, tjs_v),
        "## Limits", "",
        "- The decision heads are separate files (`heads/`), applied by s1s2; transformers.js alone does not read them.",
        _graph_limits(meta),
        _wasm_line(chk), "",
        "## License", "", lic, "",
        "## Credits", "", cred,
        "- Transformers.js layout and runtime: [transformers.js](https://github.com/huggingface/transformers.js) (Apache-2.0, Hugging Face).", "",
    ]
    text = "\n".join(lines)
    internal = [re.compile(a + b, re.I) for a, b in (("lab", "-s12"), ("study", " 0"), ("notice", " 0"))] + [re.compile(r"\b[Aa]rm [A-Z]\b")]
    bad = [r.pattern for r in internal if r.search(text)]  # patterns built from pieces so this file does not match them
    if bad:
        raise SystemExit(f"card contains internal wording {bad}; fix the source sections first")
    (out / "README.md").write_text(text)
    log(STAGE, f"card written ({len(text)} chars)")


def _git_blob_sha1(p):
    import hashlib
    b = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def verify_remote(api, name, out, revision=None):
    info = api.model_info(name, revision=revision, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    local = [p for p in out.rglob("*") if p.is_file() and not p.relative_to(out).as_posix().startswith(".")]
    bad = []
    for p in local:
        rel = p.relative_to(out).as_posix()
        s = remote.get(rel)
        if s is None:
            bad.append(f"{rel}: missing")
        elif s.lfs is not None:
            if s.lfs.sha256 != sha256_file(p) or s.size != p.stat().st_size:
                bad.append(f"{rel}: lfs sha256/size differ")
        elif s.blob_id != _git_blob_sha1(p):
            bad.append(f"{rel}: git blob differs")
    if bad:
        raise SystemExit("remote verification failed: " + "; ".join(bad))
    return info.sha, len(local)


def stage_publish(a):
    from huggingface_hub import HfApi
    api = HfApi()
    out = Path(a.out)
    meta = read_json(out / ".pack-tjs.json")
    name = meta["name"]
    mans = meta["manifests"]
    api.create_repo(name, repo_type="model", private=False, exist_ok=True)
    if a.gated:
        api.update_repo_settings(name, gated=a.gated)
    ignore = [".*", "manifest*.json", "README.md"]
    c1 = api.upload_folder(repo_id=name, folder_path=str(out), ignore_patterns=ignore,
                           commit_message="Transformers.js layout of the GenAI build (same weights): graph, data, config, tokenizer, heads")
    sha1 = c1.oid
    log(STAGE, f"files commit {sha1}")
    a.revision = sha1
    stage_pin(a)
    if not (out / "README.md").exists():
        raise SystemExit("run the card stage first")
    c2 = api.upload_folder(repo_id=name, folder_path=str(out), allow_patterns=mans + ["README.md"],
                           commit_message=f"Manifests pinned to {sha1[:8]} and model card")
    head, n = verify_remote(api, name, out)
    log(STAGE, f"published {name}@{head} ({n} files verified; manifests pin {sha1})")
    if a.collection:
        try:
            api.add_collection_item(a.collection, item_id=name, item_type="model", exists_ok=True)
            log(STAGE, f"added to collection {a.collection}")
        except Exception as e:  # noqa: BLE001
            log(STAGE, f"collection: {e}")
    if a.link_source:
        sib = meta["source"]["repo"]
        from huggingface_hub import hf_hub_download
        card = Path(hf_hub_download(sib, "README.md", force_download=True)).read_text()
        url = f"https://huggingface.co/{name}"
        if url not in card:
            line = f"Also in the Transformers.js layout (alternative format, same weights): [{name.split('/')[-1]}]({url})."
            fm, body = _front_matter(card)
            parts = body.split("\n\n")
            i = next((k for k, t in enumerate(parts) if t.startswith("PyTorch adapter")), None)
            if i is None:
                i = next(k for k, t in enumerate(parts) if t.strip() and not t.startswith(("#", "!")))
            parts.insert(i + 1, line)
            new = f"---\n{fm}\n---\n" + "\n\n".join(parts)
            api.upload_file(path_or_fileobj=new.encode(), path_in_repo="README.md", repo_id=sib,
                            commit_message="Link the Transformers.js layout")
            log(STAGE, f"linked from {sib}")


def stage_hydrate(a):
    """Rebuild the local inputs of `card`/`update` (.pack-tjs.json, parity summary, source README) from the published
    repo, when the pack folder is gone: manifests + README from the Hub, no weights."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or a.repo
    files = [s.rfilename for s in api.model_info(name).siblings]
    mans = sorted(f for f in files if re.fullmatch(r"manifest[^/]*\.json", f))
    for f in mans + ["README.md"]:
        shutil.copyfile(hf_hub_download(name, f, force_download=True), out / f)
    m0 = read_json(out / mans[0])
    b = m0["builds"][BUILD]
    prov = m0["provenance"]
    src = prov["source"]
    src_dir = out.parent / f".{out.name}.source"
    src_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hf_hub_download(src["repo"], "README.md", revision=src["revision"]), src_dir / "README.md")
    card = (out / "README.md").read_text()
    v = re.search(r"onnxruntime-node ([^,]+), transformers\.js ([^)]+)\)", card)
    br = re.search(r"decisions \d+/\d+ equal\. (.+)$", card, re.M)
    par = {k: prov["parity"][BUILD][k] for k in ("items", "decisionsEqual", "hiddenMaxAbsDiff", "probMaxAbsDiff", "measuredAt")}
    par["versions"] = {"onnxruntime-node": v.group(1), "@huggingface/transformers": v.group(2)} if v else {}
    write_json(out.parent / f".{out.name}.parity.json", par)
    embed = "data_1" in b["files"]
    write_json(out / ".pack-tjs.json", {"source": {"repo": src["repo"], "revision": src["revision"]}, "sourceDir": str(src_dir),
                                        "name": name, "graph": {"embeddingGather": embed or None}, "manifests": mans,
                                        "browser": br.group(1) if br else None, "hydratedAt": now_iso()})
    log(STAGE, f"hydrated {name}: manifests {mans}, devices {b['devices']}, browser sentence: {bool(br)}")


def stage_update(a):
    """Manifests + card only (e.g. after a browser run adds a device): pin to the same files commit, upload, verify."""
    from huggingface_hub import HfApi
    api = HfApi()
    out = Path(a.out)
    meta = read_json(out / ".pack-tjs.json")
    m0 = read_json(out / meta["manifests"][0])
    a.revision = m0["builds"][BUILD]["revision"]
    stage_pin(a)
    stage_card(a)
    api.upload_folder(repo_id=meta["name"], folder_path=str(out), allow_patterns=meta["manifests"] + ["README.md"],
                      commit_message="Manifests and card: " + (a.message or "update"))
    head, n = verify_remote(api, meta["name"], out)
    log(STAGE, f"updated {meta['name']}@{head} ({n} files verified)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m s1s2_convert.pack_tjs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["pack", "parity", "pin", "card", "publish", "update", "hydrate"])
    ap.add_argument("--out", required=True, help="output folder (the new repo's contents)")
    ap.add_argument("--repo", help="pack: source GenAI S1S2 repo id; hydrate: the published transformers.js repo id")
    ap.add_argument("--revision", help="pack: source revision (default: current main, recorded); pin: files commit sha")
    ap.add_argument("--name", help="pack: new repo id (default: <source> with -ONNX-transformersjs-S1S2)")
    ap.add_argument("--items", help="parity: JSONL rows {id, manifest, input, question}")
    ap.add_argument("--devices", help="pin: comma-separated devices the build was run on (default: cpu)")
    ap.add_argument("--lib", help="s1s2 checkout (default: the converter's parent directory)")
    ap.add_argument("--browser", help="card/pin/update: sentence on the browser run (card text and build fidelity notes; default: not run)")
    ap.add_argument("--message", help="update: commit message")
    ap.add_argument("--checks", help="card/update: JSON with the plain-snippet, pipeline(\"decision\") and WASM checks (examples/tjs_card_checks.json)")
    ap.add_argument("--gated", help="publish: gating mode for the new repo (e.g. auto), same as the source")
    ap.add_argument("--collection", help="publish: Hub collection slug to add the repo to")
    ap.add_argument("--link-source", action="store_true", help="publish: add one line to the source card linking this repo")
    ap.add_argument("--lm-head-max-mb", type=int, default=256,
                    help="pack: split lm_head so each part's fp32 unpack buffer on browser CPU (WASM) is at most this many MiB (0: no split)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    {"pack": stage_pack, "parity": stage_parity, "pin": stage_pin, "card": stage_card, "publish": stage_publish, "update": stage_update,
     "hydrate": stage_hydrate}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
