"""The image reader on the built ONNX graphs, in Python (ONNX Runtime): the same pipeline the browser runtime implements (spec/MULTIMODAL.md section 10.5), used to
measure parity against torch_reference.py on the same questions and the same pixels. NOT RUN yet: it needs the built graphs (README.md steps 1 to 5).

  python ort_reader.py --dir BUILD --questions q.jsonl --out ort.jsonl [--min-pixels 65536 --max-pixels 786432] [--provider CPUExecutionProvider]
BUILD holds vision.onnx (+ data), text.onnx (+ data pieces), embedding.fp16 (float16 [vocab, hidden]; the cuda build's embedding.onnx.data is exactly this), tokenizer.json.
Pixels come from the Hugging Face Pillow processor here (the JavaScript pre-processing is tested against it separately).
"""
import argparse, json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

SYSTEM = "You are a visual decision model. You inspect the image and answer each question by choosing exactly one of the given options. You never explain."
L = [chr(65 + i) for i in range(26)]
IMG, VS, VE = 151655, 151652, 151653
HID, LAYERS, KV, HD = 2560, 36, 8, 128


def prefix_text(ctx, n):
    c = f"\n{ctx.strip()}" if ctx and ctx.strip() else ""
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * n}<|vision_end|>{c}"


def suffix_text(instr, opts):
    return "\n".join([f"\n\nQuestion: {instr.strip()}", "Options:", *[f"{L[i]}. {o}" for i, o in enumerate(opts)], "Reply with the single letter of the correct option."]) + "<|im_end|>\n<|im_start|>assistant\nAnswer:"


def mrope(ids, grid, merge=2):
    """3 x n positions of one text + image + text sequence (Qwen3VLModel.get_rope_index)."""
    n = len(ids); pos = np.zeros((3, n), np.int64); gh, gw = grid[1] // merge, grid[2] // merge
    cur, i = 0, 0
    while i < n:
        j = i; img = ids[i] == IMG
        while j < n and (ids[j] == IMG) == img: j += 1
        if not img:
            pos[:, i:j] = np.arange(cur, cur + j - i); cur += j - i
        else:
            k = i
            for y in range(gh):
                for x in range(gw):
                    pos[:, k] = (cur, cur + y, cur + x); k += 1
            cur += max(gh, gw)
        i = j
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--questions", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-pixels", type=int, default=65536); ap.add_argument("--max-pixels", type=int, default=786432)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    a = ap.parse_args(); d = Path(a.dir)
    tk = Tokenizer.from_file(str(d / "tokenizer.json")); enc = lambda s: tk.encode(s, add_special_tokens=False).ids
    table = np.memmap(d / "embedding.fp16", dtype=np.float16, mode="r").reshape(-1, HID)
    ip = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[.5] * 3, image_std=[.5] * 3, size={"shortest_edge": a.min_pixels, "longest_edge": a.max_pixels})
    vis = ort.InferenceSession(str(d / "vision.onnx"), providers=[a.provider]); txt = ort.InferenceSession(str(d / "text.onnx"), providers=[a.provider])
    vin = {i.name: i for i in vis.get_inputs()}; tin = {i.name: i for i in txt.get_inputs()}
    f_dt = np.float16 if "float16" in tin["inputs_embeds"].type else np.float32
    opt_rows = np.stack([table[enc(" " + c)[0]].astype(np.float64) for c in L[:16]])
    ds_names = [n for n in tin if n.startswith("deepstack_")]
    with open(a.questions) as f, open(a.out, "w") as out:
        for line in f:
            q = json.loads(line)
            with Image.open(q["image"]) as im:
                o = ip(images=[im.convert("RGB")], return_tensors="np")
            px = o["pixel_values"].astype(np.float16 if "float16" in vin["pixel_values"].type else np.float32)
            grid = o["image_grid_thw"][0].tolist()
            vo = dict(zip([x.name for x in vis.get_outputs()], vis.run(None, {"pixel_values": px, "image_grid_thw": o["image_grid_thw"]})))
            n_img = vo["image_features"].shape[0]
            pre = enc(prefix_text(q.get("context", ""), n_img)); ids0 = np.array(pre)
            emb = table[ids0].astype(np.float32); m = ids0 == IMG; emb[m] = vo["image_features"].astype(np.float32)
            pos = mrope(pre, grid)
            feeds = {"inputs_embeds": emb[None].astype(f_dt), "attention_mask": np.ones((1, len(pre)), np.int64), "position_ids": pos[:, None, :]}
            for k, nme in enumerate(ds_names):
                z = np.zeros((1, len(pre), HID), np.float32); z[0, m] = vo[f"deepstack_{k}"].astype(np.float32); feeds[nme] = z.astype(f_dt)
            for l in range(LAYERS):
                feeds[f"past_key_values.{l}.key"] = feeds[f"past_key_values.{l}.value"] = np.zeros((1, KV, 0, HD), f_dt)
            res = dict(zip([x.name for x in txt.get_outputs()], txt.run(None, feeds)))
            past = {f"past_key_values.{l}.{kv}": res[f"present.{l}.{kv}"] for l in range(LAYERS) for kv in ("key", "value")}
            suf = enc(suffix_text(q["instruction"], q["options"]))
            e2 = table[np.array(suf)].astype(np.float32)[None].astype(f_dt); base = int(pos.max()) + 1
            p2 = np.broadcast_to(np.arange(base, base + len(suf), dtype=np.int64), (3, 1, len(suf))).copy()
            f2 = {"inputs_embeds": e2, "attention_mask": np.ones((1, len(pre) + len(suf)), np.int64), "position_ids": p2, **past}
            for nme in ds_names: f2[nme] = np.zeros((1, len(suf), HID), f_dt)
            h = dict(zip([x.name for x in txt.get_outputs()], txt.run(None, f2)))["hidden"][0, -1].astype(np.float64)
            lg = opt_rows[: len(q["options"])] @ h; pr = np.exp(lg - lg.max()); pr /= pr.sum()
            out.write(json.dumps({**{k: q[k] for k in ("image", "instruction", "options", "label") if k in q}, "grid": grid, "logits": lg.tolist(), "probabilities": pr.tolist()}) + "\n")


if __name__ == "__main__":
    main()
