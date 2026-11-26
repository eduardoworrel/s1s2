"""Oracle of the image reader: PyTorch float32, the adapter applied (PEFT), one independent forward per question over the whole sequence (image re-encoded each time),
option logits read at the colon of `Answer:`. NOT RUN on the Mac (a 4B in float32 needs ~18 GB); written for the RunPod parity step (README.md, step 6).

  python torch_reference.py --base Qwen/Qwen3-VL-4B-Instruct --adapter guanxuyu/visual-jev-4b-answer-sft --questions q.jsonl --out ref.jsonl [--min-pixels 65536 --max-pixels 786432]
q.jsonl: one object per line {"image": path, "context": str, "instruction": str, "options": [str, ...], "label": int | null}
Uses the standard transformers / peft APIs, the prompt of spec/MULTIMODAL.md section 10.4 (the same text and the same separate tokenisation as src/media/imagePrep.ts and ort_reader.py) and the Pillow
image processor (the same pixels as ort_reader.py); it does not run the paper's repository code.
"""
import argparse, json
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

SYSTEM = "You are a visual decision model. You inspect the image and answer each question by choosing exactly one of the given options. You never explain."
L = [chr(65 + i) for i in range(26)]


def texts(ctx, instr, opts, n_pad):
    """Prefix and suffix text of spec section 10.4. They are tokenised SEPARATELY and concatenated, as the published inference (prepare_group) and the reader do:
    training tokenised the joint text, and the two differ when a context ends in "." (the reader follows the inference path)."""
    c = f"\n{ctx.strip()}" if ctx and ctx.strip() else ""
    pre = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * n_pad}<|vision_end|>{c}"
    suf = "\n".join([f"\n\nQuestion: {instr.strip()}", "Options:", *[f"{L[i]}. {o}" for i, o in enumerate(opts)], "Reply with the single letter of the correct option."])
    return pre, suf + "<|im_end|>\n<|im_start|>assistant\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct"); ap.add_argument("--adapter", default="guanxuyu/visual-jev-4b-answer-sft")
    ap.add_argument("--questions", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-pixels", type=int, default=65536); ap.add_argument("--max-pixels", type=int, default=786432)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no-adapter", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    # The Pillow image processor, explicitly: the same pixels as ort_reader.py (AutoProcessor would pick the torchvision one when torchvision is installed; the gap between the two
    # is a number to report, not to mix into this comparison).
    ip = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[.5] * 3, image_std=[.5] * 3, size={"shortest_edge": a.min_pixels, "longest_edge": a.max_pixels})
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.base, dtype=getattr(torch, a.dtype)).to(a.device)
    if not a.no_adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    ids = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in L[:16]]
    assert all(len(i) == 1 for i in ids)
    opt_ids = torch.tensor([i[0] for i in ids], device=a.device)
    with open(a.questions) as f, open(a.out, "w") as out, torch.inference_mode():
        for n, line in enumerate(f):
            if a.limit and n >= a.limit:
                break
            q = json.loads(line)
            with Image.open(q["image"]) as im:
                px = ip(images=[im.convert("RGB")], return_tensors="pt")
            grid = px["image_grid_thw"]
            pre, suf = texts(q.get("context", ""), q["instruction"], q["options"], int(grid[0, 1] * grid[0, 2]) // 4)
            input_ids = torch.tensor([tok(pre, add_special_tokens=False)["input_ids"] + tok(suf, add_special_tokens=False)["input_ids"]], device=a.device)
            logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), pixel_values=px["pixel_values"].to(a.device, getattr(torch, a.dtype)), image_grid_thw=grid.to(a.device),
                           mm_token_type_ids=(input_ids == model.config.image_token_id).to(torch.int32), logits_to_keep=1).logits[0, -1].float()[opt_ids][: len(q["options"])]
            out.write(json.dumps({**{k: q[k] for k in ("image", "instruction", "options", "label") if k in q}, "grid": grid[0].tolist(),
                                  "logits": logits.cpu().tolist(), "probabilities": torch.softmax(logits, -1).cpu().tolist()}) + "\n")


if __name__ == "__main__":
    main()
