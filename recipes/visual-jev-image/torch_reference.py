"""Oracle of the image reader: PyTorch float32, the adapter applied (PEFT), one independent forward per question over the whole sequence (image re-encoded each time),
option logits read at the colon of `Answer:`. NOT RUN on the Mac (a 4B in float32 needs ~18 GB); written for the RunPod parity step (README.md, step 6).

  python torch_reference.py --base Qwen/Qwen3-VL-4B-Instruct --adapter guanxuyu/visual-jev-4b-answer-sft --questions q.jsonl --out ref.jsonl [--min-pixels 65536 --max-pixels 786432]
q.jsonl: one object per line {"image": path, "context": str, "instruction": str, "options": [str, ...], "label": int | null}
Uses the standard transformers / peft APIs and the prompt of spec/MULTIMODAL.md section 10.4 (the same text as src/media/imagePrep.ts); it does not run the paper's repository code.
"""
import argparse, json
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

SYSTEM = "You are a visual decision model. You inspect the image and answer each question by choosing exactly one of the given options. You never explain."
L = [chr(65 + i) for i in range(26)]


def prompt(ctx, instr, opts):
    c = f"\n{ctx.strip()}" if ctx and ctx.strip() else ""
    pre = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{c}"
    suf = "\n".join([f"\n\nQuestion: {instr.strip()}", "Options:", *[f"{L[i]}. {o}" for i, o in enumerate(opts)], "Reply with the single letter of the correct option."])
    return pre + suf + "<|im_end|>\n<|im_start|>assistant\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct"); ap.add_argument("--adapter", default="guanxuyu/visual-jev-4b-answer-sft")
    ap.add_argument("--questions", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-pixels", type=int, default=65536); ap.add_argument("--max-pixels", type=int, default=786432)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--no-adapter", action="store_true")
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.base)
    proc.image_processor.size = {"shortest_edge": a.min_pixels, "longest_edge": a.max_pixels}
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.base, dtype=torch.float32).to(a.device)
    if not a.no_adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    ids = [proc.tokenizer(" " + c, add_special_tokens=False)["input_ids"] for c in L[:16]]
    assert all(len(i) == 1 for i in ids)
    opt_ids = torch.tensor([i[0] for i in ids], device=a.device)
    with open(a.questions) as f, open(a.out, "w") as out, torch.inference_mode():
        for line in f:
            q = json.loads(line)
            with Image.open(q["image"]) as im:
                enc = proc(text=[prompt(q.get("context", ""), q["instruction"], q["options"])], images=[im.convert("RGB")], return_tensors="pt").to(a.device)
            logits = model(**enc).logits[0, -1].float()[opt_ids][: len(q["options"])]
            out.write(json.dumps({**{k: q[k] for k in ("image", "instruction", "options", "label") if k in q}, "grid": enc["image_grid_thw"][0].tolist(),
                                  "logits": logits.cpu().tolist(), "probabilities": torch.softmax(logits, -1).cpu().tolist()}) + "\n")


if __name__ == "__main__":
    main()
