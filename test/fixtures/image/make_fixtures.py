"""Fixtures for test/media-image.test.mjs, from the reference implementation (Hugging Face transformers 5.19, Pillow backend; CPU only, no model weights):
smart_resize cases, Pillow bicubic resize + normalisation + patch layout of Qwen2VLImageProcessorPil, the token ids of the Visual Jev prompt (Qwen3-VL tokenizer),
and M-RoPE position ids from Qwen3VLModel.get_rope_index on a tiny random config.
  python make_fixtures.py --tokenizer DIR_WITH_tokenizer.json   (the Qwen/Qwen3-VL-4B-Instruct snapshot)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import PIL

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
a = ap.parse_args()
out = {"pillow": PIL.__version__}

from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize, Qwen2VLImageProcessorPil

cases = [(37, 53), (50, 70), (480, 640), (1080, 1920), (31, 31), (17, 1000), (333, 2), (1024, 1024), (4000, 3000), (65, 97)]
out["smart_resize"] = [{"h": h, "w": w, "factor": 32, "min": mn, "max": mx, "out": list(smart_resize(h, w, 32, mn, mx))}
                       for (h, w) in cases for (mn, mx) in ((65536, 16777216), (1024, 4096), (65536, 786432)) if max(h, w) / min(h, w) <= 200]

# bit-exact resize + patches on small budgets
rng = np.random.default_rng(0)
ip = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[.5, .5, .5], image_std=[.5, .5, .5], size={"shortest_edge": 1024, "longest_edge": 4096})
imgs = []
for (h, w, mn, mx) in ((37, 53, 1024, 4096), (50, 70, 1024, 4096), (130, 90, 1024, 4096), (96, 64, 1024, 4096), (20, 24, 4096, 16384)):
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    arr[: h // 2, : w // 3] = (np.linspace(0, 255, h // 2 * (w // 3) * 3).reshape(h // 2, w // 3, 3)).astype(np.uint8)  # some structure, not only noise
    p = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[.5, .5, .5], image_std=[.5, .5, .5], size={"shortest_edge": mn, "longest_edge": mx})
    o = p(images=[Image.fromarray(arr)], return_tensors="np")
    rh, rw = smart_resize(h, w, 32, mn, mx)
    resized = np.asarray(Image.fromarray(arr).resize((rw, rh), Image.BICUBIC))
    imgs.append({"h": h, "w": w, "min": mn, "max": mx, "rgb": arr.reshape(-1).tolist(), "resized": [rh, rw], "resizedRgb": resized.reshape(-1).tolist(),
                 "grid": o["image_grid_thw"][0].tolist(), "pixelValues": o["pixel_values"].reshape(-1).tolist(), "shape": list(o["pixel_values"].shape)})
out["images"] = imgs

# prompt: the Visual Jev format (code/vdm/models/prompts.py of the paper repository), token ids from the Qwen3-VL tokenizer
from tokenizers import Tokenizer
tk = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
enc = lambda s: tk.encode(s, add_special_tokens=False).ids
SYSTEM = "You are a visual decision model. You inspect the image and answer each question by choosing exactly one of the given options. You never explain."
L = [chr(65 + i) for i in range(26)]
def prefix(ctx, n_pad):
    c = f"\n{ctx.strip()}" if ctx.strip() else ""
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * n_pad}<|vision_end|>{c}"
def suffix(instr, opts):
    lines = [f"\n\nQuestion: {instr.strip()}", "Options:"] + [f"{L[i]}. {o}" for i, o in enumerate(opts)] + ["Reply with the single letter of the correct option."]
    return "\n".join(lines) + "<|im_end|>\n<|im_start|>assistant\nAnswer:"
prompts = []
for ctx, n_pad, instr, opts in (("This image is one item in a visual inventory.", 6, "Is there a dog in the image?", ["yes", "no"]),
                                ("", 3, "What color is the bottle?", ["green", "clear", "red", "blue"]),
                                ("  spaced context  ", 4, "How many?  ", ["one", "two", "three", "none", "many"])):
    pre, suf = prefix(ctx, n_pad), suffix(instr, opts)
    # The published inference (prepare_group) tokenises prefix and suffix SEPARATELY and concatenates; training tokenised the joint text. They differ when a
    # token spans the boundary (a context ending in "." followed by "\n\n"): recorded as jointEqual, the reader follows the inference path.
    a_, s_, f_ = enc(pre), enc(suf), enc(pre + suf)
    prompts.append({"context": ctx, "nPad": n_pad, "instruction": instr, "options": opts, "prefixText": pre, "suffixText": suf, "prefixIds": a_, "suffixIds": s_, "jointEqual": f_ == a_ + s_})
out["prompts"] = prompts
out["optionTokenIds"] = [enc(" " + c) for c in L[:16]]
out["specialIds"] = {t: enc(t) for t in ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>")}

# M-RoPE from the reference: the full sequence (prefix + suffix) of each prompt with an image of grid (1, gh, gw) whose merged token count is nPad
import torch
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
cfg = Qwen3VLConfig(text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=152000),
                    vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32, deepstack_visual_indexes=[0], spatial_merge_size=2, num_position_embeddings=16))
model = Qwen3VLModel(cfg)
IMG = out["specialIds"]["<|image_pad|>"][0]
pos_cases = []
for gh, gw, p in ((4, 6, prompts[0]), (2, 6, prompts[1]), (6, 4, prompts[2]), (8, 8, prompts[0])):
    n_pad = (gh // 2) * (gw // 2)
    pre = prefix(p["context"], n_pad)
    ids = enc(pre) + enc(suffix(p["instruction"], p["options"]))
    t = torch.tensor([ids])
    mm = (t == IMG).to(torch.int32)
    pos, _ = model.get_rope_index(t, mm_token_type_ids=mm, image_grid_thw=torch.tensor([[1, gh, gw]]))
    pos_cases.append({"grid": [1, gh, gw], "ids": ids, "imageTokenId": IMG, "positions": pos[:, 0].tolist(), "prefixLen": len(enc(pre))})
out["rope"] = pos_cases
# position embedding resampled to the grid (Qwen3VLVisionModel: bilinear, align_corners True, merge-block order)
from transformers.vision_utils import get_vision_interpolation_indices_and_weights
pe = []
for side, hid, gh, gw in ((6, 8, 4, 6), (6, 8, 2, 2), (48, 4, 16, 12), (5, 3, 6, 4)):
    tab = torch.from_numpy(rng.standard_normal((side * side, hid)).astype(np.float32))
    idx, w = get_vision_interpolation_indices_and_weights(torch.tensor([[1, gh, gw]]), num_grid_per_side=side, mode="bilinear", align_corners=True, spatial_merge_size=2)
    o2 = (tab[idx] * w[:, :, None]).sum(1)
    pe.append({"side": side, "hidden": hid, "grid": [1, gh, gw], "table": tab.reshape(-1).tolist(), "out": o2.reshape(-1).tolist()})
out["posembed"] = pe
Path(__file__).with_name("image.json").write_text(json.dumps(out))
print("wrote", len(json.dumps(out)) // 1024, "KiB")
