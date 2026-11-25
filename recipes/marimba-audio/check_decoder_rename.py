"""Does the plain Qwen3 checkpoint written by `s1s2-convert asr-decoder` compute what the Qwen3-ASR thinker computes?

Feeds the same inputs_embeds (prompt rows + torch `audio_features`) to the thinker (marimba-audio's vendored model, float32) and to a plain
Qwen3ForCausalLM loaded from the converter's checkpoint, and reports the largest logit difference. For audio-only input the three
rotary positions of the original are equal, so this is the check that the plain rotary embedding is the same computation.
Third-party code: run in the isolated sandbox.

  python check_decoder_rename.py --qwen <Qwen3-ASR dir> --plain <converter work dir>/qwen3-decoder --refs <dec_ref dir> --out check.json [--n 3]
"""
import argparse, json
from types import SimpleNamespace
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--qwen", required=True); ap.add_argument("--plain", required=True); ap.add_argument("--refs", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=3)
a = ap.parse_args()
torch.set_num_threads(4)
from marimba._qwen import FrozenASR
from transformers import AutoModelForCausalLM

asr = FrozenASR(a.qwen, SimpleNamespace(dtype="float32", device="cpu"))
thinker = asr.model.thinker.eval()
plain = AutoModelForCausalLM.from_pretrained(a.plain, torch_dtype=torch.float32).eval()
rows = [json.loads(l) for l in open(f"{a.refs}/dec_ref.jsonl")][: a.n]
prompt = asr._build_text_prompt(context="", force_language=None) + "language"
res = []
for r in rows:
    af = torch.from_numpy(np.fromfile(f"{a.refs}/af/{r['id']}.f32", dtype=np.float32).reshape(-1, 1024))
    exp = asr.processor.replace_multimodal_special_tokens([prompt], iter([len(af)]))
    batch = asr.processor.tokenizer(exp, return_tensors="pt", padding=True)
    emb = thinker.get_input_embeddings()(batch["input_ids"])
    emb = emb.masked_scatter(thinker.get_placeholder_mask(batch["input_ids"], inputs_embeds=emb), af.to(emb.dtype))
    with torch.inference_mode():
        lt = thinker(inputs_embeds=emb, attention_mask=batch["attention_mask"], use_cache=False).logits
        lp = plain(inputs_embeds=emb, use_cache=False).logits
    d = (lt - lp).abs()
    res.append({"id": r["id"], "positions": int(lt.shape[1]), "max_abs_logit_diff": float(d.max()), "last_row_max_abs_diff": float(d[0, -1].max()),
                "argmax_equal_all_positions": bool((lt.argmax(-1) == lp.argmax(-1)).all()), "logit_abs_max": float(lt.abs().max())})
    print(res[-1], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
