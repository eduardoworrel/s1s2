"""Torch references for the decoder questions (language, term) of the Marimba audio package.

Runs the `marimba-audio` 0.1.2 engine (float32, CPU) on left-aligned 8 s windows (what its decoder lane feeds the decoder:
`decoder_snapshot`) and writes, per window: the 104 x 1024 `audio_features`, the 30 language log-probability sums, the five
hypotheses (greedy + 4 beams) and the term fractions of a list of probe terms. Resumable: one JSONL line per window.

Third-party code: run it in an isolated sandbox with pinned versions (torch 2.8.0, transformers 4.57.6, marimba-audio 0.1.2).

  python decoder_refs.py --wav DIR --clips clips.json --out DIR [--only 0:10] [--joined 12]
  clips.json: [{"fn": "<file in --wav>", "tr": "<reference transcript>"}, ...]; extra wavs (demo clips) go in --extra-wav.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--wav", required=True)
p.add_argument("--clips", required=True)
p.add_argument("--extra-wav", default=None, help="directory of further wav files, windowed the same way")
p.add_argument("--out", required=True)
p.add_argument("--only", default=None, help="a:b slice of the window list")
p.add_argument("--joined", type=int, default=12, help="how many full 8 s windows (3 clips joined) to add")
p.add_argument("--threads", type=int, default=6)
a = p.parse_args()
torch.set_num_threads(a.threads)

from marimba import Marimba, Question
from marimba.audio import decode_audio, right_aligned_window
from marimba.language import LANGUAGES
from marimba.lexical import exact_fraction, normalize
from marimba.windows import WindowSnapshot

os.makedirs(os.path.join(a.out, "af"), exist_ok=True)
clips = json.load(open(a.clips))
W = 128000


def left_window(x):
    """Audio at index 0, zeros after; longer than 8 s keeps the last 8 s (right_aligned_window, then decoder_snapshot)."""
    win, mask = right_aligned_window(x)
    first = int(np.argmax(mask))
    out = np.zeros_like(win); m = np.zeros_like(mask)
    out[: W - first] = win[first:]; m[: W - first] = mask[first:]
    return out, m


items = []
for i, c in enumerate(clips):
    items.append({"id": f"c{i:02d}", "kind": "clip", "fn": c["fn"], "tr": c["tr"], "path": os.path.join(a.wav, c["fn"])})
if a.extra_wav:
    for f in sorted(os.listdir(a.extra_wav)):
        if f.endswith(".wav"):
            items.append({"id": "d-" + f[:-4], "kind": "demo", "fn": f, "tr": "", "path": os.path.join(a.extra_wav, f)})
auds = {it["id"]: decode_audio(it["path"]) for it in items}
cl = [it for it in items if it["kind"] == "clip"]
for j in range(min(a.joined, len(cl))):
    k = j * max(1, len(cl) // max(a.joined, 1))
    ids = [cl[(k + t) % len(cl)]["id"] for t in range(3)]
    cat = np.concatenate([auds[i] for i in ids])
    jid = f"j{j:02d}"
    auds[jid] = cat
    items.append({"id": jid, "kind": "joined", "fn": "+".join(ids), "tr": " ".join(cl[(k + t) % len(cl)]["tr"] for t in range(3)), "path": None})
if a.only:
    lo, hi = (int(v) for v in a.only.split(":")); items = items[lo:hi]

done = set()
rp = os.path.join(a.out, "dec_ref.jsonl")
if os.path.exists(rp):
    done = {json.loads(l)["id"] for l in open(rp)}

m = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")
eng = m._engine
codes = sorted(LANGUAGES)
words_all = sorted({w for it in items for w in normalize(it["tr"]).split() if len(w) >= 4})
t_start = time.time()
for it in items:
    if it["id"] in done:
        continue
    win, mask = left_window(auds[it["id"]])
    t0 = time.time()
    enc = eng.encode(win)
    af = np.asarray(enc["audio_features"], dtype=np.float32)
    t1 = time.time()
    scores = eng.language_scores(af, codes)
    t2 = time.time()
    hyps = eng.hypotheses(af)
    t3 = time.time()
    own = [w for w in normalize(it["tr"]).split() if len(w) >= 4]
    rng = np.random.default_rng(int.from_bytes(it["id"].encode(), "little") % (2**32))
    others = [w for w in words_all if w not in own]
    probe = sorted(set(own[:6])) + [str(w) for w in rng.choice(others, size=min(6, len(others)), replace=False)] + ["marimba"]
    # hypotheses text contributes its own words as positives: the guaranteed-present words of the greedy text
    probe += [w for w in normalize(hyps[0]).split() if len(w) >= 4][:3]
    probe = sorted(set(probe))
    terms = {t: exact_fraction(t, hyps) for t in probe}
    row = {"id": it["id"], "kind": it["kind"], "fn": it["fn"], "tr": it["tr"], "n": int(mask.sum()), "frames": int(af.shape[0]),
           "language": scores, "hyps": hyps, "terms": terms, "ms": {"encode": (t1 - t0) * 1000, "language": (t2 - t1) * 1000, "hyps": (t3 - t2) * 1000}}
    af.tofile(os.path.join(a.out, "af", it["id"] + ".f32"))
    np.asarray(win, dtype=np.float32).tofile(os.path.join(a.out, "af", it["id"] + ".win.f32"))
    with open(rp, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(it["id"], f"{time.time() - t_start:.0f}s", max(scores, key=scores.get), repr(hyps[0][:60]), flush=True)

# prompt ids and name suffixes (once)
ip = os.path.join(a.out, "prompt_ids.json")
if not os.path.exists(ip):
    prompt = eng.asr._build_text_prompt(context="", force_language=None)
    af0 = np.zeros((104, 1024), dtype=np.float32)
    ids = eng._inputs(prompt, af0)["input_ids"][0].tolist()
    tok = eng.processor.tokenizer
    base = prompt + "language"
    base_ids = tok.encode(base, add_special_tokens=False)
    names = {}
    for c in codes:
        full = tok.encode(base + " " + LANGUAGES[c], add_special_tokens=False)
        assert full[: len(base_ids)] == base_ids
        names[c] = {"name": LANGUAGES[c], "ids": full[len(base_ids):]}
    json.dump({"prompt": prompt, "ids": ids, "base_ids": base_ids, "names": names, "audio_pad": 151676}, open(ip, "w"), ensure_ascii=False)
    print("prompt ids", len(ids), "base", len(base_ids))

# the public API on a few windows (it takes the prefill path): same numbers expected
cp = os.path.join(a.out, "public_check.json")
if not os.path.exists(cp) and items:
    res = []
    for it in [x for x in items if x["kind"] == "clip"][:3]:
        win, mask = left_window(auds[it["id"]])
        ref = next(json.loads(l) for l in open(rp) if json.loads(l)["id"] == it["id"])
        word = next(iter(ref["terms"]))
        qs = [Question.choice("What language?", codes, task="language"), Question.noul(f'Did someone say "{word}"?')]
        dec = m.decide_snapshot(WindowSnapshot(win, mask, 0.0, 8.0), qs)
        res.append({"id": it["id"], "term": word, "language": {k: float(v) for k, v in (dec[0].probabilities or {}).items()}, "term_p": dec[1].probability, "ref_term": ref["terms"][word]})
    json.dump(res, open(cp, "w"))
    print("public check written")
