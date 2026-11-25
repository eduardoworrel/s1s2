"""asr-decoder: the text decoder of Qwen3-ASR-0.6B as an ONNX Runtime GenAI build that reads embeddings.

Qwen3-ASR's thinker is a Qwen3 causal LM (28 layers, hidden 1024, vocabulary 151936) whose prompt holds 104 audio-feature rows in
place of `<|audio_pad|>` tokens. For audio-only input the three rotary positions of the original are equal, so a plain Qwen3 does
the same computation. This command:

  1. renames `thinker.model.*` to `model.*`, drops the audio tower, checks that `lm_head` is byte-identical to the embedding (tied),
     and writes a plain Qwen3ForCausalLM checkpoint (bf16) with a normal config (no multimodal rotary section);
  2. builds it with the ONNX Runtime GenAI 0.19.0 builder (same fork fixes as `build`) with `exclude_embeds=true`: the graph takes
     `inputs_embeds`, the caller supplies the rows (token rows from the table, audio rows from the tower);
  3. writes `embed_tokens.fp16.bin` (float16 little-endian [vocab, hidden]) and `decoder_tokens.json` (byte-level BPE pieces and the
     ids of the special tokens, for decoding generated ids without a tokenizer library).

  s1s2-convert asr-decoder --model Qwen/Qwen3-ASR-0.6B --revision <40-hex> --work W --out OUT [--precision int4 --ep webgpu]

Output OUT/: decoder/{model.onnx,model.onnx.data,genai_config.json}, embed_tokens.fp16.bin, decoder_tokens.json, decoder.json (record with sha256).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from .util import log, write_json, sha256_file

PREFIX = "thinker.model."
HEAD = "thinker.lm_head.weight"
EMBED = PREFIX + "embed_tokens.weight"
TOWER = "thinker.audio_tower."


def plain_checkpoint(src: Path, dst: Path, expect_sha256=None):
    """Write the plain Qwen3 checkpoint of the thinker's text model into `dst`. -> record."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    st = src / "model.safetensors"
    h = sha256_file(st)
    if expect_sha256 and h != expect_sha256:
        raise SystemExit(f"{st}: sha256 {h} != expected {expect_sha256}")
    cfg = json.loads((src / "config.json").read_text())
    tc = dict(cfg["thinker_config"]["text_config"])
    if tc.get("model_type") != "qwen3":
        raise SystemExit(f"text_config.model_type is {tc.get('model_type')!r}, expected 'qwen3'")
    rs = tc.get("rope_scaling") or {}
    if rs.get("mrope_section") is None:
        log("asr-decoder", "note: text_config has no mrope section")
    tensors = {}
    with safe_open(str(st), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            if k.startswith(TOWER) or k == HEAD:
                continue
            if not k.startswith(PREFIX):
                raise SystemExit(f"unexpected tensor {k}")
            tensors["model." + k[len(PREFIX):]] = f.get_tensor(k).contiguous()
        head, emb = f.get_tensor(HEAD), f.get_tensor(EMBED)
    tied = bool(torch.equal(head, emb))
    if not tied:
        raise SystemExit("lm_head differs from the embedding: this converter writes a tied head only")
    drop = ("_name_or_path", "architectures", "rope_scaling", "id2label", "label2id", "tokenizer_class")
    clean = {k: v for k, v in tc.items() if v is not None and k not in drop and k not in
             {"add_cross_attention", "bad_words_ids", "tf_legacy_loss", "torchscript", "use_bfloat16", "is_decoder", "is_encoder_decoder",
              "chunk_size_feed_forward", "cross_attention_hidden_size", "diversity_penalty", "encoder_no_repeat_ngram_size", "exponential_decay_length_penalty",
              "length_penalty", "max_length", "min_length", "no_repeat_ngram_size", "num_beam_groups", "num_beams", "num_return_sequences", "output_attentions",
              "output_hidden_states", "output_scores", "problem_type", "prefix", "pruned_heads", "remove_invalid_values", "repetition_penalty", "return_dict",
              "return_dict_in_generate", "temperature", "tie_encoder_decoder", "top_k", "top_p", "typical_p", "do_sample", "early_stopping", "finetuning_task",
              "task_specific_params"}}
    clean.update({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "tie_word_embeddings": True, "torch_dtype": "bfloat16", "rope_scaling": None})
    dst.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})
    write_json(dst / "config.json", clean)
    for f in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    return {"source_sha256": h, "tensors": len(tensors), "tied_head": tied, "embedding": emb, "config": clean}


def byte_level_pieces(src: Path, vocab: int):
    """{tokens: [piece per id], added: [ids whose piece is literal text], special: [ids skipped when decoding]}."""
    vocab_json = json.loads((src / "vocab.json").read_text())
    tokens = [""] * vocab
    for piece, i in vocab_json.items():
        tokens[i] = piece
    tc = json.loads((src / "tokenizer_config.json").read_text())
    added, special = [], []
    for i, d in tc.get("added_tokens_decoder", {}).items():
        i = int(i)
        tokens[i] = d["content"]
        added.append(i)
        if d.get("special"):
            special.append(i)
    return {"tokens": tokens, "added": sorted(added), "special": sorted(special)}


# The 30 languages of the Qwen3-ASR decoder's built-in language identification (code -> the name the model writes after "language").
LANGUAGES = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
             "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
             "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino",
             "fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian"}


def prompt_spec(src: Path):
    """Token ids of the fixed prompt around the audio rows, of "language", and of the 30 names written after it."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(src))
    ct = json.loads((src / "chat_template.json").read_text())["chat_template"]
    text = tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}],
                                   chat_template=ct, add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text, add_special_tokens=False)
    pad = tok.convert_tokens_to_ids("<|audio_pad|>")
    if ids.count(pad) != 1:
        raise SystemExit(f"the prompt holds {ids.count(pad)} <|audio_pad|> tokens, expected 1")
    k = ids.index(pad)
    base = tok.encode(text + "language", add_special_tokens=False)
    if base[: len(ids)] != ids:
        raise SystemExit('the prompt followed by "language" does not keep the prompt tokens')
    langs = {}
    for code, name in LANGUAGES.items():
        full = tok.encode(text + "language " + name, add_special_tokens=False)
        if full[: len(base)] != base:
            raise SystemExit(f"{name}: unstable token prefix")
        langs[code] = {"name": name, "ids": full[len(base):]}
    gen = json.loads((src / "generation_config.json").read_text())
    eos = gen["eos_token_id"] if isinstance(gen["eos_token_id"], list) else [gen["eos_token_id"]]
    return {"before": ids[:k], "after": ids[k + 1:], "audioPad": pad, "languageToken": base[len(ids):][0] if len(base) == len(ids) + 1 else base[-1],
            "languages": langs, "eos": eos, "asrTextToken": tok.convert_tokens_to_ids("<asr_text>")}


def write_fp16_table(emb, path: Path):
    import torch
    f16 = emb.to(torch.float32).to(torch.float16)
    back = f16.to(torch.float32)
    err = float((back - emb.to(torch.float32)).abs().max())
    if not torch.isfinite(f16).all():
        raise SystemExit("embedding table does not fit float16")
    with open(path, "wb") as fh:
        fh.write(f16.contiguous().view(torch.int16).numpy().astype("<i2").tobytes())
    return {"max_abs_error_bf16_to_fp16": err, "rows": int(emb.shape[0]), "hidden": int(emb.shape[1])}


def main(argv):
    ap = argparse.ArgumentParser(prog="s1s2-convert asr-decoder", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local directory of the checkpoint (Qwen/Qwen3-ASR-0.6B)")
    ap.add_argument("--revision", required=True, help="40-hex revision the directory was downloaded at (recorded)")
    ap.add_argument("--repo", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--expect-sha256", help="sha256 of model.safetensors")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="int4", choices=["int4", "fp16", "fp32"])
    ap.add_argument("--ep", default="webgpu", choices=["webgpu", "cpu"])
    ap.add_argument("--builder-src", default=None)
    a = ap.parse_args(argv)
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
        raise SystemExit("--revision: a 40-hex commit")
    from .build import build
    W, OUT, SRC = Path(a.work), Path(a.out), Path(a.model)
    plain = W / "qwen3-decoder"
    log("asr-decoder", "plain Qwen3 checkpoint")
    rec = plain_checkpoint(SRC, plain, a.expect_sha256)
    OUT.mkdir(parents=True, exist_ok=True)
    emb = rec.pop("embedding")
    vocab = int(rec["config"]["vocab_size"])
    log("asr-decoder", "embedding table (float16) and tokens")
    table = write_fp16_table(emb, OUT / "embed_tokens.fp16.bin")
    toks = byte_level_pieces(SRC, vocab)
    (OUT / "decoder_tokens.json").write_text(json.dumps(toks, ensure_ascii=False, separators=(",", ":")))
    prompt = prompt_spec(SRC)
    write_json(OUT / "decoder_prompt.json", prompt)
    log("asr-decoder", f"prompt: {len(prompt['before'])} + audio + {len(prompt['after'])} ids; languages: {sum(len(v['ids']) == 1 for v in prompt['languages'].values())} of one token")
    log("asr-decoder", f"{a.precision} {a.ep} build (exclude_embeds)")
    prov = build(plain, OUT / "decoder", W, precision=a.precision, ep=a.ep, builder_src=a.builder_src, extra=["exclude_embeds=true"], hidden=False)
    files = {}
    for rel in ("decoder/model.onnx", "decoder/model.onnx.data", "decoder/genai_config.json", "embed_tokens.fp16.bin", "decoder_tokens.json", "decoder_prompt.json"):
        p = OUT / rel
        files[rel] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    write_json(OUT / "decoder.json", {
        "schema": "s1s2.asr-decoder/1", "base": {"repo": a.repo, "revision": a.revision, "model_safetensors_sha256": rec["source_sha256"]},
        "tied_head": rec["tied_head"], "text_config": rec["config"], "embedding_table": table, "build": prov, "files": files,
        "tokens": {"count": vocab, "added": len(toks["added"]), "special": len(toks["special"])},
    })
    log("asr-decoder", "done: " + ", ".join(f"{k} {v['size']}" for k, v in files.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
