"""
Stage-B data prep: build a chat-heavy fine-tuning mix.

Takes the SAME tokenizer as pretraining (a model is married to its
tokenizer) and packs a mix of ~70% chat samples + ~30% book/code text into
finetune.bin. Fine-tuning on this mix at a low learning rate teaches
assistant behavior without erasing the language learned in pretraining.

Run:
  python -m my_ai.prepare_finetune \
      --data my_ai/data/processed --raw my_ai/data/raw --chat-frac 0.7
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from my_ai.data.dataset import clean_text, load_corpus, tokenize_and_pack
from my_ai.tokenizer.tokenizer import load_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="my_ai/data/processed",
                    help="dir with tokenizer.json from pretraining")
    ap.add_argument("--raw", default="my_ai/data/raw")
    ap.add_argument("--chat-file", default="chat_synthetic.jsonl")
    ap.add_argument("--chat-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    tok = load_tokenizer(os.path.join(args.data, "tokenizer.json"))
    rng = random.Random(args.seed)

    # chat docs: every *_synthetic.jsonl in raw (chat, math, knowledge, ...)
    chat_docs: list[str] = []
    chat_files = [f for f in os.listdir(args.raw) if f.endswith("_synthetic.jsonl")]
    if args.chat_file not in chat_files and os.path.exists(os.path.join(args.raw, args.chat_file)):
        chat_files.append(args.chat_file)
    for cf in chat_files:
        with open(os.path.join(args.raw, cf), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chat_docs.append(json.loads(line)["text"])
                    except (json.JSONDecodeError, KeyError):
                        continue

    # user corrections ("learn from mistakes"): repeated 20x so the few
    # hand-taught fixes actually leave a mark in the weights
    corr_path = os.path.join("my_ai", "chat", "memory_store", "corrections.json")
    if os.path.exists(corr_path):
        try:
            with open(corr_path, encoding="utf-8") as f:
                corrections = json.load(f)
            for c in corrections:
                chat_docs.extend(
                    [f"<|user|>{c['q']}<eos><|assistant|>{c['a']}<eos>"] * 20)
            print(f"included {len(corrections)} user corrections (x20 weight)")
        except (json.JSONDecodeError, KeyError):
            pass

    # background text docs (books + code), chopped into chat-sized pieces
    bg_docs_full = [d for d in load_corpus([args.raw])
                    if not d.startswith("<|user|>")]
    bg_pieces: list[str] = []
    for d in bg_docs_full:
        for i in range(0, len(d), 2000):
            piece = clean_text(d[i:i + 2000])
            if len(piece) > 200:
                bg_pieces.append(piece)
    rng.shuffle(bg_pieces)

    # target mix by document count
    n_chat = len(chat_docs)
    n_bg = int(n_chat * (1 - args.chat_frac) / max(args.chat_frac, 1e-9))
    docs = chat_docs + bg_pieces[:n_bg]
    rng.shuffle(docs)

    out_path = os.path.join(args.data, "finetune.bin")
    arr = tokenize_and_pack(docs, tok, out_path)
    print(f"finetune.bin: {len(arr):,} tokens "
          f"({n_chat} chat docs + {min(n_bg, len(bg_pieces))} text pieces, "
          f"target chat fraction {args.chat_frac})")

    meta_path = os.path.join(args.data, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["finetune_tokens"] = int(len(arr))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
