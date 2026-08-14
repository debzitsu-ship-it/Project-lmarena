"""
Stage-5 entry point: raw files -> tokenizer + packed token binaries.

Run:
  .venv/bin/python -m my_ai.prepare_data --input my_ai/data/raw \
      --out my_ai/data/processed --vocab-size 2048

Produces:
  <out>/tokenizer.json  trained BPE tokenizer
  <out>/train.bin / val.bin / test.bin   packed uint16 token files (memmap-ready)
  <out>/meta.json       bookkeeping (vocab size, token counts)
"""
from __future__ import annotations

import argparse
import json
import os

from my_ai.data.dataset import load_corpus, split_corpus, tokenize_and_pack
from my_ai.tokenizer.tokenizer import BPETokenizer, N_SPECIAL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="files or directories")
    ap.add_argument("--out", default="my_ai/data/processed")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--tokenizer-sample-chars", type=int, default=2_000_000,
                    help="train BPE on at most this much text (speed)")
    args = ap.parse_args()

    print("loading corpus ...")
    docs = load_corpus(args.input)
    n_chars = sum(len(d) for d in docs)
    print(f"  {len(docs)} documents, {n_chars:,} characters")
    if not docs:
        raise SystemExit("no usable documents found -- check --input paths")

    train_docs, val_docs, test_docs = split_corpus(docs, args.val_frac, args.test_frac)
    print(f"  split: {len(train_docs)} train / {len(val_docs)} val / {len(test_docs)} test docs")

    print(f"training BPE tokenizer (vocab {args.vocab_size}) ...")
    assert args.vocab_size >= N_SPECIAL + 256
    sample = "\n".join(train_docs)[: args.tokenizer_sample_chars]
    # Strip literal special-token markers: they are encoded as reserved IDs
    # at packing time, so BPE must not waste merges on their byte spellings.
    from my_ai.tokenizer.tokenizer import SPECIAL_TOKENS
    for marker in SPECIAL_TOKENS:
        sample = sample.replace(marker, " ")
    tok = BPETokenizer()
    tok.train(sample, vocab_size=args.vocab_size, verbose=True)
    os.makedirs(args.out, exist_ok=True)
    tok.save(os.path.join(args.out, "tokenizer.json"))
    print(f"  saved tokenizer, actual vocab {tok.vocab_size}")

    meta = {"vocab_size": tok.vocab_size}
    for name, subset in [("train", train_docs), ("val", val_docs), ("test", test_docs)]:
        if not subset:
            continue
        arr = tokenize_and_pack(subset, tok, os.path.join(args.out, f"{name}.bin"))
        meta[f"{name}_tokens"] = int(len(arr))
        print(f"  {name}.bin: {len(arr):,} tokens")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
