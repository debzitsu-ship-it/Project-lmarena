"""
Evaluation: quantitative metrics for a trained checkpoint.

Metrics
-------
loss        : average cross-entropy per token on held-out data (lower = better)
perplexity  : exp(loss). Intuition: "the model is as confused as if it were
              choosing uniformly among `ppl` tokens at each step."
              Random init on vocab V gives ppl = V; any value far below V
              means real learning happened.
tokens/sec  : inference/eval throughput on this machine.
params      : total learnable parameter count.

Run: .venv/bin/python -m my_ai.evaluate --checkpoint my_ai/checkpoints/best.pt \
        --data my_ai/data/processed --split test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from my_ai.data.dataset import TokenBatchStream, open_token_file
from my_ai.training.trainer import load_checkpoint, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="my_ai/checkpoints/best.pt")
    ap.add_argument("--data", default="my_ai/data/processed")
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = pick_device()
    model, ckpt = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    toks = open_token_file(os.path.join(args.data, f"{args.split}.bin"), meta["vocab_size"])
    stream = TokenBatchStream(toks, model.cfg.context_length, args.batch_size,
                              device=device, seed=123)

    losses, n_tokens = [], 0
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = stream.next_batch()
            _, loss = model(x, y)
            losses.append(loss.item())
            n_tokens += x.numel()
    dt = time.time() - t0

    avg = sum(losses) / len(losses)
    print(f"checkpoint      : {args.checkpoint} (step {ckpt.get('step')})")
    print(f"split           : {args.split} ({len(toks):,} tokens)")
    print(f"parameters      : {model.num_parameters():,}")
    print(f"loss            : {avg:.4f}")
    print(f"perplexity      : {math.exp(min(avg, 20)):.2f}  (random init would be ~{model.cfg.vocab_size})")
    print(f"eval throughput : {n_tokens / dt:,.0f} tokens/sec on {device}")
    if device == "cuda":
        print(f"gpu memory      : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
