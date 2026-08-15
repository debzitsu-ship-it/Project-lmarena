"""
Stage-B: chat fine-tuning (SFT) on top of a pretrained checkpoint.

Loads the pretrained model, continues training on the chat-heavy
finetune.bin at LOW learning rate for a SHORT run, and saves to a separate
checkpoint lineage (chat_best.pt / chat_latest.pt) so the pretrained base
is never overwritten.

Run:
  python -m my_ai.finetune --base my_ai/checkpoints/latest.pt \
      --data my_ai/data/processed --steps 1200 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import os

from my_ai.data.dataset import TokenBatchStream, open_token_file
from my_ai.tokenizer.tokenizer import load_tokenizer
from my_ai.training.trainer import (TrainConfig, load_checkpoint, pick_device,
                                    train)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="my_ai/checkpoints/latest.pt")
    ap.add_argument("--data", default="my_ai/data/processed")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="my_ai/checkpoints/chat")
    args = ap.parse_args()

    device = pick_device()
    model, ckpt = load_checkpoint(args.base, device=device)
    print(f"base: {args.base} (step {ckpt.get('step')}), "
          f"{model.num_parameters():,} params, device {device}")

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    tokenizer = load_tokenizer(os.path.join(args.data, "tokenizer.json"))

    ft_tokens = open_token_file(os.path.join(args.data, "finetune.bin"),
                                meta["vocab_size"])
    val_path = os.path.join(args.data, "val.bin")
    val_tokens = open_token_file(val_path, meta["vocab_size"]) \
        if os.path.exists(val_path) else None

    tc = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.steps,
        lr=args.lr,
        min_lr=args.lr / 10,
        warmup_steps=max(20, args.steps // 20),
        eval_interval=max(100, args.steps // 6),
        sample_interval=max(200, args.steps // 4),
        checkpoint_dir=args.out_dir,
        log_interval=50,
    )
    train_stream = TokenBatchStream(ft_tokens, model.cfg.context_length,
                                    tc.batch_size, device=device, seed=11)
    val_stream = TokenBatchStream(val_tokens, model.cfg.context_length,
                                  tc.batch_size, device=device, seed=99) \
        if val_tokens is not None else None

    metrics = train(model, train_stream, val_stream, tc, device,
                    tokenizer=tokenizer,
                    sample_prompt="<|user|>Tell me about python.<eos><|assistant|>")

    import shutil
    shutil.copy(os.path.join(args.data, "tokenizer.json"),
                os.path.join(args.out_dir, "tokenizer.json"))
    print("\n=== fine-tune metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
