"""
Stage-6 entry point: train the model on prepared data.

Run (after prepare_data):
  .venv/bin/python -m my_ai.train --config my_ai/configs/tiny_5m.json \
      --data my_ai/data/processed --steps 2000

Use --auto to let the hardware inspector choose the config.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from my_ai.data.dataset import TokenBatchStream, open_token_file
from my_ai.model.transformer import ModelConfig, TransformerLM
from my_ai.tokenizer.tokenizer import load_tokenizer
from my_ai.training.trainer import TrainConfig, load_checkpoint, pick_device, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to model config JSON")
    ap.add_argument("--auto", action="store_true", help="pick config from hardware")
    ap.add_argument("--data", default="my_ai/data/processed")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--checkpoint-dir", default="my_ai/checkpoints")
    ap.add_argument("--sample-prompt", default="The ")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    if args.auto or args.config is None:
        from my_ai.hardware import inspect_hardware, recommend_config
        name, why = recommend_config(inspect_hardware())
        args.config = f"my_ai/configs/{name}.json"
        print(f"auto-selected {args.config}: {why}")

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    tokenizer = load_tokenizer(os.path.join(args.data, "tokenizer.json"))

    cfg = ModelConfig.from_json(args.config)
    cfg.vocab_size = meta["vocab_size"]  # match the actual tokenizer

    train_toks = open_token_file(os.path.join(args.data, "train.bin"), cfg.vocab_size)
    val_path = os.path.join(args.data, "val.bin")
    val_toks = open_token_file(val_path, cfg.vocab_size) if os.path.exists(val_path) else None
    print(f"tokens: train {len(train_toks):,}" +
          (f" / val {len(val_toks):,}" if val_toks is not None else " (no val set)"))

    batch_size = args.batch_size or (16 if device == "cpu" else 32)
    tc = TrainConfig(batch_size=batch_size, grad_accum_steps=args.grad_accum,
                     max_steps=args.steps, lr=args.lr,
                     warmup_steps=max(20, args.steps // 20),
                     eval_interval=max(100, args.steps // 10),
                     sample_interval=max(200, args.steps // 5),
                     checkpoint_dir=args.checkpoint_dir)

    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device=device)
        print(f"resumed from {args.resume} at step {ckpt.get('step')}")
    else:
        model = TransformerLM(cfg)
    print(f"parameters: {model.num_parameters():,} "
          f"({model.num_parameters(non_embedding=True):,} non-embedding)")

    train_stream = TokenBatchStream(train_toks, cfg.context_length, tc.batch_size, device=device)
    val_stream = (TokenBatchStream(val_toks, cfg.context_length, tc.batch_size, device=device, seed=99)
                  if val_toks is not None and len(val_toks) > cfg.context_length + 1 else None)

    metrics = train(model, train_stream, val_stream, tc, device,
                    tokenizer=tokenizer, sample_prompt=args.sample_prompt)

    # keep the tokenizer next to the checkpoints so chat "just works"
    shutil.copy(os.path.join(args.data, "tokenizer.json"),
                os.path.join(tc.checkpoint_dir, "tokenizer.json"))

    print("\n=== final metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
