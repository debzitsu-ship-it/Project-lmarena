"""
Training system.

What actually happens each step (the honest version):
1. Sample a batch (x, y) where y is x shifted one token left.
2. Forward pass -> logits -> cross-entropy loss = "how surprised was the
   model by the true next token, averaged over every position".
3. loss.backward() computes, via the chain rule, the gradient of the loss
   with respect to EVERY parameter: "if I nudged this weight up a tiny bit,
   would the loss go up or down, and how fast?"
4. AdamW steps each parameter a small distance against its gradient
   (with per-parameter adaptive step sizes + decoupled weight decay).
That's all "learning" is: millions of tiny nudges that make the training
text incrementally less surprising. Nothing more mystical than that.

Features: warmup+cosine LR schedule, gradient clipping, gradient
accumulation, mixed precision (auto-enabled on CUDA), checkpoint save/load,
best-checkpoint tracking by validation loss, tokens/sec + memory metrics,
periodic sample generation.
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import torch

from my_ai.model.transformer import ModelConfig, TransformerLM


@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 2000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_batches: int = 20
    sample_interval: int = 500
    checkpoint_dir: str = "my_ai/checkpoints"
    seed: int = 1337
    log_interval: int = 50


def pick_device() -> str:
    if torch.cuda.is_available():
        # Smoke-test the GPU: some environments (e.g. Kaggle P100 with a
        # newer PyTorch build) report CUDA available but have no compiled
        # kernels for the card ("no kernel image"). Fall back to CPU then.
        try:
            (torch.zeros(2, device="cuda") + 1).sum().item()
            return "cuda"
        except Exception as e:
            print(f"[warn] CUDA present but unusable ({type(e).__name__}); "
                  f"falling back to CPU. Tip: on Kaggle choose 'GPU T4 x2' "
                  f"instead of P100.")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, tc: TrainConfig) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if step >= tc.max_steps:
        return tc.min_lr
    frac = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * frac))


def make_optimizer(model: TransformerLM, tc: TrainConfig) -> torch.optim.AdamW:
    # weight-decay 2D matrices (real "weights"); don't decay biases/norms/embeddings' gains
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tc.lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluate(model: TransformerLM, stream, n_batches: int, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = stream.next_batch()
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path: str, model: TransformerLM, optimizer, step: int,
                    best_val: float, tc: TrainConfig) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "model_config": model.cfg.to_dict(),
        "train_config": asdict(tc),
        "step": step,
        "best_val": best_val,
    }, path)


def load_checkpoint(path: str, device: str = "cpu",
                    optimizer=None) -> tuple[TransformerLM, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TransformerLM(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return model, ckpt


def train(model: TransformerLM, train_stream, val_stream, tc: TrainConfig,
          device: str, tokenizer=None, sample_prompt: str = "The",
          on_log=None) -> dict:
    """Run the full training loop. Returns final metrics dict."""
    torch.manual_seed(tc.seed)
    model.to(device).train()
    optimizer = make_optimizer(model, tc)

    use_amp = device == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16
                             if torch.cuda.is_bf16_supported() else torch.float16) \
        if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not torch.cuda.is_bf16_supported())

    best_val = float("inf")
    tokens_seen = 0
    t0 = time.time()
    history = []

    for step in range(tc.max_steps):
        lr = lr_at(step, tc)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(tc.grad_accum_steps):
            x, y = train_stream.next_batch()
            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / tc.grad_accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()
            tokens_seen += x.numel()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % tc.log_interval == 0 or step == tc.max_steps - 1:
            dt = time.time() - t0
            tps = tokens_seen / max(dt, 1e-9)
            mem = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
            msg = (f"step {step:5d} | loss {loss_accum:.4f} | ppl {math.exp(min(loss_accum, 20)):9.2f} "
                   f"| lr {lr:.2e} | {tps:,.0f} tok/s" + (f" | gpu {mem:.2f}GB" if mem else ""))
            print(msg)
            history.append({"step": step, "train_loss": loss_accum, "lr": lr, "tokens_per_sec": tps})
            if on_log:
                on_log(history[-1])

        if val_stream is not None and step > 0 and step % tc.eval_interval == 0:
            val_loss = evaluate(model, val_stream, tc.eval_batches, device)
            print(f"  eval @ {step}: val loss {val_loss:.4f} | val ppl {math.exp(min(val_loss, 20)):.2f}")
            save_checkpoint(os.path.join(tc.checkpoint_dir, "latest.pt"),
                            model, optimizer, step, best_val, tc)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(tc.checkpoint_dir, "best.pt"),
                                model, optimizer, step, best_val, tc)
                print(f"  new best -> saved {tc.checkpoint_dir}/best.pt")

        if tokenizer is not None and step > 0 and step % tc.sample_interval == 0:
            from my_ai.inference.generate import generate_text
            sample = generate_text(model, tokenizer, sample_prompt, max_new_tokens=80,
                                   temperature=0.8, device=device)
            print(f"  sample: {sample[:200]!r}")

    # final eval + save
    final_val = evaluate(model, val_stream, tc.eval_batches, device) if val_stream else float("nan")
    save_checkpoint(os.path.join(tc.checkpoint_dir, "latest.pt"),
                    model, optimizer, tc.max_steps, best_val, tc)
    if val_stream and final_val < best_val:
        best_val = final_val
        save_checkpoint(os.path.join(tc.checkpoint_dir, "best.pt"),
                        model, optimizer, tc.max_steps, best_val, tc)
    dt = time.time() - t0
    return {
        "final_train_loss": loss_accum,
        "final_val_loss": final_val,
        "final_val_perplexity": math.exp(min(final_val, 20)) if final_val == final_val else None,
        "best_val_loss": best_val if best_val != float("inf") else None,
        "tokens_seen": tokens_seen,
        "tokens_per_sec": tokens_seen / dt,
        "wall_time_sec": dt,
        "parameters": model.num_parameters(),
    }
