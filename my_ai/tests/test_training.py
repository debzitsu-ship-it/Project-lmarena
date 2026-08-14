"""Training-system tests: loss decreases (overfit check), checkpoint
save/load round-trip, LR schedule. Run: .venv/bin/pytest my_ai/tests/test_training.py -q
"""
import math

import numpy as np
import torch

from my_ai.data.dataset import TokenBatchStream, tokenize_and_pack
from my_ai.model.transformer import ModelConfig, TransformerLM
from my_ai.tokenizer.tokenizer import CharTokenizer
from my_ai.training.trainer import (TrainConfig, load_checkpoint, lr_at,
                                    save_checkpoint, train)

TEXT = "the quick brown fox jumps over the lazy dog. " * 60


def _setup(tmp_path):
    tok = CharTokenizer()
    tok.train(TEXT)
    arr = tokenize_and_pack([TEXT], tok, str(tmp_path / "toks.bin"))
    cfg = ModelConfig(vocab_size=tok.vocab_size, context_length=64,
                      n_layers=2, n_heads=2, d_model=64, d_ff=128, dropout=0.0)
    torch.manual_seed(0)
    model = TransformerLM(cfg)
    return tok, arr, model


def test_overfit_tiny_sample(tmp_path):
    """Stage-4 gate: a tiny model must overfit a tiny corpus (loss plunges)."""
    tok, arr, model = _setup(tmp_path)
    stream = TokenBatchStream(arr, 64, 8, seed=0)
    tc = TrainConfig(batch_size=8, max_steps=150, lr=3e-3, warmup_steps=10,
                     eval_interval=10_000, sample_interval=10_000,
                     log_interval=10_000, checkpoint_dir=str(tmp_path / "ck"))
    metrics = train(model, stream, None, tc, device="cpu")
    start_loss = math.log(tok.vocab_size)          # ~uniform at init
    assert metrics["final_train_loss"] < start_loss * 0.35, \
        f"loss {metrics['final_train_loss']:.3f} did not drop enough from ~{start_loss:.3f}"


def test_checkpoint_roundtrip(tmp_path):
    _, _, model = _setup(tmp_path)
    tc = TrainConfig(checkpoint_dir=str(tmp_path))
    p = str(tmp_path / "ck.pt")
    save_checkpoint(p, model, None, step=42, best_val=1.23, tc=tc)
    model2, ckpt = load_checkpoint(p)
    assert ckpt["step"] == 42 and abs(ckpt["best_val"] - 1.23) < 1e-9
    x = torch.randint(0, model.cfg.vocab_size, (1, 16))
    model.eval(); model2.eval()
    with torch.no_grad():
        l1, _ = model(x)
        l2, _ = model2(x)
    assert torch.allclose(l1, l2, atol=1e-6)       # identical weights


def test_lr_schedule():
    tc = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000)
    assert lr_at(0, tc) < lr_at(50, tc) < lr_at(99, tc)          # warmup rises
    assert abs(lr_at(100, tc) - 1e-3) < 1e-5                     # peak after warmup
    assert lr_at(500, tc) < 1e-3                                 # decaying
    assert abs(lr_at(1000, tc) - 1e-4) < 1e-9                    # floor at min_lr
