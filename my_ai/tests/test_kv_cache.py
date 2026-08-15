"""KV-cache correctness: cached generation must match full-reforward exactly.
Run: .venv/bin/pytest my_ai/tests/test_kv_cache.py -q
"""
import torch

from my_ai.inference.generate import generate
from my_ai.model.transformer import ModelConfig, TransformerLM

CFG = ModelConfig(vocab_size=100, context_length=64, n_layers=2, n_heads=2,
                  d_model=32, d_ff=64, dropout=0.0)


def make_model():
    torch.manual_seed(0)
    return TransformerLM(CFG).eval()


def test_cached_logits_match_full_forward():
    """Incremental forward_with_cache must equal the full forward, step by step."""
    m = make_model()
    ids = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        # full forward logits at the last position, growing one token at a time
        logits_cache, caches = m.forward_with_cache(ids[:, :5])
        for t in range(5, 10):
            full_logits, _ = m(ids[:, : t])
            assert torch.allclose(logits_cache, full_logits[0, -1], atol=1e-5), f"mismatch at t={t}"
            logits_cache, caches = m.forward_with_cache(ids[:, t : t + 1], caches)
        full_logits, _ = m(ids)
        assert torch.allclose(logits_cache, full_logits[0, -1], atol=1e-5)


def test_greedy_generation_identical_with_and_without_cache():
    m = make_model()
    prompt = [5, 6, 7, 8]
    a = generate(m, list(prompt), max_new_tokens=20, temperature=0.0,
                 stop_tokens=set(), use_cache=True)
    b = generate(m, list(prompt), max_new_tokens=20, temperature=0.0,
                 stop_tokens=set(), use_cache=False)
    assert a == b and len(a) == 20


def test_cache_overflow_falls_back():
    """Prompt+max_new > context must still work (falls back to sliding window)."""
    m = make_model()
    prompt = [i % 100 for i in range(50)]
    out = generate(m, prompt, max_new_tokens=30, temperature=0.0,
                   stop_tokens=set(), use_cache=True)  # 50+30 > 64 -> fallback
    assert len(out) == 30


def test_cache_respects_causality():
    """Cached path must not attend beyond context or corrupt positions."""
    m = make_model()
    with torch.no_grad():
        ids = torch.randint(0, 100, (1, 30))
        lc, caches = m.forward_with_cache(ids)
        lf, _ = m(ids)
        assert torch.allclose(lc, lf[0, -1], atol=1e-5)
