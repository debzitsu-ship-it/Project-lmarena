"""Model tests: forward pass shape, causal masking, loss, gradients, init.
Run: .venv/bin/pytest my_ai/tests/test_model.py -q
"""
import math

import torch
import pytest

from my_ai.model.transformer import ModelConfig, TransformerLM

CFG = ModelConfig(vocab_size=100, context_length=32, n_layers=2, n_heads=2,
                  d_model=32, d_ff=64, dropout=0.0)


def make_model():
    torch.manual_seed(0)
    return TransformerLM(CFG)


def test_forward_shapes():
    m = make_model()
    x = torch.randint(0, 100, (3, 16))
    logits, loss = m(x)
    assert logits.shape == (3, 16, 100)
    assert loss is None


def test_loss_is_scalar_and_near_uniform_at_init():
    m = make_model()
    x = torch.randint(0, 100, (4, 16))
    y = torch.randint(0, 100, (4, 16))
    _, loss = m(x, y)
    assert loss.dim() == 0
    # randomly initialized model ~ uniform distribution => loss ~ ln(vocab)
    assert abs(loss.item() - math.log(100)) < 1.0


def test_causal_masking():
    """Changing a FUTURE token must not change logits at earlier positions."""
    m = make_model().eval()
    x1 = torch.randint(0, 100, (1, 16))
    x2 = x1.clone()
    x2[0, 10:] = (x2[0, 10:] + 1) % 100  # perturb only positions >= 10
    with torch.no_grad():
        l1, _ = m(x1)
        l2, _ = m(x2)
    assert torch.allclose(l1[0, :10], l2[0, :10], atol=1e-5)   # past unchanged
    assert not torch.allclose(l1[0, 10:], l2[0, 10:], atol=1e-3)  # future changed


def test_backward_produces_gradients():
    m = make_model()
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))
    _, loss = m(x, y)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_context_length_enforced():
    m = make_model()
    with pytest.raises(AssertionError):
        m(torch.randint(0, 100, (1, 33)))  # 33 > context 32


def test_weight_tying():
    m = make_model()
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


def test_param_count_reasonable():
    m = make_model()
    n = m.num_parameters()
    assert 10_000 < n < 200_000  # tiny test config
