"""Generation tests. Run: .venv/bin/pytest my_ai/tests/test_generate.py -q"""
import torch

from my_ai.inference.generate import generate, generate_text
from my_ai.model.transformer import ModelConfig, TransformerLM
from my_ai.tokenizer.tokenizer import CharTokenizer, EOS

CFG = ModelConfig(vocab_size=64, context_length=32, n_layers=2, n_heads=2,
                  d_model=32, d_ff=64, dropout=0.0)


def make_model():
    torch.manual_seed(0)
    return TransformerLM(CFG).eval()


def test_greedy_is_deterministic():
    m = make_model()
    a = generate(m, [5, 6, 7], max_new_tokens=10, temperature=0.0)
    b = generate(m, [5, 6, 7], max_new_tokens=10, temperature=0.0)
    assert a == b


def test_max_tokens_respected():
    m = make_model()
    out = generate(m, [1, 2], max_new_tokens=5, temperature=1.0, stop_tokens=set())
    assert len(out) <= 5


def test_top_k_restricts_choices():
    m = make_model()
    torch.manual_seed(1)
    out = generate(m, [3], max_new_tokens=20, temperature=1.0, top_k=1, stop_tokens=set())
    torch.manual_seed(1)
    greedy = generate(m, [3], max_new_tokens=20, temperature=0.0, stop_tokens=set())
    assert out == greedy                       # top_k=1 == greedy


def test_stop_token_halts():
    m = make_model()
    out = generate(m, [3], max_new_tokens=200, temperature=1.0,
                   stop_tokens=set(range(64)))  # every token is a stop token
    assert out == []


def test_generate_text_roundtrip():
    tok = CharTokenizer()
    tok.train("abcdefghij " * 10)
    m = TransformerLM(ModelConfig(vocab_size=tok.vocab_size, context_length=32,
                                  n_layers=1, n_heads=2, d_model=32, d_ff=64,
                                  dropout=0.0)).eval()
    text = generate_text(m, tok, "abc", max_new_tokens=10, temperature=1.0)
    assert text.startswith("abc")


def test_prompt_longer_than_context_is_cropped():
    m = make_model()
    long_prompt = list(range(1, 60)) * 2       # 118 ids > context 32
    long_prompt = [i % 64 for i in long_prompt]
    out = generate(m, long_prompt, max_new_tokens=3, temperature=0.0, stop_tokens=set())
    assert len(out) == 3                       # no crash, still generates
