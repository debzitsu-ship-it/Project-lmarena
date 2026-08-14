"""Tokenizer correctness tests. Run: .venv/bin/pytest my_ai/tests/test_tokenizer.py -q"""
import pytest

from my_ai.tokenizer.tokenizer import (
    BPETokenizer, CharTokenizer, BOS, EOS, UNK, N_SPECIAL,
    USER_TOK, ASSISTANT_TOK, encode_with_specials, load_tokenizer,
)

SAMPLE = "hello world! the quick brown fox jumps over the lazy dog. " * 20


def test_char_roundtrip():
    tok = CharTokenizer()
    tok.train(SAMPLE)
    ids = tok.encode("hello world!")
    assert tok.decode(ids) == "hello world!"


def test_char_unknown_maps_to_unk():
    tok = CharTokenizer()
    tok.train("abc")
    ids = tok.encode("aZ")  # 'Z' unseen
    assert ids[1] == UNK


def test_char_bos_eos():
    tok = CharTokenizer()
    tok.train(SAMPLE)
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == BOS and ids[-1] == EOS
    assert tok.decode(ids) == "hi"  # specials skipped on decode


def test_bpe_roundtrip_any_text():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=N_SPECIAL + 256 + 50)
    for text in ["hello world!", "ZzZ unseen glyphs \u00e9\u00e8\u00ea", "emoji \U0001f600 ok", "  spaces  "]:
        assert tok.decode(tok.encode(text)) == text  # bytes => no <unk> ever


def test_bpe_learns_compression():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=N_SPECIAL + 256 + 100)
    n_bytes = len("the quick brown fox".encode("utf-8"))
    n_tokens = len(tok.encode("the quick brown fox"))
    assert n_tokens < n_bytes  # merges must compress in-domain text


def test_bpe_save_load(tmp_path):
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=N_SPECIAL + 256 + 30)
    p = str(tmp_path / "tok.json")
    tok.save(p)
    tok2 = load_tokenizer(p)
    text = "the lazy dog"
    assert tok2.encode(text) == tok.encode(text)
    assert tok2.decode(tok2.encode(text)) == text


def test_encode_with_specials():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=N_SPECIAL + 256 + 30)
    ids = encode_with_specials(tok, "<|user|>hello world<eos><|assistant|>hi")
    assert ids[0] == USER_TOK
    assert EOS in ids and ASSISTANT_TOK in ids
    assert tok.decode(ids) == "hello worldhi"  # specials dropped in plain decode


def test_bpe_vocab_size_accounting():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=N_SPECIAL + 256 + 10)
    assert tok.vocab_size == N_SPECIAL + 256 + len(tok.merges)
    assert len(tok.merges) <= 10
