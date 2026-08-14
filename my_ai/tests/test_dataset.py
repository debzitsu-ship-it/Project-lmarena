"""Dataset pipeline tests. Run: .venv/bin/pytest my_ai/tests/test_dataset.py -q"""
import json

import numpy as np

from my_ai.data.dataset import (
    TokenBatchStream, clean_text, load_corpus, load_file, split_corpus,
    tokenize_and_pack, open_token_file,
)
from my_ai.tokenizer.tokenizer import CharTokenizer, EOS


def test_load_txt_json_jsonl(tmp_path):
    (tmp_path / "a.txt").write_text("plain text document " * 5)
    (tmp_path / "b.json").write_text(json.dumps([{"text": "json doc one " * 5}, "json doc two " * 5]))
    with open(tmp_path / "c.jsonl", "w") as f:
        f.write(json.dumps({"text": "jsonl doc " * 5}) + "\n")
        f.write("{{{{ corrupted line\n")                    # must be skipped
        f.write(json.dumps("bare string doc " * 5) + "\n")
    docs = load_corpus([str(tmp_path)])
    assert len(docs) == 5  # corrupted line dropped, everything else kept


def test_corrupted_file_does_not_crash(tmp_path):
    (tmp_path / "bad.json").write_text("this is not json at all")
    assert load_file(str(tmp_path / "bad.json")) == []


def test_clean_text():
    dirty = "hello\x00\x07  world\n\n\n\n\nnext   line\t\tend"
    cleaned = clean_text(dirty)
    assert "\x00" not in cleaned and "\x07" not in cleaned
    for line in cleaned.split("\n"):
        assert "  " not in line and "\t" not in line   # whitespace runs collapsed
    assert "\n\n\n" not in cleaned                      # blank-line runs collapsed


def test_split_deterministic():
    docs = [f"doc {i} " * 10 for i in range(100)]
    a = split_corpus(docs, seed=1)
    b = split_corpus(docs, seed=1)
    assert a == b
    train, val, test = a
    assert len(train) + len(val) + len(test) == 100
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test)


def test_pack_and_stream(tmp_path):
    docs = ["the cat sat on the mat. " * 20, "dogs run fast in the park. " * 20]
    tok = CharTokenizer()
    tok.train("".join(docs))
    path = str(tmp_path / "train.bin")
    arr = tokenize_and_pack(docs, tok, path)
    assert (arr == EOS).sum() == 2                     # one <eos> per doc
    arr2 = open_token_file(path, tok.vocab_size)
    assert np.array_equal(arr, arr2)                   # reload matches

    stream = TokenBatchStream(arr2, context_length=16, batch_size=4)
    x, y = stream.next_batch()
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert (x[:, 1:] == y[:, :-1]).all()               # y is x shifted by one
