"""
Dataset pipeline: load -> clean -> split -> tokenize -> pack -> stream.

Supported inputs
----------------
.txt   : raw text.
.json  : either a list of strings, a list of {"text": ...} objects,
         or a single object with a "text" field.
.jsonl : one JSON object (or string) per line.

Pipeline
--------
1. load_corpus()       reads every supported file under a directory/file list,
                       skipping corrupted files/records instead of crashing.
2. clean_text()        normalizes unicode, strips control chars, collapses
                       whitespace runs.
3. split_corpus()      deterministic train/val/test split by document.
4. tokenize_and_pack() encodes everything, concatenates with <eos> separators,
                       and memory-maps the result to disk as uint16/uint32 --
                       this is how we "stream" big data: the token file lives
                       on disk and np.memmap pages in only what a batch needs.
5. TokenBatchStream    yields random (x, y) next-token training pairs:
                       x = tokens[i : i+T],  y = tokens[i+1 : i+T+1]
                       (y is x shifted left by one -- at every position the
                       target is simply "the next token").
"""
from __future__ import annotations

import json
import os
import random
import unicodedata

import numpy as np
import torch


# ---------------------------------------------------------------- loading
def _texts_from_json_obj(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        t = obj.get("text")
        return [t] if isinstance(t, str) else []
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_texts_from_json_obj(item))
        return out
    return []


def load_file(path: str) -> list[str]:
    """Return list of document strings from one file; [] if unreadable."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".txt":
            with open(path, encoding="utf-8", errors="replace") as f:
                return [f.read()]
        if ext == ".json":
            with open(path, encoding="utf-8", errors="replace") as f:
                return _texts_from_json_obj(json.load(f))
        if ext == ".jsonl":
            docs = []
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        docs.extend(_texts_from_json_obj(json.loads(line)))
                    except json.JSONDecodeError:
                        continue  # skip corrupted line, keep the rest
            return docs
    except (OSError, json.JSONDecodeError, UnicodeError):
        return []
    return []


def load_corpus(paths: list[str], min_doc_chars: int = 32) -> list[str]:
    """Load all .txt/.json/.jsonl under the given files/directories."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files.extend(os.path.join(root, n) for n in sorted(names)
                             if os.path.splitext(n)[1].lower() in (".txt", ".json", ".jsonl"))
        else:
            files.append(p)
    docs = []
    for f in files:
        for doc in load_file(f):
            doc = clean_text(doc)
            if len(doc) >= min_doc_chars:  # drop empty/corrupted fragments
                docs.append(doc)
    return docs


# ---------------------------------------------------------------- cleaning
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # drop control characters except newline/tab
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    # collapse 3+ newlines to 2, and runs of spaces/tabs to one space
    out_lines = []
    blank = 0
    for line in text.split("\n"):
        line = " ".join(line.split())
        if line:
            blank = 0
            out_lines.append(line)
        else:
            blank += 1
            if blank <= 1:
                out_lines.append("")
    return "\n".join(out_lines).strip()


# ---------------------------------------------------------------- splitting
def split_corpus(docs: list[str], val_frac: float = 0.05, test_frac: float = 0.05,
                 seed: int = 1337) -> tuple[list[str], list[str], list[str]]:
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(docs) * val_frac)) if len(docs) > 2 else 0
    n_test = max(1, int(len(docs) * test_frac)) if len(docs) > 2 else 0
    val = [docs[i] for i in idx[:n_val]]
    test = [docs[i] for i in idx[n_val:n_val + n_test]]
    train = [docs[i] for i in idx[n_val + n_test:]]
    return train, val, test


# ---------------------------------------------------------------- packing
def tokenize_and_pack(docs: list[str], tokenizer, out_path: str) -> np.ndarray:
    """Encode docs, join with <eos>, write a binary token file, return memmap.

    Literal special-token markers in the text (e.g. "<|user|>") are encoded
    as their reserved IDs so chat-formatted corpora train the role tokens.
    """
    from my_ai.tokenizer.tokenizer import EOS, encode_with_specials
    ids: list[int] = []
    for d in docs:
        ids.extend(encode_with_specials(tokenizer, d))
        ids.append(EOS)  # document boundary marker
    dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32
    arr = np.array(ids, dtype=dtype)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr.tofile(out_path)
    return np.memmap(out_path, dtype=dtype, mode="r")


def open_token_file(path: str, vocab_size: int, in_ram_below: int = 200_000_000) -> np.ndarray:
    """Open a packed token file.

    Files smaller than `in_ram_below` bytes are loaded fully into RAM
    (much faster on slow disks); larger ones stay memory-mapped so huge
    corpora still stream from disk without exhausting memory.
    """
    dtype = np.uint16 if vocab_size < 65536 else np.uint32
    if os.path.getsize(path) <= in_ram_below:
        return np.fromfile(path, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r")


# ---------------------------------------------------------------- streaming
class TokenBatchStream:
    """
    Streams random next-token training batches from a (possibly huge)
    memory-mapped token array without loading it into RAM.
    """

    def __init__(self, tokens: np.ndarray, context_length: int, batch_size: int,
                 device: str = "cpu", seed: int = 0):
        assert len(tokens) > context_length + 1, "token file smaller than one sequence"
        self.tokens = tokens
        self.T = context_length
        self.B = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.tokens) - self.T - 1, size=self.B)
        x = np.stack([self.tokens[s: s + self.T] for s in starts]).astype(np.int64)
        y = np.stack([self.tokens[s + 1: s + self.T + 1] for s in starts]).astype(np.int64)
        return (torch.from_numpy(x).to(self.device),
                torch.from_numpy(y).to(self.device))
