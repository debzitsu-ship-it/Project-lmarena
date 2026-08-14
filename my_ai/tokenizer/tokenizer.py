"""
Tokenizers implemented from scratch. No external tokenizer libraries.

Concepts
--------
Vocabulary     : the fixed set of strings (tokens) the model knows.
Token IDs      : each token gets an integer index; the model only ever
                 sees integers, never text.
Special tokens : reserved IDs with meaning to the *application*, not to
                 language itself: <pad> (batch padding), <bos>/<eos>
                 (sequence boundaries), <unk> (unknown), and chat-role
                 markers <|user|>, <|assistant|>, <|system|>.
Encoding       : text -> list[int].
Decoding       : list[int] -> text (should round-trip).
Unknown tokens : characters never seen in training map to <unk> in the
                 char tokenizer. The BPE tokenizer operates on raw BYTES,
                 so it can encode *any* text and <unk> is never needed --
                 that's one big reason real LLMs use byte-level BPE.
Context length : the model has a maximum number of tokens it can attend
                 over (e.g. 256). The tokenizer doesn't enforce it; the
                 dataset/inference code truncates to it.
"""
from __future__ import annotations

import json
import os
from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<|user|>", "<|assistant|>", "<|system|>"]
PAD, BOS, EOS, UNK = 0, 1, 2, 3
USER_TOK, ASSISTANT_TOK, SYSTEM_TOK = 4, 5, 6
N_SPECIAL = len(SPECIAL_TOKENS)


class CharTokenizer:
    """Character-level tokenizer. Vocab = every character seen in training."""

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return N_SPECIAL + len(self.stoi)

    def train(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i + N_SPECIAL for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(ch, UNK) for ch in text]
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i < N_SPECIAL:
                continue  # skip special tokens when rendering text
            out.append(self.itos.get(i, ""))
        return "".join(out)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "stoi": self.stoi}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        tok = cls()
        tok.stoi = obj["stoi"]
        tok.itos = {i: ch for ch, i in tok.stoi.items()}
        return tok


class BPETokenizer:
    """
    Byte-level Byte-Pair Encoding, from scratch.

    Training:
      1. Start with a base vocab of all 256 byte values.
      2. Count every adjacent pair of tokens in the corpus.
      3. Merge the most frequent pair into a new token.
      4. Repeat until vocab_size is reached.

    Because the base alphabet is raw bytes, ANY string can be encoded --
    there are no unknown tokens. Merges learned on our corpus become
    subwords: frequent words become single tokens, rare words split into
    pieces.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}   # (a, b) -> merged id
        self.vocab: dict[int, bytes] = {}              # id -> byte string
        self._word_cache: dict[str, list[int]] = {}    # speeds up repeated words
        self._build_base_vocab()

    def _build_base_vocab(self) -> None:
        # ids [0, N_SPECIAL) are special; bytes occupy [N_SPECIAL, N_SPECIAL+256)
        self.vocab = {N_SPECIAL + b: bytes([b]) for b in range(256)}

    @property
    def vocab_size(self) -> int:
        return N_SPECIAL + 256 + len(self.merges)

    # ---------------- training ----------------
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= N_SPECIAL + 256, "vocab_size must cover specials + 256 bytes"
        n_merges = vocab_size - N_SPECIAL - 256

        # Work on word chunks so merges never cross whitespace boundaries
        # in weird ways; keep the leading space attached to a word (like GPT-2)
        # so " the" and "the" are learned as units.
        words = self._split_words(text)
        word_counts = Counter(words)
        # each unique word -> current token sequence
        seqs: dict[str, list[int]] = {
            w: [N_SPECIAL + b for b in w.encode("utf-8")] for w in word_counts
        }

        for step in range(n_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for w, seq in seqs.items():
                c = word_counts[w]
                for a, b in zip(seq, seq[1:]):
                    pair_counts[(a, b)] += c
            if not pair_counts:
                break
            (a, b), freq = pair_counts.most_common(1)[0]
            if freq < 2:
                break  # nothing worth merging
            new_id = N_SPECIAL + 256 + len(self.merges)
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            for w in seqs:
                seqs[w] = self._apply_merge(seqs[w], (a, b), new_id)
            if verbose and (step % 100 == 0 or step == n_merges - 1):
                print(f"merge {step + 1}/{n_merges}: {self.vocab[new_id]!r} (freq {freq})")

    @staticmethod
    def _split_words(text: str) -> list[str]:
        """Split into words, keeping the leading space attached."""
        words, cur = [], ""
        for ch in text:
            if ch == " ":
                if cur:
                    words.append(cur)
                cur = " "
            elif ch in "\n\t":
                if cur:
                    words.append(cur)
                words.append(ch)
                cur = ""
            else:
                cur += ch
        if cur:
            words.append(cur)
        return words

    @staticmethod
    def _apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # ---------------- encode / decode ----------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        for word in self._split_words(text):
            cached = self._word_cache.get(word)
            if cached is not None:
                ids.extend(cached)
                continue
            seq = [N_SPECIAL + b for b in word.encode("utf-8")]
            # apply merges in the order they were learned (lowest id = earliest)
            while len(seq) >= 2:
                best_pair, best_id = None, None
                for a, b in zip(seq, seq[1:]):
                    mid = self.merges.get((a, b))
                    if mid is not None and (best_id is None or mid < best_id):
                        best_pair, best_id = (a, b), mid
                if best_pair is None:
                    break
                seq = self._apply_merge(seq, best_pair, best_id)
            if len(self._word_cache) < 200_000:
                self._word_cache[word] = seq
            ids.extend(seq)
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: list[int]) -> str:
        chunks = []
        for i in ids:
            if i < N_SPECIAL:
                continue
            chunks.append(self.vocab.get(i, b""))
        return b"".join(chunks).decode("utf-8", errors="replace")

    def encode_special(self, token: str) -> int:
        return SPECIAL_TOKENS.index(token)

    # ---------------- persistence ----------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "type": "bpe",
                "merges": [[a, b, mid] for (a, b), mid in self.merges.items()],
            }, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        tok = cls()
        for a, b, mid in obj["merges"]:
            tok.merges[(a, b)] = mid
            tok.vocab[mid] = tok.vocab[a] + tok.vocab[b]
        return tok


def encode_with_specials(tokenizer, text: str) -> list[int]:
    """
    Encode text that may contain LITERAL special-token markers such as
    "<|user|>" or "<eos>". The markers become their reserved single IDs;
    everything between them is encoded normally. This is how chat-formatted
    training data gets its role tokens.
    """
    import re
    pattern = "(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")"
    ids: list[int] = []
    for part in re.split(pattern, text):
        if not part:
            continue
        if part in SPECIAL_TOKENS:
            ids.append(SPECIAL_TOKENS.index(part))
        else:
            ids.extend(tokenizer.encode(part))
    return ids


def load_tokenizer(path: str):
    """Load either tokenizer type from a saved JSON file."""
    with open(path, encoding="utf-8") as f:
        t = json.load(f)["type"]
    return CharTokenizer.load(path) if t == "char" else BPETokenizer.load(path)
