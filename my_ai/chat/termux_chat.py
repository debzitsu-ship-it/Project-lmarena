"""
Chat with the from-scratch model using ONLY NumPy — no PyTorch needed.
Made for phones (Termux on Android) and any machine where torch won't install.

This re-implements the exact same Transformer forward pass as
my_ai/model/transformer.py, but with plain NumPy matrix math. Same weights,
same tokenizer, same sampling. 100% local; no external AI API.

Termux setup (Android):
    pkg update
    pkg install git python python-numpy
    git clone -b arena/01a0011c-project-lmarena https://github.com/debzitsu-ship-it/Project-lmarena.git
    cd Project-lmarena
    python -m my_ai.chat.termux_chat

Type your message and press Enter. Commands: remember: <fact> | facts | reset | quit
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# tokenizer is pure Python -- safe to import (no torch anywhere in it)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from my_ai.tokenizer.tokenizer import (EOS, USER_TOK, ASSISTANT_TOK, SYSTEM_TOK,
                                       load_tokenizer)


# --------------------------------------------------------------- model (NumPy)
class NumpyLM:
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.w = {k: data[k] for k in data.files if k != "__config__"}
        self.cfg = json.loads(bytes(data["__config__"]).decode())
        self.n_layers = self.cfg["n_layers"]
        self.n_heads = self.cfg["n_heads"]
        self.head_dim = self.cfg["d_model"] // self.n_heads
        self.ctx = self.cfg["context_length"]

    @staticmethod
    def _ln(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5) * g + b

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    def forward_last(self, ids: list[int]) -> np.ndarray:
        """Return logits for the LAST position only. ids length <= ctx."""
        w = self.w
        T = len(ids)
        x = w["tok_emb.weight"][ids] + w["pos_emb.weight"][:T]        # (T, C)
        mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)  # causal

        for i in range(self.n_layers):
            p = f"blocks.{i}."
            h = self._ln(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            qkv = h @ w[p + "attn.qkv.weight"].T                      # (T, 3C)
            q, k, v = np.split(qkv, 3, axis=-1)
            # (T, C) -> (heads, T, head_dim)
            q = q.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)
            k = k.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)
            v = v.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)
            att = q @ k.transpose(0, 2, 1) / np.sqrt(self.head_dim) + mask
            y = self._softmax(att) @ v                                # (h, T, hd)
            y = y.transpose(1, 0, 2).reshape(T, -1)
            x = x + y @ w[p + "attn.proj.weight"].T

            h = self._ln(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            h = self._gelu(h @ w[p + "ff.net.0.weight"].T + w[p + "ff.net.0.bias"])
            x = x + h @ w[p + "ff.net.2.weight"].T + w[p + "ff.net.2.bias"]

        x = self._ln(x[-1:], w["ln_f.weight"], w["ln_f.bias"])        # last pos
        return (x @ w["lm_head.weight"].T)[0]                         # (vocab,)


# --------------------------------------------------------------- sampling
def sample(logits: np.ndarray, generated: list[int], temperature=0.7,
           top_k=40, top_p=0.95, rep_penalty=1.1, rng=None) -> int:
    rng = rng or np.random.default_rng()
    logits = logits.astype(np.float64).copy()
    for t in set(generated):
        logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty
    if temperature <= 0:
        return int(logits.argmax())
    logits /= temperature
    if top_k:
        kth = np.sort(logits)[-top_k]
        logits[logits < kth] = -1e9
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p and 0 < top_p < 1:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        cut = order[cum - probs[order] > top_p]
        probs[cut] = 0
        probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(model: NumpyLM, ids: list[int], max_new=150, **kw) -> list[int]:
    stop = {EOS, USER_TOK, ASSISTANT_TOK}
    out: list[int] = []
    for _ in range(max_new):
        logits = model.forward_last(ids[-model.ctx:])
        nxt = sample(logits, out, **kw)
        if nxt in stop:
            break
        out.append(nxt)
        ids.append(nxt)
        # stream the token as it is produced
        sys.stdout.write(TOKENIZER.decode([nxt]))
        sys.stdout.flush()
    return out


# --------------------------------------------------------------- chat loop
def build_prompt(tokenizer, facts, history, user_msg, ctx) -> list[int]:
    ids: list[int] = []
    for f in facts:
        ids += [SYSTEM_TOK] + tokenizer.encode(f"fact: {f}\n")
    for turn in history:
        role = USER_TOK if turn["role"] == "user" else ASSISTANT_TOK
        ids += [role] + tokenizer.encode(turn["text"]) + [EOS]
    ids += [USER_TOK] + tokenizer.encode(user_msg) + [EOS, ASSISTANT_TOK]
    return ids[-(ctx - 8):]


TOKENIZER = None


def main() -> None:
    global TOKENIZER
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz = os.path.join(root, "checkpoints", "model_numpy.npz")
    tok_path = os.path.join(root, "checkpoints", "tokenizer.json")

    print("loading model (NumPy, no PyTorch) …")
    model = NumpyLM(npz)
    TOKENIZER = load_tokenizer(tok_path)
    n_params = sum(int(np.prod(v.shape)) for v in model.w.values())
    print(f"ready: {n_params:,} parameters | context {model.ctx} | 100% local\n"
          "commands: remember: <fact> | facts | reset | quit\n")

    history, facts = [], []
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        low = msg.lower()
        if low in ("quit", "exit", "/quit"):
            break
        if low == "reset":
            history = []
            print("[context cleared]\n")
            continue
        if low == "facts":
            print("[facts]", facts or "(none)", "\n")
            continue
        if low.startswith("remember:"):
            facts.append(msg.split(":", 1)[1].strip())
            print("[fact stored — in this session's memory, not the model's weights]\n")
            continue

        ids = build_prompt(TOKENIZER, facts, history, msg, model.ctx)
        print("AI: ", end="", flush=True)
        out = generate(model, ids)
        print("\n")
        history.append({"role": "user", "text": msg})
        history.append({"role": "assistant", "text": TOKENIZER.decode(out)})


if __name__ == "__main__":
    main()
