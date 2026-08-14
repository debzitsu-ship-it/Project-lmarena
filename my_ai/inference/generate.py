"""
Inference: turning a trained model into text, one token at a time.

Loop: encode prompt -> forward pass -> logits for the LAST position ->
turn logits into a probability distribution -> pick one token -> append ->
repeat. Every knob below only changes the "pick one token" step:

temperature : divide logits by t before softmax. t<1 sharpens (more
              deterministic), t>1 flattens (more random). t=0 => greedy.
top-k       : keep only the k highest-probability tokens, renormalize.
top-p       : keep the smallest set of tokens whose cumulative probability
              exceeds p (nucleus sampling), renormalize.
repetition penalty : divide the logits of already-generated tokens by a
              factor > 1 so the model is less keen to repeat itself.
stop tokens : cut generation when any of these token ids is produced
              (e.g. <eos>, or <|user|> in chat mode).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from my_ai.model.transformer import TransformerLM
from my_ai.tokenizer.tokenizer import EOS


@torch.no_grad()
def generate(model: TransformerLM, ids: list[int], max_new_tokens: int = 200,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, repetition_penalty: float = 1.0,
             stop_tokens: set[int] | None = None, device: str = "cpu",
             stream_callback=None) -> list[int]:
    """Return the newly generated token ids (prompt not included)."""
    model.eval()
    stop_tokens = stop_tokens or {EOS}
    ctx = model.cfg.context_length
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    new_ids: list[int] = []

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -ctx:]                    # crop to context window
        logits, _ = model(idx_cond)
        logits = logits[0, -1, :]                   # last position only

        if repetition_penalty != 1.0 and new_ids:
            for t in set(new_ids):
                if logits[t] > 0:
                    logits[t] /= repetition_penalty
                else:
                    logits[t] *= repetition_penalty

        if temperature <= 0:                        # greedy decoding
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = float("-inf")
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                cutoff = cum - probs > top_p        # keep first token past p too
                sorted_logits[cutoff] = float("-inf")
                logits = torch.full_like(logits, float("-inf"))
                logits[sorted_idx] = sorted_logits
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

        if next_id in stop_tokens:
            break
        new_ids.append(next_id)
        if stream_callback:
            stream_callback(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)

    return new_ids


def generate_text(model: TransformerLM, tokenizer, prompt: str, **kwargs) -> str:
    """Prompt may contain literal special-token markers like '<|user|>';
    they are encoded as their reserved single IDs, not as raw bytes."""
    from my_ai.tokenizer.tokenizer import encode_with_specials
    ids = encode_with_specials(tokenizer, prompt)
    out = generate(model, ids, **kwargs)
    return prompt + tokenizer.decode(out)
