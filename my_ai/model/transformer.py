"""
A decoder-only Transformer language model, implemented from scratch in PyTorch.

No pretrained weights. No Hugging Face model classes. Every layer below is
built from basic tensor ops + nn.Linear/nn.Embedding/nn.LayerNorm, which are
just parameter containers -- their weights start RANDOM and are learned by us.

What the pieces mean (plain language)
-------------------------------------
Token embeddings : a lookup table (vocab_size x d_model). Row i is a learned
    vector for token i. Initially random noise; training pushes tokens that
    behave similarly toward similar vectors.
Positional embeddings : a second table (context_length x d_model) so the model
    knows token ORDER; attention alone is order-blind.
Causal self-attention : each position builds a query, compares it against the
    keys of all EARLIER positions, and takes a weighted average of their
    values. "Causal" = a triangular mask sets attention to future positions
    to -inf, so position t can never peek at t+1. That is what makes
    next-token prediction honest.
Multi-head : we split d_model into n_heads independent attention computations
    so different heads can track different relationships (syntax, coreference,
    ...), then concatenate.
Feed-forward : a per-token 2-layer MLP (d_model -> d_ff -> d_model) with GELU.
    Most of the model's "knowledge" capacity lives here.
LayerNorm + residuals : x = x + sublayer(norm(x)). Residuals give gradients a
    highway through deep stacks; pre-norm keeps activations stable.
Dropout : randomly zeroes activations during training to reduce overfitting.
Output projection : maps the final d_model vector at each position to
    vocab_size raw scores (logits) = "how plausible is each token next?".
    We tie its weight to the embedding table (saves params, helps small models).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    context_length: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1
    tie_weights: bool = True

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            obj = json.load(f)
        fields = {k: v for k, v in obj.items() if k in cls.__dataclass_fields__}
        return cls(**fields)

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must divide evenly into heads"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        # one fused projection for Q, K, V (3x d_model outputs)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # causal mask: mask[t, s] = True where s > t (future positions)
        mask = torch.triu(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product: (B, h, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))  # causal mask
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                                   # (B, h, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-norm Transformer block: x = x + attn(ln(x)); x = x + ff(ln(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)  # RANDOM initialization -- no pretrained anything
        # GPT-2-style scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or "net.2.weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx     : (B, T) token ids
        targets : (B, T) ids shifted one to the left, or None
        returns : (logits (B, T, vocab), loss or None)
        """
        B, T = idx.shape
        assert T <= self.cfg.context_length, f"sequence length {T} > context {self.cfg.context_length}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # cross-entropy over every position; ignore_index=-100 lets the
            # chat fine-tuning stage mask out prompt tokens later.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
            if not self.cfg.tie_weights:
                n -= self.tok_emb.weight.numel()
        return n
