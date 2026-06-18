"""Tiny SAM 3.1 text encoder path in MLX."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .tokenizer import DEFAULT_CONTEXT_LENGTH, SAM3Tokenizer

__all__ = [
    "SAM3TextConfig",
    "SAM3TextEncoder",
    "SAM3TextOutput",
]


@dataclass(frozen=True)
class SAM3TextConfig:
    d_model: int = 256
    context_length: int = DEFAULT_CONTEXT_LENGTH
    vocab_size: int = 514
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    use_causal_mask: bool = True

    def __post_init__(self) -> None:
        if min(self.d_model, self.context_length, self.vocab_size, self.width, self.heads, self.layers) <= 0:
            raise ValueError("SAM3 text config dimensions must be positive")
        if self.width % self.heads != 0:
            raise ValueError("SAM3 text width must be divisible by heads")
        if self.mlp_ratio <= 0:
            raise ValueError("SAM3 text mlp_ratio must be positive")


@dataclass
class SAM3TextOutput:
    language_mask: mx.array
    language_features: mx.array
    language_embeds: mx.array
    token_ids: mx.array


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2.0)))


def _causal_mask(seq_len: int, dtype) -> mx.array:
    mask = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=1)
    return mx.where(mask, mx.array(-1e9, dtype=dtype), mx.array(0.0, dtype=dtype))


class _SAM3TextBlock(nn.Module):
    def __init__(self, cfg: SAM3TextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.width // cfg.heads
        self.ln_1 = nn.LayerNorm(cfg.width)
        self.q_proj = nn.Linear(cfg.width, cfg.width)
        self.k_proj = nn.Linear(cfg.width, cfg.width)
        self.v_proj = nn.Linear(cfg.width, cfg.width)
        self.out_proj = nn.Linear(cfg.width, cfg.width)
        self.ln_2 = nn.LayerNorm(cfg.width)
        hidden = int(cfg.width * cfg.mlp_ratio)
        self.mlp_fc = nn.Linear(cfg.width, hidden)
        self.mlp_proj = nn.Linear(hidden, cfg.width)

    def _attention(self, x: mx.array, mask: mx.array | None) -> mx.array:
        batch, seq_len, width = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.cfg.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch, seq_len, self.cfg.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch, seq_len, self.cfg.heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask[None, None, :, :]
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, width)
        return self.out_proj(out)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        x = x + self._attention(self.ln_1(x), mask)
        x = x + self.mlp_proj(_gelu(self.mlp_fc(self.ln_2(x))))
        return x


class SAM3TextEncoder(nn.Module):
    """SAM3 text encoder wrapper for token ids or raw strings."""

    def __init__(self, cfg: SAM3TextConfig, tokenizer: SAM3Tokenizer | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.positional_embedding = mx.zeros((cfg.context_length, cfg.width))
        self.blocks = [_SAM3TextBlock(cfg) for _ in range(cfg.layers)]
        self.ln_final = nn.LayerNorm(cfg.width)
        self.resizer = nn.Linear(cfg.width, cfg.d_model)

    def _tokenize(self, text: list[str] | tuple[str, ...] | str) -> mx.array:
        if self.tokenizer is None:
            raise ValueError("SAM3TextEncoder requires a tokenizer for string inputs")
        return mx.array(self.tokenizer(text, context_length=self.cfg.context_length), dtype=mx.int32)

    def __call__(self, text_or_tokens) -> SAM3TextOutput:
        token_ids = self._tokenize(text_or_tokens) if isinstance(text_or_tokens, (str, list, tuple)) else text_or_tokens
        token_ids = mx.array(token_ids, dtype=mx.int32)
        if len(token_ids.shape) != 2:
            raise ValueError(f"SAM3TextEncoder expects token ids shape (B,T), got {token_ids.shape}")
        batch, seq_len = token_ids.shape
        if seq_len > self.cfg.context_length:
            raise ValueError(f"SAM3TextEncoder seq_len {seq_len} exceeds context_length {self.cfg.context_length}")
        if int(mx.max(token_ids).item()) >= self.cfg.vocab_size:
            raise ValueError("SAM3TextEncoder token id exceeds vocab_size")

        embeds = self.token_embedding(token_ids)
        x = embeds + self.positional_embedding[:seq_len]
        attn_mask = _causal_mask(seq_len, x.dtype) if self.cfg.use_causal_mask else None
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln_final(x)
        resized = self.resizer(x)
        padding_mask = token_ids == 0
        return SAM3TextOutput(
            language_mask=padding_mask,
            language_features=resized.transpose(1, 0, 2),
            language_embeds=embeds.transpose(1, 0, 2),
            token_ids=token_ids,
        )
