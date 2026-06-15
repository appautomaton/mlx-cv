"""Qwen2.5 language-backbone building blocks.

This submodule is the MLX boundary for Qwen2. Package-root/config imports stay
mlx-free; later slices extend this file with attention, decoder layers, and the
registered builder.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import Qwen2Config

__all__ = [
    "Qwen2RMSNorm",
    "Qwen2RotaryEmbedding",
    "Qwen2MLP",
    "rotate_half",
    "apply_rotary_pos_emb",
    "repeat_kv",
]


class Qwen2RMSNorm(nn.Module):
    """Qwen2 RMSNorm, equivalent to the reference Llama/T5-style layer norm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        x = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.astype(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    """1D Qwen2 RoPE table with the reference half-split convention."""

    def __init__(
        self,
        dim: int,
        *,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )

    def __call__(self, x: mx.array, seq_len: int | None = None) -> tuple[mx.array, mx.array]:
        seq_len = int(seq_len if seq_len is not None else x.shape[-2])
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


def rotate_half(x: mx.array) -> mx.array:
    """Rotate the last-dimension halves: ``[x1, x2] -> [-x2, x1]``."""
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    position_ids: mx.array,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[mx.array, mx.array]:
    """Apply gathered RoPE tables to query/key tensors."""
    cos = mx.expand_dims(cos[position_ids], axis=unsqueeze_dim)
    sin = mx.expand_dims(sin[position_ids], axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Module):
    """Reference SwiGLU block: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise NotImplementedError(f"Qwen2MLP only supports hidden_act='silu', got {config.hidden_act!r}")
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        return self.down_proj((gate * mx.sigmoid(gate)) * self.up_proj(x))


def repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
    """Expand KV heads from ``(B, kv_heads, T, D)`` to attention heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = mx.expand_dims(hidden_states, axis=2)
    hidden_states = mx.broadcast_to(
        hidden_states,
        (batch, num_key_value_heads, n_rep, slen, head_dim),
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
