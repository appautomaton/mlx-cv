"""Qwen2.5 language-backbone building blocks.

This submodule is the MLX boundary for Qwen2. Package-root/config imports stay
mlx-free; later slices extend this file with attention, decoder layers, and the
registered builder.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import Qwen2Config

__all__ = [
    "Qwen2RMSNorm",
    "Qwen2RotaryEmbedding",
    "Qwen2Attention",
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

    def __call__(self, x: mx.array, seq_len: int | None = None) -> tuple[mx.array, mx.array]:
        seq_len = int(seq_len if seq_len is not None else x.shape[-2])
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
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


class Qwen2Attention(nn.Module):
    """Manual additive-mask GQA attention for Qwen2."""

    def __init__(self, config: Qwen2Config, layer_idx: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads: {self.hidden_size} vs {self.num_heads}"
            )
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{self.num_heads} vs {self.num_key_value_heads}"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        past_key_value=None,
        *,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[mx.array, mx.array | None, object | None]:
        if past_key_value is not None or use_cache:
            raise NotImplementedError("Qwen2Attention cache support is added in the cache slice")

        batch, q_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = mx.broadcast_to(
                mx.arange(q_len, dtype=mx.int32).reshape(1, q_len),
                (batch, q_len),
            )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = mx.transpose(
            query_states.reshape(batch, q_len, self.num_heads, self.head_dim),
            (0, 2, 1, 3),
        )
        key_states = mx.transpose(
            key_states.reshape(batch, q_len, self.num_key_value_heads, self.head_dim),
            (0, 2, 1, 3),
        )
        value_states = mx.transpose(
            value_states.reshape(batch, q_len, self.num_key_value_heads, self.head_dim),
            (0, 2, 1, 3),
        )

        kv_seq_len = key_states.shape[-2]
        rotary_seq_len = max(kv_seq_len, int(mx.max(position_ids)) + 1)
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (query_states @ mx.transpose(key_states, (0, 1, 3, 2))) / math.sqrt(
            self.head_dim
        )
        expected_shape = (batch, self.num_heads, q_len, kv_seq_len)
        if attn_weights.shape != expected_shape:
            raise ValueError(f"Attention weights should have shape {expected_shape}, got {attn_weights.shape}")

        if attention_mask is not None:
            mask_shape = (batch, 1, q_len, kv_seq_len)
            if attention_mask.shape != mask_shape:
                raise ValueError(f"Attention mask should have shape {mask_shape}, got {attention_mask.shape}")
            attn_weights = attn_weights + attention_mask

        attn_probs = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(query_states.dtype)
        attn_output = attn_probs @ value_states
        output_shape = (batch, self.num_heads, q_len, self.head_dim)
        if attn_output.shape != output_shape:
            raise ValueError(f"Attention output should have shape {output_shape}, got {attn_output.shape}")

        attn_output = mx.transpose(attn_output, (0, 2, 1, 3)).reshape(batch, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_probs if output_attentions else None, None
