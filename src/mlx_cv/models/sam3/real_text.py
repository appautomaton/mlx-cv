"""Faithful MLX port of the SAM 3 text encoder (CLIP text tower + projections).

Mirrors how ``transformers.models.sam3.Sam3Model`` wraps a stock
``CLIPTextModelWithProjection`` plus an outer ``text_projection`` Linear, so the
real ``facebook/sam3`` ``detector_model.text_encoder.*`` / ``text_projection.*``
tensors load 1:1 (391 keys). Parameter paths match the upstream state dict exactly.

The numeric parity tap is ``Sam3TextEncoderOutput.pooler_output`` — i.e.
``text_projection(text_encoder.last_hidden_state)`` of shape ``[B, seq, 256]`` —
matching ``Sam3Model.get_text_features(...).pooler_output``. No torch/transformers
imports. See ``.agent/work/2026-06-18-sam3-real-architecture-port/PLAN.md`` (slice 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .real_config import Sam3TextConfig

__all__ = [
    "Sam3TextEncoderOutput",
    "Sam3CLIPTextTransformer",
    "Sam3CLIPTextModelWithProjection",
    "Sam3TextEncoder",
    "build_sam3_text_real",
]

_MASK_FILL = -1e9


# --- CLIP text submodules -----------------------------------------------------


class Sam3CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        seq_len = input_ids.shape[-1]
        position_ids = mx.arange(seq_len)
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class Sam3CLIPAttention(nn.Module):
    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        embed_dim = config.hidden_size
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array | None) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        queries = self.q_proj(hidden_states).reshape(*shape).transpose(0, 2, 1, 3)
        keys = self.k_proj(hidden_states).reshape(*shape).transpose(0, 2, 1, 3)
        values = self.v_proj(hidden_states).reshape(*shape).transpose(0, 2, 1, 3)

        attn_weights = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = attn_weights @ values  # [B, heads, seq, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class Sam3CLIPMLP(nn.Module):
    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(hidden_states)))


class Sam3CLIPEncoderLayer(nn.Module):
    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.self_attn = Sam3CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Sam3CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array | None) -> mx.array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Sam3CLIPEncoder(nn.Module):
    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.layers = [Sam3CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array | None) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


def _causal_mask(seq_len: int, attention_mask: mx.array | None) -> mx.array:
    """Additive causal (+ optional padding) mask broadcastable to [B, heads, seq, seq]."""

    mask = mx.triu(mx.full((seq_len, seq_len), _MASK_FILL), k=1)[None, None]
    if attention_mask is not None:
        padding = (1.0 - attention_mask.astype(mx.float32)) * _MASK_FILL
        mask = mask + padding[:, None, None, :]
    return mask


class Sam3CLIPTextTransformer(nn.Module):
    """Stock CLIP text transformer (``text_encoder.text_model``)."""

    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.embeddings = Sam3CLIPTextEmbeddings(config)
        self.encoder = Sam3CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None) -> mx.array:
        hidden_states = self.embeddings(input_ids)
        mask = _causal_mask(input_ids.shape[-1], attention_mask)
        hidden_states = self.encoder(hidden_states, mask)
        return self.final_layer_norm(hidden_states)


class Sam3CLIPTextModelWithProjection(nn.Module):
    """CLIP text transformer + inner CLIP ``text_projection`` (``text_encoder``)."""

    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.text_model = Sam3CLIPTextTransformer(config)
        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None) -> tuple[mx.array, mx.array]:
        last_hidden_state = self.text_model(input_ids, attention_mask)
        # CLIP pools at the EOT token (highest id in each sequence) for text_embeds.
        eot = mx.argmax(input_ids.astype(mx.int32), axis=-1)
        batch_size, _, hidden = last_hidden_state.shape
        index = mx.broadcast_to(eot[:, None, None], (batch_size, 1, hidden))
        pooled = mx.take_along_axis(last_hidden_state, index, axis=1)[:, 0, :]
        text_embeds = self.text_projection(pooled)
        return last_hidden_state, text_embeds


# --- top-level SAM3 text encoder ----------------------------------------------


@dataclass
class Sam3TextEncoderOutput:
    last_hidden_state: mx.array  # [B, seq, hidden]
    pooler_output: mx.array  # [B, seq, detr_hidden] — the SAM3 parity tap
    text_embeds: mx.array  # [B, projection_dim] — inner CLIP projection of the EOT token


class Sam3TextEncoder(nn.Module):
    """Faithful SAM 3 text encoder: CLIP tower + inner/outer projections (391 tensors)."""

    def __init__(self, config: Sam3TextConfig, detr_hidden_size: int = 256):
        super().__init__()
        self.config = config
        self.text_encoder = Sam3CLIPTextModelWithProjection(config)
        self.text_projection = nn.Linear(config.hidden_size, detr_hidden_size)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> Sam3TextEncoderOutput:
        last_hidden_state, text_embeds = self.text_encoder(input_ids, attention_mask)
        pooler_output = self.text_projection(last_hidden_state)
        return Sam3TextEncoderOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            text_embeds=text_embeds,
        )


def build_sam3_text_real(config: Sam3TextConfig, detr_hidden_size: int = 256) -> Sam3TextEncoder:
    return Sam3TextEncoder(config, detr_hidden_size=detr_hidden_size)
