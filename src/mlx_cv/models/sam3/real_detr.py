"""Faithful MLX port of the SAM 3 DETR encoder + dot-product scoring head.

Mirrors ``transformers.models.sam3.modeling_sam3`` so the real ``facebook/sam3``
``detector_model.detr_encoder.*`` (156) and ``dot_product_scoring.*`` (10) tensors
load 1:1. Both subsystems are on the text-prompt critical path:

- ``Sam3DetrEncoder``: 6 layers of vision self-attention (with additive position
  encoding on q/k) + cross-attention to the text features + ReLU MLP. The numeric
  tap is ``last_hidden_state`` (fused multi-level vision features).
- ``Sam3DotProductScoring``: residual text MLP + masked mean pool + dot product of
  projected decoder queries against projected pooled text. The tap is ``scores``.

No torch/transformers imports. See
``.agent/work/2026-06-18-sam3-real-architecture-port/PLAN.md`` (slice 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .real_config import Sam3DETRDecoderConfig, Sam3DETREncoderConfig

__all__ = [
    "Sam3Attention",
    "Sam3FFN",
    "Sam3DetrEncoderLayer",
    "Sam3DetrEncoder",
    "Sam3DotProductScoring",
    "Sam3DetrEncoderOutput",
]


_ACT = {"relu": nn.relu, "gelu": nn.gelu}


class Sam3Attention(nn.Module):
    """Standard multi-head attention with separate q/k/v/o projections."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(
        self, query: mx.array, key: mx.array, value: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        batch_size, q_len, _ = query.shape
        k_len = key.shape[1]
        queries = self.q_proj(query).reshape(batch_size, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(key).reshape(batch_size, k_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(value).reshape(batch_size, k_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn_weights = (queries @ keys.transpose(0, 1, 3, 2)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = attn_weights @ values
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, q_len, -1)
        return self.o_proj(attn_output)


class Sam3FFN(nn.Module):
    """``Sam3MLP``: fc1 -> activation -> fc2 (ReLU for the detr/geometry stacks)."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = _ACT[hidden_act]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.fc2(self.activation(self.fc1(hidden_states)))


class Sam3DetrEncoderLayer(nn.Module):
    def __init__(self, config: Sam3DETREncoderConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Sam3Attention(config.hidden_size, config.num_attention_heads)
        self.cross_attn = Sam3Attention(config.hidden_size, config.num_attention_heads)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Sam3FFN(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        vision_feats: mx.array,
        prompt_feats: mx.array,
        vision_pos_encoding: mx.array,
        prompt_cross_attn_mask: mx.array | None = None,
    ) -> mx.array:
        # Vision self-attention (position encoding added to q/k, not v).
        residual = vision_feats
        hidden_states = self.layer_norm1(vision_feats)
        hidden_with_pos = hidden_states + vision_pos_encoding
        hidden_states = self.self_attn(hidden_with_pos, hidden_with_pos, hidden_states)
        hidden_states = residual + hidden_states

        # Cross-attention: vision queries attend to text/prompt features.
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.cross_attn(hidden_states, prompt_feats, prompt_feats, prompt_cross_attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@dataclass
class Sam3DetrEncoderOutput:
    last_hidden_state: mx.array
    pos_embeds_flattened: mx.array
    text_features: mx.array


class Sam3DetrEncoder(nn.Module):
    """DETR-style encoder fusing multi-level vision features with text prompts."""

    def __init__(self, config: Sam3DETREncoderConfig):
        super().__init__()
        self.config = config
        self.layers = [Sam3DetrEncoderLayer(config) for _ in range(config.num_layers)]

    @staticmethod
    def _flatten_levels(features: list[mx.array], pos_embeds: list[mx.array]) -> tuple[mx.array, mx.array]:
        # Each level is NHWC [B, H, W, C]; flatten spatial and concat across levels.
        flat_feats = [f.reshape(f.shape[0], f.shape[1] * f.shape[2], f.shape[3]) for f in features]
        flat_pos = [p.reshape(p.shape[0], p.shape[1] * p.shape[2], p.shape[3]) for p in pos_embeds]
        return mx.concatenate(flat_feats, axis=1), mx.concatenate(flat_pos, axis=1)

    def __call__(
        self,
        vision_features: list[mx.array],
        text_features: mx.array,
        vision_pos_embeds: list[mx.array],
        prompt_cross_attn_mask: mx.array | None = None,
    ) -> Sam3DetrEncoderOutput:
        features_flattened, pos_embeds_flattened = self._flatten_levels(vision_features, vision_pos_embeds)
        hidden_states = features_flattened
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                prompt_feats=text_features,
                vision_pos_encoding=pos_embeds_flattened,
                prompt_cross_attn_mask=prompt_cross_attn_mask,
            )
        return Sam3DetrEncoderOutput(
            last_hidden_state=hidden_states,
            pos_embeds_flattened=pos_embeds_flattened,
            text_features=text_features,
        )


class Sam3DotProductScoring(nn.Module):
    """Confidence scoring via dot product of projected queries and pooled text."""

    def __init__(self, config: Sam3DETRDecoderConfig):
        super().__init__()
        hidden_size = config.hidden_size
        # Sam3DecoderMLP(num_layers=2): layer1 -> relu -> layer2.
        self.text_mlp = _ScoringMLP(hidden_size, config.intermediate_size)
        self.text_mlp_out_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = float(1.0 / math.sqrt(hidden_size))
        self.clamp_max_val = 12.0

    def __call__(
        self, decoder_hidden_states: mx.array, text_features: mx.array, text_mask: mx.array | None = None
    ) -> mx.array:
        residual = text_features
        text_features = self.text_mlp(text_features)
        text_features = text_features + residual
        text_features = self.text_mlp_out_norm(text_features)

        if text_mask is None:
            pooled_text = mx.mean(text_features, axis=1)
        else:
            is_valid = text_mask.astype(text_features.dtype)[..., None]
            num_valid = mx.maximum(mx.sum(is_valid, axis=1), mx.array(1.0))
            pooled_text = mx.sum(text_features * is_valid, axis=1) / num_valid

        proj_text = self.text_proj(pooled_text)  # [B, hidden]
        proj_queries = self.query_proj(decoder_hidden_states)  # [..., B?, Q, hidden]

        # decoder_hidden_states: [num_layers, B, Q, hidden]; proj_text: [B, hidden].
        proj_text = proj_text[..., None]  # [B, hidden, 1]
        scores = proj_queries @ proj_text[None]  # [L, B, Q, 1]
        scores = scores * self.scale
        return mx.clip(scores, -self.clamp_max_val, self.clamp_max_val)


class _ScoringMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, intermediate_size)
        self.layer2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.layer2(nn.relu(self.layer1(hidden_states)))
