"""Faithful MLX port of the SAM 3 DETR decoder (``Sam3DetrDecoder``).

Mirrors ``transformers.models.sam3.modeling_sam3`` so the real ``facebook/sam3``
``detector_model.detr_decoder.*`` tensors load 1:1 (247 keys): a 6-layer,
200-query decoder with a presence token, post-norm self/text-cross/vision-cross
attention, box relative-position-bias (log-scale) on the vision cross-attention,
and per-layer iterative box refinement.

Taps (PLAN slice 5): ``intermediate_hidden_states`` ``[L, B, Q, hidden]``,
``reference_boxes`` ``[L, B, Q, 4]``, ``presence_logits`` ``[L, B, 1]``. No
torch/transformers imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .real_config import Sam3DETRDecoderConfig
from .real_detr import Sam3Attention, Sam3FFN
from .real_vision import Sam3SinePositionEmbedding

__all__ = ["Sam3DetrDecoder", "Sam3DetrDecoderLayer", "Sam3DetrDecoderOutput"]


def inverse_sigmoid(x: mx.array, eps: float = 1e-3) -> mx.array:
    x = mx.clip(x, 0.0, 1.0)
    x1 = mx.clip(x, eps, None)
    x2 = mx.clip(1.0 - x, eps, None)
    return mx.log(x1 / x2)


def box_cxcywh_to_xyxy(boxes: mx.array) -> mx.array:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return mx.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


class Sam3DecoderMLP(nn.Module):
    """2- or 3-layer ReLU MLP with ``layer1``/``layer2``/``layer3`` naming."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        if num_layers == 2:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.layer3 = None
        elif num_layers == 3:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Only 2 or 3 layers supported, got {num_layers}")

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.layer1(x))
        if self.layer3 is not None:
            x = nn.relu(self.layer2(x))
            return self.layer3(x)
        return self.layer2(x)


class Sam3DetrDecoderLayer(nn.Module):
    def __init__(self, config: Sam3DETRDecoderConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Sam3Attention(hidden_size, config.num_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.text_cross_attn = Sam3Attention(hidden_size, config.num_attention_heads)
        self.text_cross_attn_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.vision_cross_attn = Sam3Attention(hidden_size, config.num_attention_heads)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = Sam3FFN(hidden_size, config.intermediate_size, config.hidden_act)
        self.mlp_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        query_pos: mx.array,
        text_features: mx.array,
        vision_features: mx.array,
        vision_pos_encoding: mx.array,
        text_cross_attn_mask: mx.array | None = None,
        vision_cross_attn_mask: mx.array | None = None,
    ) -> mx.array:
        # Prepend a zero query_pos row for the presence token (position 0).
        query_pos = mx.pad(query_pos, [(0, 0), (1, 0), (0, 0)])

        # Self-attention (post-norm).
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        attn = self.self_attn(query_with_pos, query_with_pos, hidden_states)
        hidden_states = self.self_attn_layer_norm(residual + attn)

        # Text cross-attention (post-norm).
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        attn = self.text_cross_attn(query_with_pos, text_features, text_features, text_cross_attn_mask)
        hidden_states = self.text_cross_attn_layer_norm(residual + attn)

        # Vision cross-attention with RPB (post-norm).
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        key_with_pos = vision_features + vision_pos_encoding
        attn = self.vision_cross_attn(query_with_pos, key_with_pos, vision_features, vision_cross_attn_mask)
        hidden_states = self.vision_cross_attn_layer_norm(residual + attn)

        # MLP (post-norm).
        residual = hidden_states
        hidden_states = self.mlp_layer_norm(residual + self.mlp(hidden_states))
        return hidden_states


@dataclass
class Sam3DetrDecoderOutput:
    intermediate_hidden_states: mx.array  # [L, B, Q, hidden]
    reference_boxes: mx.array  # [L, B, Q, 4]
    presence_logits: mx.array  # [L, B, 1]


class Sam3DetrDecoder(nn.Module):
    """6-layer, 200-query DETR decoder with box RPB and iterative refinement."""

    def __init__(self, config: Sam3DETRDecoderConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.layers = [Sam3DetrDecoderLayer(config) for _ in range(config.num_layers)]
        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.box_head = Sam3DecoderMLP(hidden_size, hidden_size, 4, 3)
        self.query_embed = nn.Embedding(config.num_queries, hidden_size)
        self.reference_points = nn.Embedding(config.num_queries, 4)
        self.presence_token = nn.Embedding(1, hidden_size)
        self.presence_head = Sam3DecoderMLP(hidden_size, hidden_size, 1, 3)
        self.presence_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.clamp_presence_logit_max_val = 10.0
        self.ref_point_head = Sam3DecoderMLP(2 * hidden_size, hidden_size, hidden_size, 2)
        self.box_rpb_embed_x = Sam3DecoderMLP(2, hidden_size, config.num_attention_heads, 2)
        self.box_rpb_embed_y = Sam3DecoderMLP(2, hidden_size, config.num_attention_heads, 2)
        self.position_encoding = Sam3SinePositionEmbedding(hidden_size // 2, normalize=False)

    def _get_rpb_matrix(self, reference_boxes: mx.array, height: int, width: int) -> mx.array:
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        batch_size, num_queries, _ = boxes_xyxy.shape

        coords_h = mx.arange(height).astype(mx.float32) / height
        coords_w = mx.arange(width).astype(mx.float32) / width

        flat = boxes_xyxy.reshape(batch_size * num_queries, 1, 4)
        deltas_y = (coords_h.reshape(1, height, 1) - flat[:, :, 1:4:2]).reshape(batch_size, num_queries, height, 2)
        deltas_x = (coords_w.reshape(1, width, 1) - flat[:, :, 0:3:2]).reshape(batch_size, num_queries, width, 2)

        log8 = math.log(8.0)

        def _log_scale(deltas: mx.array) -> mx.array:
            scaled = deltas * 8.0
            return mx.sign(scaled) * mx.log(mx.abs(scaled) + 1.0) / log8

        deltas_x = self.box_rpb_embed_x(_log_scale(deltas_x))  # [B, Q, W, heads]
        deltas_y = self.box_rpb_embed_y(_log_scale(deltas_y))  # [B, Q, H, heads]

        rpb = deltas_y[:, :, :, None, :] + deltas_x[:, :, None, :, :]  # [B, Q, H, W, heads]
        rpb = rpb.reshape(batch_size, num_queries, height * width, -1).transpose(0, 3, 1, 2)
        return mx.pad(rpb, [(0, 0), (0, 0), (1, 0), (0, 0)])  # prepend presence row -> [B, heads, Q+1, H*W]

    def __call__(
        self,
        vision_features: mx.array,
        text_features: mx.array,
        vision_pos_encoding: mx.array,
        text_cross_attn_mask: mx.array | None = None,
        spatial_shapes: list[tuple[int, int]] | None = None,
    ) -> Sam3DetrDecoderOutput:
        batch_size = vision_features.shape[0]

        query_embeds = mx.broadcast_to(self.query_embed.weight[None], (batch_size, *self.query_embed.weight.shape))
        reference_boxes = mx.sigmoid(
            mx.broadcast_to(self.reference_points.weight[None], (batch_size, *self.reference_points.weight.shape))
        )
        presence_token = mx.broadcast_to(
            self.presence_token.weight[None], (batch_size, *self.presence_token.weight.shape)
        )
        hidden_states = mx.concatenate([presence_token, query_embeds], axis=1)

        intermediate_outputs: list[mx.array] = []
        intermediate_boxes: list[mx.array] = [reference_boxes]
        intermediate_presence: list[mx.array] = []

        single_level = spatial_shapes is not None and len(spatial_shapes) == 1

        for layer in self.layers:
            query_sine_embed = self.position_encoding.encode_boxes(reference_boxes)
            query_pos = self.ref_point_head(query_sine_embed)

            vision_cross_attn_mask = None
            if single_level:
                height, width = spatial_shapes[0]
                vision_cross_attn_mask = self._get_rpb_matrix(reference_boxes, height, width)

            hidden_states = layer(
                hidden_states,
                query_pos=query_pos,
                text_features=text_features,
                vision_features=vision_features,
                vision_pos_encoding=vision_pos_encoding,
                text_cross_attn_mask=text_cross_attn_mask,
                vision_cross_attn_mask=vision_cross_attn_mask,
            )

            query_hidden_states = hidden_states[:, 1:]
            normed = self.output_layer_norm(query_hidden_states)

            reference_boxes_before = inverse_sigmoid(reference_boxes)
            delta_boxes = self.box_head(normed)
            new_reference_boxes = mx.sigmoid(delta_boxes + reference_boxes_before)
            reference_boxes = new_reference_boxes

            intermediate_outputs.append(normed)
            intermediate_boxes.append(new_reference_boxes)

            presence_hidden = hidden_states[:, :1]
            presence_logits = self.presence_head(self.presence_layer_norm(presence_hidden))[..., 0]
            presence_logits = mx.clip(
                presence_logits, -self.clamp_presence_logit_max_val, self.clamp_presence_logit_max_val
            )
            intermediate_presence.append(presence_logits)

        return Sam3DetrDecoderOutput(
            intermediate_hidden_states=mx.stack(intermediate_outputs),
            reference_boxes=mx.stack(intermediate_boxes[:-1]),
            presence_logits=mx.stack(intermediate_presence),
        )
