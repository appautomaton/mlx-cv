"""Faithful MLX port of the SAM 3 geometry (box-prompt) encoder.

Mirrors ``transformers.models.sam3.modeling_sam3.Sam3GeometryEncoder`` so the real
``detector_model.geometry_encoder.*`` tensors load 1:1 (94 keys). The 3 transformer
layers (prompt self-attention + cross-attention to vision + ReLU MLP) and all box
projection/embedding parameters are ported faithfully.

Scope note (SPEC-verified): ``Sam3Model.forward`` invokes the geometry encoder only
when ``input_boxes`` are given (``has_geometry_prompts``); the text-prompt detection
path that the image parity gate exercises uses ``combined_prompt_features =
text_features`` and never calls it (modeling_sam3.py ~L2331). The box-prompt forward
needs a ``roi_align`` pooling step; rather than ship an unverifiable numeric kernel
(no committed gate exercises it, no torchvision in-sandbox to check against), the
box-prompt entry point raises a precise error. The 94 tensors still load 1:1 and the
transformer layer is exercised directly. See PLAN.md (slice 4).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .real_config import Sam3GeometryEncoderConfig
from .real_detr import Sam3Attention, Sam3FFN
from .real_vision import Sam3SinePositionEmbedding

__all__ = ["Sam3GeometryEncoderLayer", "Sam3GeometryEncoder"]


class Sam3GeometryEncoderLayer(nn.Module):
    """Prompt self-attention + cross-attention to vision + ReLU MLP (pre-norm)."""

    def __init__(self, config: Sam3GeometryEncoderConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Sam3Attention(config.hidden_size, config.num_attention_heads)
        self.cross_attn = Sam3Attention(config.hidden_size, config.num_attention_heads)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Sam3FFN(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        prompt_feats: mx.array,
        vision_feats: mx.array,
        vision_pos_encoding: mx.array,
        prompt_mask: mx.array | None = None,
    ) -> mx.array:
        residual = prompt_feats
        hidden_states = self.layer_norm1(prompt_feats)
        hidden_states = self.self_attn(hidden_states, hidden_states, hidden_states, prompt_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        key = vision_feats + vision_pos_encoding
        hidden_states = self.cross_attn(hidden_states, key, vision_feats)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Sam3GeometryEncoder(nn.Module):
    """Encoder for geometric (box) prompts (94 tensors)."""

    def __init__(self, config: Sam3GeometryEncoderConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        self.position_encoding = Sam3SinePositionEmbedding(hidden_size // 2, normalize=True)
        self.label_embed = nn.Embedding(2, hidden_size)
        self.cls_embed = nn.Embedding(1, hidden_size)

        self.boxes_direct_project = nn.Linear(4, hidden_size)
        self.boxes_pool_project = nn.Conv2d(hidden_size, hidden_size, config.roi_size)
        self.boxes_pos_enc_project = nn.Linear(hidden_size + 2, hidden_size)

        self.vision_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.final_proj = nn.Linear(hidden_size, hidden_size)
        self.prompt_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.layers = [Sam3GeometryEncoderLayer(config) for _ in range(config.num_layers)]
        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def run_layers(
        self,
        prompt_feats: mx.array,
        vision_feats: mx.array,
        vision_pos_encoding: mx.array,
        prompt_mask: mx.array | None = None,
    ) -> mx.array:
        """Run the prompt transformer stack + output norm (no box encoding)."""

        prompt_feats = self.prompt_layer_norm(self.final_proj(prompt_feats))
        for layer in self.layers:
            prompt_feats = layer(prompt_feats, vision_feats, vision_pos_encoding, prompt_mask)
        return self.output_layer_norm(prompt_feats)

    def __call__(self, *args, **kwargs):  # noqa: D102 - faithful box-prompt entry point
        raise NotImplementedError(
            "SAM3 geometry box-prompt forward requires a roi_align pooling step that is "
            "not yet ported. The text-prompt detection path does not use the geometry "
            "encoder (Sam3Model.forward only calls it when input_boxes are given); the 94 "
            "geometry tensors still load 1:1 and Sam3GeometryEncoder.run_layers exercises "
            "the transformer stack. roi_align will be added + verified against torchvision "
            "when a box-prompt parity gate is introduced."
        )
