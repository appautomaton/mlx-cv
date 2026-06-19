"""Faithful MLX port of the SAM 3 mask decoder (``Sam3MaskDecoder``).

Mirrors ``transformers.models.sam3.modeling_sam3`` so the real ``facebook/sam3``
``detector_model.mask_decoder.*`` tensors load 1:1 (32 keys): an FPN pixel decoder
(top-down, nearest upsample, Conv2d + GroupNorm(8) + ReLU), a 3-layer mask embedder,
a prompt cross-attention refinement of the encoder features, and instance/semantic
Conv2d 1x1 heads. Channels-last (NHWC) throughout. No torch/transformers imports.

Taps (PLAN slice 6): ``pred_masks`` ``[B, Q, H, W]`` and ``semantic_seg``
``[B, 1, H, W]``. This completes the image detector subsystems (SPEC AC3).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .real_config import Sam3MaskDecoderConfig
from .real_detr import Sam3Attention

__all__ = ["Sam3MaskEmbedder", "Sam3PixelDecoder", "Sam3MaskDecoder", "Sam3MaskDecoderOutput"]


def _nearest_resize(x: mx.array, out_height: int, out_width: int) -> mx.array:
    """Nearest-neighbour resize of NHWC ``x`` to ``(out_height, out_width)``.

    Matches ``F.interpolate(mode="nearest")``: ``src = floor(dst * in / out)``.
    """

    _, height, width, _ = x.shape
    ys = (mx.arange(out_height) * height) // out_height
    xs = (mx.arange(out_width) * width) // out_width
    return x[:, ys][:, :, xs]


class Sam3MaskEmbedder(nn.Module):
    """3-layer MLP embedding decoder queries for mask prediction (ReLU between)."""

    def __init__(self, config: Sam3MaskDecoderConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(3)]

    def __call__(self, queries: mx.array) -> mx.array:
        hidden_states = queries
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if i < len(self.layers) - 1:
                hidden_states = nn.relu(hidden_states)
        return hidden_states


class Sam3PixelDecoder(nn.Module):
    """FPN pixel decoder: top-down nearest upsample + Conv2d + GroupNorm(8) + ReLU."""

    def __init__(self, config: Sam3MaskDecoderConfig):
        super().__init__()
        hidden_size = config.hidden_size
        stages = config.num_upsampling_stages
        self.conv_layers = [nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1) for _ in range(stages)]
        self.norms = [nn.GroupNorm(8, hidden_size, pytorch_compatible=True) for _ in range(stages)]
        self.out_channels = hidden_size

    def __call__(self, backbone_features: list[mx.array]) -> mx.array:
        # Each feature is NHWC [B, H_i, W_i, C], ordered low -> high resolution.
        prev_fpn = backbone_features[-1]
        for layer_idx, backbone_feat in enumerate(reversed(backbone_features[:-1])):
            prev_fpn = _nearest_resize(prev_fpn, backbone_feat.shape[1], backbone_feat.shape[2])
            prev_fpn = prev_fpn + backbone_feat
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = self.norms[layer_idx](prev_fpn)
            prev_fpn = nn.relu(prev_fpn)
        return prev_fpn


@dataclass
class Sam3MaskDecoderOutput:
    pred_masks: mx.array  # [B, Q, H, W]
    semantic_seg: mx.array  # [B, 1, H, W]


class Sam3MaskDecoder(nn.Module):
    """Combine decoder queries with pixel features to predict instance masks (32 tensors)."""

    def __init__(self, config: Sam3MaskDecoderConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.pixel_decoder = Sam3PixelDecoder(config)
        self.mask_embedder = Sam3MaskEmbedder(config)
        self.instance_projection = nn.Conv2d(self.pixel_decoder.out_channels, hidden_size, kernel_size=1)
        self.semantic_projection = nn.Conv2d(self.pixel_decoder.out_channels, 1, kernel_size=1)
        self.prompt_cross_attn = Sam3Attention(hidden_size, config.num_attention_heads)
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def _embed_pixels(self, backbone_features: list[mx.array], encoder_hidden_states: mx.array) -> mx.array:
        backbone_visual_feats = list(backbone_features)
        height, width = backbone_features[-1].shape[1], backbone_features[-1].shape[2]
        spatial_dim = height * width
        batch_size = encoder_hidden_states.shape[0]
        hidden_size = encoder_hidden_states.shape[-1]
        # encoder visual tokens (h*W + w order) reshape directly to NHWC.
        encoder_visual = encoder_hidden_states[:, :spatial_dim, :].reshape(batch_size, height, width, hidden_size)
        backbone_visual_feats[-1] = encoder_visual
        return self.pixel_decoder(backbone_visual_feats)

    def __call__(
        self,
        decoder_queries: mx.array,
        backbone_features: list[mx.array],
        encoder_hidden_states: mx.array,
        prompt_features: mx.array | None = None,
        prompt_cross_attn_mask: mx.array | None = None,
    ) -> Sam3MaskDecoderOutput:
        if prompt_features is not None:
            residual = encoder_hidden_states
            normed = self.prompt_cross_attn_norm(encoder_hidden_states)
            attn = self.prompt_cross_attn(normed, prompt_features, prompt_features, prompt_cross_attn_mask)
            encoder_hidden_states = residual + attn

        pixel_embed = self._embed_pixels(backbone_features, encoder_hidden_states)  # [B, H, W, C]

        instance_embeds = self.instance_projection(pixel_embed)  # [B, H, W, C]
        mask_embeddings = self.mask_embedder(decoder_queries)  # [B, Q, C]

        batch_size, height, width, channels = instance_embeds.shape
        instance_flat = instance_embeds.reshape(batch_size, height * width, channels)
        pred_masks = mask_embeddings @ instance_flat.transpose(0, 2, 1)  # [B, Q, H*W]
        pred_masks = pred_masks.reshape(batch_size, -1, height, width)

        semantic_seg = self.semantic_projection(pixel_embed).transpose(0, 3, 1, 2)  # [B, 1, H, W]

        return Sam3MaskDecoderOutput(pred_masks=pred_masks, semantic_seg=semantic_seg)
