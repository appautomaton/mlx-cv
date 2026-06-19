"""Faithful MLX port of the SAM 3 video tracker — memory encoder + neck (slice 8).

Mirrors ``transformers.models.sam3_tracker_video.modeling_sam3_tracker_video`` so the
real ``facebook/sam3`` ``tracker_model.memory_encoder.*`` (40) tensors load 1:1; the
``tracker_neck`` (22) reuses the detector's :class:`Sam3VisionNeck` (it is the same
``Sam3VisionNeck`` class on ``detector_config.vision_config`` upstream).

SAM2-style memory encoder: a mask down-sampler (progressive stride-2 convs), a
1x1 feature projection, a memory fuser of depthwise-conv "CX" blocks, and a final
channel projection, plus a parameter-free sine position encoding. Channels-last
(NHWC) throughout (channels-first LayerNorm over C == ``nn.LayerNorm`` on the last
axis). No torch/transformers imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .real_video_config import Sam3TrackerVideoConfig
from .real_vision import Sam3SinePositionEmbedding

__all__ = [
    "Sam3TrackerMemoryFuserCXBlock",
    "Sam3TrackerMemoryFuser",
    "Sam3TrackerMaskDownSampler",
    "Sam3TrackerMemoryEncoder",
    "Sam3TrackerMemoryEncoderOutput",
]


class Sam3TrackerMemoryFuserCXBlock(nn.Module):
    """ConvNeXt-style block: depthwise conv -> LN -> 1x1 (Linear) MLP -> layer scale."""

    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        dim = config.memory_fuser_embed_dim
        self.depthwise_conv = nn.Conv2d(
            dim, dim, kernel_size=config.memory_fuser_kernel_size, padding=config.memory_fuser_padding, groups=dim
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, config.memory_fuser_intermediate_dim)
        self.pointwise_conv2 = nn.Linear(config.memory_fuser_intermediate_dim, dim)
        self.scale = config.memory_fuser_layer_scale_init_value * mx.ones((dim,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # NHWC throughout (upstream permutes NCHW<->NHWC around the pointwise convs).
        residual = hidden_states
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.scale * hidden_states
        return residual + hidden_states


class Sam3TrackerMemoryFuser(nn.Module):
    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        self.layers = [Sam3TrackerMemoryFuserCXBlock(config) for _ in range(config.memory_fuser_num_layers)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _MaskDownSamplerLayer(nn.Module):
    def __init__(self, config: Sam3TrackerVideoConfig, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=config.mask_downsampler_kernel_size,
            stride=config.mask_downsampler_stride,
            padding=config.mask_downsampler_padding,
        )
        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.layer_norm(self.conv(x)))


class Sam3TrackerMaskDownSampler(nn.Module):
    """Progressively downsample a single-channel mask by ``total_stride`` (NHWC)."""

    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        num_layers = int(
            math.log2(config.mask_downsampler_total_stride) // math.log2(config.mask_downsampler_stride)
        )
        layers: list[nn.Module] = []
        in_chans, out_chans = 1, 1
        for _ in range(num_layers):
            out_chans = in_chans * (config.mask_downsampler_stride**2)
            layers.append(_MaskDownSamplerLayer(config, in_chans, out_chans))
            in_chans = out_chans
        self.layers = layers
        self.final_conv = nn.Conv2d(out_chans, config.mask_downsampler_embed_dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


@dataclass
class Sam3TrackerMemoryEncoderOutput:
    vision_features: mx.array  # [B, H, W, output_channels]
    vision_pos_enc: mx.array  # [B, H, W, output_channels]


class Sam3TrackerMemoryEncoder(nn.Module):
    """Fuse pixel features with a downsampled mask into a compact memory (40 tensors)."""

    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        hidden_size = config.memory_encoder_hidden_size
        output_channels = config.memory_encoder_output_channels
        self.mask_downsampler = Sam3TrackerMaskDownSampler(config)
        self.feature_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.memory_fuser = Sam3TrackerMemoryFuser(config)
        self.position_encoding = Sam3SinePositionEmbedding(output_channels // 2, normalize=True)
        self.projection = nn.Conv2d(hidden_size, output_channels, kernel_size=1)

    def __call__(self, vision_features: mx.array, masks: mx.array) -> Sam3TrackerMemoryEncoderOutput:
        # vision_features: NHWC [B, H, W, hidden]; masks: NHWC [B, H*16, W*16, 1].
        masks = self.mask_downsampler(masks)
        vision_features = self.feature_projection(vision_features)
        vision_features = vision_features + masks
        vision_features = self.memory_fuser(vision_features)
        vision_features = self.projection(vision_features)

        batch_size, height, width, _ = vision_features.shape
        vision_pos_enc = self.position_encoding(batch_size, height, width)
        return Sam3TrackerMemoryEncoderOutput(vision_features=vision_features, vision_pos_enc=vision_pos_enc)
