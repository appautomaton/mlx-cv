"""SAM3 video memory encoder modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import SAM3VideoMemoryConfig
from .multiplex_state import SAM3MultiplexState

__all__ = [
    "SAM3MaskDownSampler",
    "SAM3MemoryCXBlock",
    "SAM3MemoryEncoder",
    "SAM3MemoryEncoderOutput",
    "SAM3MemoryFuser",
    "SAM3MemoryMaskInput",
    "bucket_features_to_object_space",
    "build_multiplex_memory_mask_input",
    "mask_logits_for_memory",
]


def _sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-x))


def _nchw_to_nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _nhwc_to_nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


def _conv_nchw(conv: nn.Module, x: mx.array) -> mx.array:
    return _nhwc_to_nchw(conv(_nchw_to_nhwc(x)))


def _layer_norm_nchw(norm: nn.LayerNorm, x: mx.array) -> mx.array:
    return _nhwc_to_nchw(norm(_nchw_to_nhwc(x)))


def _resize_axis_nearest(x: mx.array, out_size: int, axis: int) -> mx.array:
    in_size = int(x.shape[axis])
    out_size = int(out_size)
    if out_size <= 0:
        raise ValueError("resize output size must be positive")
    if in_size == out_size:
        return x
    coords = mx.arange(out_size, dtype=mx.float32) * (in_size / out_size)
    idx = mx.minimum(mx.floor(coords).astype(mx.int32), in_size - 1)
    return mx.take(x, idx, axis=axis)


def _resize_nchw_nearest(x: mx.array, size: tuple[int, int]) -> mx.array:
    return _resize_axis_nearest(_resize_axis_nearest(x, int(size[0]), axis=2), int(size[1]), axis=3)


def _position_encoding_nchw(batch: int, channels: int, height: int, width: int, dtype) -> mx.array:
    y = (mx.arange(height, dtype=mx.float32) + 0.5) / height
    x = (mx.arange(width, dtype=mx.float32) + 0.5) / width
    yy = mx.broadcast_to(y[:, None], (height, width))
    xx = mx.broadcast_to(x[None, :], (height, width))
    base = mx.stack([xx, yy], axis=0)
    repeats = math.ceil(channels / 2)
    tiled = mx.concatenate([base] * repeats, axis=0)[:channels]
    return mx.broadcast_to(tiled[None, ...].astype(dtype), (batch, channels, height, width))


@dataclass
class SAM3MemoryMaskInput:
    mask_for_mem_object_space: mx.array
    mask_for_mem_mux_space: mx.array
    condition_mask_channels: mx.array | None
    encoder_input_channels: mx.array


@dataclass
class SAM3MemoryEncoderOutput:
    features: mx.array
    pos_enc: mx.array
    mask_channels: mx.array
    taps: dict[str, mx.array]

    def __getitem__(self, key: str) -> mx.array:
        if key == "vision_features":
            return self.features
        if key == "vision_pos_enc":
            return self.pos_enc
        if key == "mask_channels":
            return self.mask_channels
        raise KeyError(key)


def mask_logits_for_memory(
    pred_masks_high_res: mx.array,
    *,
    apply_sigmoid: bool = True,
    scale: float = 2.0,
    bias: float = -1.0,
) -> mx.array:
    if apply_sigmoid:
        pred_masks_high_res = _sigmoid(pred_masks_high_res)
    if scale != 1.0:
        pred_masks_high_res = pred_masks_high_res * scale
    if bias != 0.0:
        pred_masks_high_res = pred_masks_high_res + bias
    return pred_masks_high_res


def build_multiplex_memory_mask_input(
    mask_for_mem: mx.array,
    multiplex_state: SAM3MultiplexState,
    *,
    condition_as_mask_input: bool,
    conditioning_objects: Iterable[int] | None = None,
    condition_fg: float = 1.0,
    condition_bg: float = 0.0,
) -> SAM3MemoryMaskInput:
    if len(mask_for_mem.shape) != 4 or int(mask_for_mem.shape[1]) != 1:
        raise ValueError(f"SAM3 memory mask input expects (O,1,H,W), got {tuple(mask_for_mem.shape)}")
    muxed = multiplex_state.mux(mask_for_mem)[:, :, 0, :, :]
    condition_channels = None
    encoder_input = muxed
    if condition_as_mask_input:
        num_objects = int(mask_for_mem.shape[0])
        cond = np.full(tuple(mask_for_mem.shape), float(condition_bg), dtype=np.float32)
        for object_idx in sorted(set(int(v) for v in (conditioning_objects or ()))):
            if object_idx < 0 or object_idx >= num_objects:
                raise ValueError(f"SAM3 conditioning object index out of range: {object_idx}")
            cond[object_idx, :, :, :] = float(condition_fg)
        condition_channels = multiplex_state.mux(mx.array(cond, dtype=mask_for_mem.dtype))[:, :, 0, :, :]
        encoder_input = mx.concatenate([muxed, condition_channels], axis=1)
    return SAM3MemoryMaskInput(
        mask_for_mem_object_space=mask_for_mem,
        mask_for_mem_mux_space=muxed,
        condition_mask_channels=condition_channels,
        encoder_input_channels=encoder_input,
    )


def bucket_features_to_object_space(features: mx.array, multiplex_state: SAM3MultiplexState) -> mx.array:
    if len(features.shape) != 4:
        raise ValueError(f"SAM3 bucket features expect (B,C,H,W), got {tuple(features.shape)}")
    expanded = mx.broadcast_to(
        features[:, None, :, :, :],
        (
            multiplex_state.num_buckets,
            multiplex_state.multiplex_count,
            int(features.shape[1]),
            int(features.shape[2]),
            int(features.shape[3]),
        ),
    )
    return multiplex_state.demux(expanded)


class SAM3MaskDownSampler(nn.Module):
    """Progressively downsample muxed mask channels to the tracker feature grid."""

    def __init__(self, cfg: SAM3VideoMemoryConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers = 0
        stride_product = 1
        while stride_product < cfg.mask_total_stride:
            stride_product *= cfg.mask_downsample_stride
            layers += 1
        if stride_product != cfg.mask_total_stride:
            raise ValueError("SAM3 mask downsample strides must multiply to total_stride")
        self.convs = []
        self.norms = []
        in_channels = cfg.mask_input_channels
        for _ in range(layers):
            out_channels = cfg.hidden_dim
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    cfg.mask_downsample_stride,
                    stride=cfg.mask_downsample_stride,
                )
            )
            self.norms.append(nn.LayerNorm(out_channels))
            in_channels = out_channels

    def __call__(self, x: mx.array) -> mx.array:
        if len(x.shape) != 4:
            raise ValueError(f"SAM3 mask downsampler expects NCHW input, got {tuple(x.shape)}")
        if int(x.shape[1]) != self.cfg.mask_input_channels:
            raise ValueError(
                f"SAM3 memory mask channels {int(x.shape[1])} must equal {self.cfg.mask_input_channels}"
            )
        for conv, norm in zip(self.convs, self.norms):
            x = _conv_nchw(conv, x)
            x = nn.gelu(_layer_norm_nchw(norm, x))
        return x


class SAM3MemoryCXBlock(nn.Module):
    """Compact ConvNeXt-style fuser block using NCHW public layout."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.fc1 = nn.Linear(channels, channels * 4)
        self.fc2 = nn.Linear(channels * 4, channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        y = _conv_nchw(self.conv, x)
        y = _nchw_to_nhwc(y)
        y = self.norm(y)
        y = self.fc2(nn.gelu(self.fc1(y)))
        y = _nhwc_to_nchw(y)
        return residual + y


class SAM3MemoryFuser(nn.Module):
    def __init__(self, channels: int, num_layers: int) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("SAM3 memory fuser requires at least one layer")
        self.layers = [SAM3MemoryCXBlock(channels) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class SAM3MemoryEncoder(nn.Module):
    """Multiplex-aware SAM3 memory encoder.

    Inputs and outputs use NCHW layout. The mask input is already in bucket
    channel space, e.g. ``(Buc, 2*M, 32, 32)`` for conditional tiny fixtures.
    """

    def __init__(self, cfg: SAM3VideoMemoryConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_downsampler = SAM3MaskDownSampler(cfg)
        self.pix_feat_proj = nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, 1)
        self.fuser = SAM3MemoryFuser(cfg.hidden_dim, cfg.fuser_layers)

    def __call__(
        self,
        pix_feat: mx.array,
        mask_channels: mx.array,
        *,
        skip_mask_sigmoid: bool = True,
        capture_taps: bool = False,
    ) -> SAM3MemoryEncoderOutput:
        if len(pix_feat.shape) != 4:
            raise ValueError(f"SAM3 memory pix_feat expects (B,C,H,W), got {tuple(pix_feat.shape)}")
        if int(pix_feat.shape[1]) != self.cfg.hidden_dim:
            raise ValueError(f"SAM3 memory pix_feat channels must equal {self.cfg.hidden_dim}")
        if int(mask_channels.shape[0]) != int(pix_feat.shape[0]):
            raise ValueError("SAM3 memory pix_feat and mask channel batches must match")
        masks = mask_channels if skip_mask_sigmoid else _sigmoid(mask_channels)
        masks = self.mask_downsampler(masks)
        if tuple(masks.shape[2:]) != tuple(pix_feat.shape[2:]):
            masks = _resize_nchw_nearest(masks, (int(pix_feat.shape[2]), int(pix_feat.shape[3])))
        x = _conv_nchw(self.pix_feat_proj, pix_feat) + masks
        x = self.fuser(x)
        pos = _position_encoding_nchw(
            int(x.shape[0]),
            int(x.shape[1]),
            int(x.shape[2]),
            int(x.shape[3]),
            x.dtype,
        )
        taps = {}
        if capture_taps:
            taps["memory.encoder_input_channels"] = mask_channels
            taps["memory.features.bucket_space"] = x
        return SAM3MemoryEncoderOutput(features=x, pos_enc=pos, mask_channels=mask_channels, taps=taps)
