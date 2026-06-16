"""RF-DETR multi-scale feature projector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from ....core.features import BackboneFeatures, FeatureMap, Layout

__all__ = [
    "RFDETRPyramidLevel",
    "RFDETRFeaturePyramid",
    "RFDETRMultiScaleProjector",
    "RFDETRP4C2fProjector",
]


@dataclass
class RFDETRPyramidLevel:
    data: mx.array
    feature: FeatureMap
    mask: mx.array
    position: mx.array
    stride: int


@dataclass
class RFDETRFeaturePyramid:
    levels: list[RFDETRPyramidLevel]

    @property
    def features(self) -> list[FeatureMap]:
        return [level.feature for level in self.levels]

    @property
    def masks(self) -> list[mx.array]:
        return [level.mask for level in self.levels]

    @property
    def positions(self) -> list[mx.array]:
        return [level.position for level in self.levels]


def _feature_to_bhwc(feature: FeatureMap) -> mx.array:
    if feature.layout == Layout.BHWC:
        return feature.data
    if feature.layout == Layout.BNC:
        if feature.grid is None:
            raise ValueError("BNC feature requires grid metadata to reshape into BHWC")
        b, n, c = feature.data.shape
        h, w = feature.grid
        if n != h * w:
            raise ValueError(f"BNC token count {n} does not match grid {feature.grid}")
        return feature.data.reshape(b, h, w, c)
    raise ValueError(f"RF-DETR projector expects BHWC or BNC features, got {feature.layout}")


def _resize_axis_nearest(x: mx.array, out_size: int, axis: int) -> mx.array:
    in_size = x.shape[axis]
    if out_size <= 0:
        raise ValueError(f"resize output size must be positive, got {out_size}")
    if in_size == out_size:
        return x
    coords = mx.arange(out_size, dtype=mx.float32) * (in_size / out_size)
    idx = mx.minimum(mx.floor(coords).astype(mx.int32), in_size - 1)
    return mx.take(x, idx, axis=axis)


def _resize_bhwc_nearest(x: mx.array, size: tuple[int, int]) -> mx.array:
    return _resize_axis_nearest(_resize_axis_nearest(x, int(size[0]), axis=1), int(size[1]), axis=2)


def _position_grid(batch: int, height: int, width: int) -> mx.array:
    y = (mx.arange(height, dtype=mx.float32) + 0.5) / height
    x = (mx.arange(width, dtype=mx.float32) + 0.5) / width
    yy = mx.broadcast_to(y[:, None], (height, width))
    xx = mx.broadcast_to(x[None, :], (height, width))
    grid = mx.stack([xx, yy], axis=-1)
    return mx.broadcast_to(grid[None, ...], (batch, height, width, 2))


def _activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name.lower() in {"leakyrelu", "lrelu"}:
        return nn.LeakyReLU(0.1)
    raise ValueError(f"unsupported RF-DETR projector activation: {name!r}")


class _RFDETRConvX(nn.Module):
    """Upstream ConvX in BHWC layout, preserving ``conv``/``bn`` parameter paths."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int | tuple[int, int] = 3,
        stride: int = 1,
        *,
        groups: int = 1,
        act: str | None = "relu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.LayerNorm(out_channels, eps=1e-6) if layer_norm else nn.BatchNorm(out_channels)
        self.act = _activation(act)

    def __call__(self, x: mx.array) -> mx.array:
        return self.act(self.bn(self.conv(x)))


class _RFDETRBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        act: str = "silu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = _RFDETRConvX(in_channels, hidden_channels, 3, act=act, layer_norm=layer_norm)
        self.cv2 = _RFDETRConvX(
            hidden_channels,
            out_channels,
            3,
            groups=groups,
            act=act,
            layer_norm=layer_norm,
        )
        self.add = bool(shortcut and in_channels == out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class _RFDETRC2f(nn.Module):
    """P4 C2f stage used by upstream RF-DETR Nano checkpoints."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_blocks: int = 3,
        shortcut: bool = False,
        groups: int = 1,
        expansion: float = 0.5,
        act: str = "silu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.c = int(out_channels * expansion)
        self.cv1 = _RFDETRConvX(in_channels, 2 * self.c, 1, act=act, layer_norm=layer_norm)
        self.cv2 = _RFDETRConvX((2 + num_blocks) * self.c, out_channels, 1, act=act, layer_norm=layer_norm)
        self.m = [
            _RFDETRBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                groups=groups,
                expansion=1.0,
                act=act,
                layer_norm=layer_norm,
            )
            for _ in range(num_blocks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        y = self.cv1(x)
        parts = [y[..., : self.c], y[..., self.c : 2 * self.c]]
        for block in self.m:
            parts.append(block(parts[-1]))
        return self.cv2(mx.concatenate(parts, axis=-1))


class RFDETRMultiScaleProjector(nn.Module):
    """Project DINOv2 patch features into an RF-DETR-style multi-scale pyramid."""

    def __init__(
        self,
        *,
        in_channels: Sequence[int],
        out_channels: int,
        scale_factors: Sequence[float] = (2.0, 1.0, 0.5),
    ) -> None:
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature level")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        self.in_channels = tuple(int(c) for c in in_channels)
        self.out_channels = int(out_channels)
        self.scale_factors = tuple(float(s) for s in scale_factors)
        if any(s <= 0 for s in self.scale_factors):
            raise ValueError("scale_factors must be positive")
        fused_channels = sum(self.in_channels)
        self.projections = [nn.Conv2d(fused_channels, self.out_channels, 1) for _ in self.scale_factors]

    def __call__(self, features: BackboneFeatures | Sequence[FeatureMap]) -> RFDETRFeaturePyramid:
        if isinstance(features, BackboneFeatures):
            input_features = features.intermediates or [features.patch_tokens]
        else:
            input_features = list(features)
        if len(input_features) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} features, got {len(input_features)}")

        maps = [_feature_to_bhwc(feature) for feature in input_features]
        batch = maps[0].shape[0]
        base_grid = input_features[-1].grid
        base_stride = input_features[-1].stride
        if base_grid is None or base_stride is None:
            raise ValueError("RF-DETR projector requires grid and stride metadata")
        for i, (feature, expected_channels, fmap) in enumerate(zip(input_features, self.in_channels, maps)):
            if fmap.shape[-1] != expected_channels:
                raise ValueError(
                    f"feature {i} has {fmap.shape[-1]} channels, expected {expected_channels}"
                )
            if feature.grid is None or feature.stride is None:
                raise ValueError("all RF-DETR projector inputs require grid and stride metadata")
            if fmap.shape[0] != batch:
                raise ValueError("all RF-DETR projector inputs must share batch size")

        levels: list[RFDETRPyramidLevel] = []
        base_h, base_w = base_grid
        for scale, projection in zip(self.scale_factors, self.projections):
            target_h = max(1, int(round(base_h * scale)))
            target_w = max(1, int(round(base_w * scale)))
            resized = [_resize_bhwc_nearest(fmap, (target_h, target_w)) for fmap in maps]
            fused = mx.concatenate(resized, axis=-1)
            projected = projection(fused)
            stride = max(1, int(round(base_stride / scale)))
            feature = FeatureMap(
                projected,
                layout=Layout.BHWC,
                grid=(target_h, target_w),
                stride=stride,
            )
            mask = mx.zeros((batch, target_h, target_w), dtype=mx.bool_)
            pos = _position_grid(batch, target_h, target_w)
            levels.append(
                RFDETRPyramidLevel(
                    data=projected,
                    feature=feature,
                    mask=mask,
                    position=pos,
                    stride=stride,
                )
            )
        return RFDETRFeaturePyramid(levels)


class RFDETRP4C2fProjector(nn.Module):
    """RF-DETR Nano P4-only projector over four DINOv2 feature maps.

    This mirrors upstream ``projector_scale=['P4']`` for inference: no sampling
    modules are applied, the four same-grid DINO maps are concatenated, and a
    single C2f stage produces one 256-channel feature level. Training-time
    stochastic feature dropping remains outside this local inference path.
    """

    def __init__(
        self,
        *,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int = 3,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"RF-DETR Nano P4 projector expects four input features, got {len(in_channels)}")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        self.in_channels = tuple(int(c) for c in in_channels)
        self.out_channels = int(out_channels)
        self.scale_factors = (1.0,)
        self.layer_norm = bool(layer_norm)
        fused_channels = sum(self.in_channels)
        self.stages = [
            [
                _RFDETRC2f(
                    fused_channels,
                    self.out_channels,
                    num_blocks=num_blocks,
                    layer_norm=self.layer_norm,
                ),
                nn.LayerNorm(self.out_channels, eps=1e-6),
            ]
        ]
        self.inference_exclusions = ("training_feature_drop",)

    def __call__(self, features: BackboneFeatures | Sequence[FeatureMap]) -> RFDETRFeaturePyramid:
        if isinstance(features, BackboneFeatures):
            input_features = features.intermediates or [features.patch_tokens]
        else:
            input_features = list(features)
        if len(input_features) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} features, got {len(input_features)}")

        maps = [_feature_to_bhwc(feature) for feature in input_features]
        batch = maps[0].shape[0]
        base_grid = input_features[-1].grid
        base_stride = input_features[-1].stride
        if base_grid is None or base_stride is None:
            raise ValueError("RF-DETR P4 projector requires grid and stride metadata")
        for i, (feature, expected_channels, fmap) in enumerate(zip(input_features, self.in_channels, maps)):
            if fmap.shape[-1] != expected_channels:
                raise ValueError(f"feature {i} has {fmap.shape[-1]} channels, expected {expected_channels}")
            if feature.grid != base_grid or feature.stride != base_stride:
                raise ValueError("RF-DETR P4 projector inputs must share grid and stride metadata")
            if fmap.shape[0] != batch:
                raise ValueError("RF-DETR P4 projector inputs must share batch size")

        projected = mx.concatenate(maps, axis=-1)
        for layer in self.stages[0]:
            projected = layer(projected)
        height, width = base_grid
        feature = FeatureMap(projected, layout=Layout.BHWC, grid=base_grid, stride=base_stride)
        level = RFDETRPyramidLevel(
            data=projected,
            feature=feature,
            mask=mx.zeros((batch, height, width), dtype=mx.bool_),
            position=_position_grid(batch, height, width),
            stride=base_stride,
        )
        return RFDETRFeaturePyramid([level])
