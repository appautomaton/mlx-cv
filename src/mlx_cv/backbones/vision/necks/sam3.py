"""SAM 3.1 image-mode feature neck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from ....core.features import BackboneFeatures, FeatureMap, Layout

__all__ = ["SAM3FeatureNeck", "SAM3FeaturePyramid", "SAM3PyramidLevel"]


@dataclass
class SAM3PyramidLevel:
    data: mx.array
    feature: FeatureMap
    mask: mx.array
    position: mx.array
    stride: int


@dataclass
class SAM3FeaturePyramid:
    levels: list[SAM3PyramidLevel]

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
    if feature.layout is Layout.BHWC:
        return feature.data
    if feature.layout is Layout.BNC:
        if feature.grid is None:
            raise ValueError("BNC feature requires grid metadata to reshape into BHWC")
        b, n, c = feature.data.shape
        h, w = feature.grid
        if n != h * w:
            raise ValueError(f"BNC token count {n} does not match grid {feature.grid}")
        return feature.data.reshape(b, h, w, c)
    raise ValueError(f"SAM3 neck expects BHWC or BNC features, got {feature.layout}")


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


class SAM3FeatureNeck(nn.Module):
    """Project SAM3 image/VL features into a mask-decoder pyramid."""

    def __init__(
        self,
        *,
        in_channels: Sequence[int],
        out_channels: int,
        scale_factors: Sequence[float] = (1.0, 0.5, 0.25),
    ) -> None:
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature level")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        self.in_channels = tuple(int(c) for c in in_channels)
        self.out_channels = int(out_channels)
        self.scale_factors = tuple(float(scale) for scale in scale_factors)
        if any(scale <= 0 for scale in self.scale_factors):
            raise ValueError("scale_factors must be positive")
        fused_channels = sum(self.in_channels)
        self.projections = [nn.Conv2d(fused_channels, self.out_channels, 1) for _ in self.scale_factors]

    def __call__(self, features: BackboneFeatures | Sequence[FeatureMap]) -> SAM3FeaturePyramid:
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
            raise ValueError("SAM3 neck requires grid and stride metadata")
        for i, (feature, expected_channels, fmap) in enumerate(zip(input_features, self.in_channels, maps)):
            if fmap.shape[-1] != expected_channels:
                raise ValueError(f"feature {i} has {fmap.shape[-1]} channels, expected {expected_channels}")
            if feature.grid is None or feature.stride is None:
                raise ValueError("all SAM3 neck inputs require grid and stride metadata")
            if fmap.shape[0] != batch:
                raise ValueError("all SAM3 neck inputs must share batch size")

        levels: list[SAM3PyramidLevel] = []
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
            position = _position_grid(batch, target_h, target_w)
            levels.append(SAM3PyramidLevel(projected, feature, mask, position, stride))
        return SAM3FeaturePyramid(levels)
