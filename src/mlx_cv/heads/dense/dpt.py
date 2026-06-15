"""Depth Anything 3 DPT dense prediction head in MLX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from ...core.features import HeadInput, HeadOutput, Layout
from ...core.registry import register_head

__all__ = [
    "DPTConfig",
    "DPTHead",
    "build_dpt",
    "resize_bilinear_align_corners",
]


@dataclass(frozen=True)
class DPTConfig:
    dim_in: int
    patch_size: int = 14
    output_dim: int = 1
    activation: str = "exp"
    conf_activation: str = "expp1"
    features: int = 256
    out_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024)
    pos_embed: bool = False
    down_ratio: int = 1
    head_name: str = "depth"
    use_sky_head: bool = False
    norm_type: str = "idt"

    @classmethod
    def from_dict(cls, d: dict) -> "DPTConfig":
        out_channels = d.get("out_channels", cls.out_channels)
        return cls(
            dim_in=d["dim_in"],
            patch_size=d.get("patch_size", 14),
            output_dim=d.get("output_dim", 1),
            activation=d.get("activation", "exp"),
            conf_activation=d.get("conf_activation", "expp1"),
            features=d.get("features", 256),
            out_channels=tuple(out_channels),
            pos_embed=d.get("pos_embed", False),
            down_ratio=d.get("down_ratio", 1),
            head_name=d.get("head_name", "depth"),
            use_sky_head=d.get("use_sky_head", False),
            norm_type=d.get("norm_type", "idt"),
        )


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _resize_axis_align_corners(x: mx.array, out_size: int, axis: int) -> mx.array:
    in_size = x.shape[axis]
    if in_size == out_size:
        return x
    if out_size <= 0:
        raise ValueError(f"resize output size must be positive, got {out_size}")
    if out_size == 1:
        coords = mx.zeros((1,), dtype=mx.float32)
    else:
        coords = mx.arange(out_size, dtype=mx.float32) * ((in_size - 1) / (out_size - 1))
    lo = mx.floor(coords).astype(mx.int32)
    hi = mx.minimum(lo + 1, in_size - 1)
    w = (coords - lo.astype(mx.float32))
    left = mx.take(x, lo, axis=axis)
    right = mx.take(x, hi, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = out_size
    w = w.reshape(shape)
    return left * (1 - w) + right * w


def resize_bilinear_align_corners(x: mx.array, size: tuple[int, int]) -> mx.array:
    """Bilinear resize for BHWC tensors matching torch ``align_corners=True``."""
    h, w = int(size[0]), int(size[1])
    x = _resize_axis_align_corners(x, h, axis=1)
    x = _resize_axis_align_corners(x, w, axis=2)
    return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.conv1(_relu(x))
        out = self.conv2(_relu(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, *, has_residual: bool = True) -> None:
        super().__init__()
        self.has_residual = has_residual
        self.resConfUnit1 = ResidualConvUnit(features) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    def __call__(self, x: mx.array, residual: mx.array | None = None, *, size: tuple[int, int] | None = None) -> mx.array:
        y = x
        if self.has_residual and residual is not None and self.resConfUnit1 is not None:
            y = y + self.resConfUnit1(residual)
        y = self.resConfUnit2(y)
        if size is None:
            size = (y.shape[1] * 2, y.shape[2] * 2)
        y = resize_bilinear_align_corners(y, size)
        return self.out_conv(y)


class Scratch(nn.Module):
    def __init__(self, out_channels: Sequence[int], features: int) -> None:
        super().__init__()
        self.layer1_rn = nn.Conv2d(out_channels[0], features, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, 3, padding=1, bias=False)
        self.refinenet1 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        self.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        self.output_conv2 = [
            nn.Conv2d(features // 2, 32, 3, padding=1),
            Identity(),
            nn.Conv2d(32, 1, 1),
        ]


class DPTHead(nn.Module):
    """DA3 DPT head consuming four BNC ViT intermediates."""

    def __init__(self, cfg: DPTConfig) -> None:
        super().__init__()
        if len(cfg.out_channels) != 4:
            raise ValueError("DPTHead requires exactly four out_channels entries")
        if cfg.use_sky_head:
            raise NotImplementedError("DPTHead supports DA3 monocular parity with use_sky_head=False")
        if cfg.pos_embed:
            raise NotImplementedError("DPTHead supports the DA3 fixture path with pos_embed=False")
        if cfg.down_ratio <= 0:
            raise ValueError("down_ratio must be positive")
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.output_dim = cfg.output_dim
        self.activation = cfg.activation
        self.conf_activation = cfg.conf_activation
        self.head_name = cfg.head_name
        self.has_conf = cfg.output_dim > 1
        if cfg.norm_type == "layer":
            self.norm = nn.LayerNorm(cfg.dim_in)
        elif cfg.norm_type == "idt":
            self.norm = Identity()
        else:
            raise ValueError("norm_type must be 'idt' or 'layer'")
        self.projects = [nn.Conv2d(cfg.dim_in, oc, 1) for oc in cfg.out_channels]
        self.resize_layers = [
            nn.ConvTranspose2d(cfg.out_channels[0], cfg.out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(cfg.out_channels[1], cfg.out_channels[1], 2, stride=2),
            Identity(),
            nn.Conv2d(cfg.out_channels[3], cfg.out_channels[3], 3, stride=2, padding=1),
        ]
        self.scratch = Scratch(cfg.out_channels, cfg.features)
        self.scratch.output_conv2[-1] = nn.Conv2d(32, cfg.output_dim, 1)

    def _activate(self, x: mx.array, activation: str) -> mx.array:
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return mx.exp(x)
        if act == "expp1":
            return mx.exp(x) + 1
        if act == "expm1":
            return mx.expm1(x)
        if act == "relu":
            return _relu(x)
        if act == "sigmoid":
            return mx.sigmoid(x)
        if act == "softplus":
            return mx.logaddexp(x, mx.zeros_like(x))
        if act == "tanh":
            return mx.tanh(x)
        return x

    def _project_stage(self, x: mx.array, grid: tuple[int, int], stage_idx: int) -> mx.array:
        b, n, c = x.shape
        ph, pw = grid
        if n != ph * pw:
            raise ValueError(f"stage {stage_idx} has {n} tokens but grid {grid} has {ph * pw}")
        x = self.norm(x).reshape(b, ph, pw, c)
        x = self.projects[stage_idx](x)
        return self.resize_layers[stage_idx](x)

    def _fuse(self, feats: list[mx.array], taps: dict[str, mx.array] | None) -> mx.array:
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[1:3])
        if taps is not None:
            taps["fusion_4"] = out
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[1:3])
        if taps is not None:
            taps["fusion_3"] = out
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[1:3])
        if taps is not None:
            taps["fusion_2"] = out
        out = self.scratch.refinenet1(out, l1_rn)
        if taps is not None:
            taps["fusion_1"] = out
        return out

    def __call__(self, inp: HeadInput, *, capture_taps: bool = False) -> HeadOutput:
        features = inp.features
        if len(features.intermediates) != 4:
            raise ValueError(f"DPTHead requires four intermediates, got {len(features.intermediates)}")
        grid = inp.grid or features.grid
        if grid is None:
            raise ValueError("DPTHead requires a patch grid")
        taps: dict[str, mx.array] | None = {} if capture_taps else None
        resized: list[mx.array] = []
        for i, fm in enumerate(features.intermediates):
            if fm.layout is not Layout.BNC:
                raise ValueError(f"DPTHead requires BNC intermediates, got {fm.layout}")
            stage = self._project_stage(fm.data, grid, i)
            resized.append(stage)
            if taps is not None:
                taps[f"projected_{i}"] = stage
        fused = self._fuse(resized, taps)
        if inp.image_size is None:
            h_out = int(grid[0] * self.patch_size / self.cfg.down_ratio)
            w_out = int(grid[1] * self.patch_size / self.cfg.down_ratio)
        else:
            h_out = int(inp.image_size[0] / self.cfg.down_ratio)
            w_out = int(inp.image_size[1] / self.cfg.down_ratio)
        feat = self.scratch.output_conv1(fused)
        feat = resize_bilinear_align_corners(feat, (h_out, w_out))
        logits = self.scratch.output_conv2[0](feat)
        logits = _relu(logits)
        logits = self.scratch.output_conv2[2](logits)
        if taps is not None:
            taps["output_logits"] = logits

        data: dict[str, mx.array | dict[str, mx.array]] = {}
        if self.has_conf:
            depth = self._activate(logits[..., :-1], self.activation)[..., 0]
            conf = self._activate(logits[..., -1], self.conf_activation)
            data[self.head_name] = depth
            data[f"{self.head_name}_conf"] = conf
        else:
            data[self.head_name] = self._activate(logits[..., 0], self.activation)
        if taps is not None:
            data["taps"] = taps
        return HeadOutput(data=data)


@register_head("dpt")
def build_dpt(config) -> DPTHead:
    cfg = config if isinstance(config, DPTConfig) else DPTConfig.from_dict(config)
    return DPTHead(cfg)
