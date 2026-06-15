"""Depth Anything V3 monocular model assembly."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...backbones.vision.dinov2 import DINOv2ViT
from ...core.features import HeadInput, HeadOutput
from ...core.registry import register_model
from ...heads.dense import DPTHead
from .config import DA3MonocularConfig

__all__ = ["DepthAnythingV3Monocular", "build_depth_anything_v3_monocular"]


class DepthAnythingV3Monocular(nn.Module):
    """DA3 monocular path: DINOv2 selected intermediates -> DPT depth head."""

    def __init__(self, cfg: DA3MonocularConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DINOv2ViT(cfg.backbone)
        self.head = DPTHead(cfg.head)

    def __call__(self, x: mx.array, *, capture_taps: bool = False) -> HeadOutput:
        if x.ndim != 4:
            raise ValueError(f"DepthAnythingV3Monocular expects NCHW input, got shape {x.shape}")
        if x.shape[1] != self.cfg.backbone.in_chans:
            raise ValueError(
                f"DepthAnythingV3Monocular expects NCHW input with {self.cfg.backbone.in_chans} "
                f"channels at axis 1, got shape {x.shape}"
            )
        if x.shape[2] % self.cfg.backbone.patch_size or x.shape[3] % self.cfg.backbone.patch_size:
            raise ValueError(
                f"DepthAnythingV3Monocular input height/width must be divisible by patch size "
                f"{self.cfg.backbone.patch_size}, got shape {x.shape}"
            )
        feats = self.backbone.forward_features(
            x,
            intermediate_layers=self.cfg.out_layers,
            capture_taps=capture_taps,
        )
        out = self.head(
            HeadInput(features=feats, image_size=(int(x.shape[2]), int(x.shape[3]))),
            capture_taps=capture_taps,
        )
        if capture_taps:
            taps = {}
            taps.update({f"dinov2.{k}": v for k, v in feats.extras.get("taps", {}).items()})
            taps.update({f"dpt.{k}": v for k, v in out.data.get("taps", {}).items()})
            out.data["taps"] = taps
        return out


@register_model("depth-anything-v3-monocular")
def build_depth_anything_v3_monocular(config) -> DepthAnythingV3Monocular:
    cfg = config if isinstance(config, DA3MonocularConfig) else DA3MonocularConfig.from_dict(config)
    return DepthAnythingV3Monocular(cfg)
