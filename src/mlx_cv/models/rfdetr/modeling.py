"""RF-DETR DINOv2 feature path."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...backbones.vision.dinov2 import DINOv2ViT
from ...backbones.vision.necks import RFDETRFeaturePyramid, RFDETRMultiScaleProjector
from ...core.features import BackboneFeatures
from .config import RFDETRConfig

__all__ = ["RFDETRDINOv2Adapter", "RFDETRFeatureExtractor"]


class RFDETRDINOv2Adapter(nn.Module):
    """RF-DETR-owned wrapper around the local DINOv2 backbone."""

    def __init__(self, cfg: RFDETRConfig) -> None:
        super().__init__()
        if not cfg.out_layers:
            raise ValueError("RF-DETR requires at least one DINOv2 output layer")
        invalid = [i for i in cfg.out_layers if i < 0 or i >= cfg.backbone.depth]
        if invalid:
            raise ValueError(f"RF-DETR out_layers outside DINOv2 depth: {invalid}")
        self.cfg = cfg
        self.backbone = DINOv2ViT(cfg.backbone)

    def __call__(self, x: mx.array, *, capture_taps: bool = False) -> BackboneFeatures:
        if len(x.shape) != 4:
            raise ValueError(f"RF-DETR DINOv2 adapter expects NCHW input, got {x.shape}")
        if x.shape[1] != self.cfg.backbone.in_chans:
            raise ValueError(
                f"RF-DETR DINOv2 adapter expects {self.cfg.backbone.in_chans} channels, got {x.shape}"
            )
        patch = self.cfg.backbone.patch_size
        if x.shape[2] % patch or x.shape[3] % patch:
            raise ValueError(f"RF-DETR input height/width must be divisible by patch size {patch}, got {x.shape}")
        return self.backbone.forward_features(
            x,
            intermediate_layers=self.cfg.out_layers,
            capture_taps=capture_taps,
        )


class RFDETRFeatureExtractor(nn.Module):
    """DINOv2 adapter plus multi-scale projector for RF-DETR."""

    def __init__(self, cfg: RFDETRConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = RFDETRDINOv2Adapter(cfg)
        self.projector = RFDETRMultiScaleProjector(
            in_channels=(cfg.backbone.embed_dim,) * len(cfg.out_layers),
            out_channels=cfg.projector_out_channels,
            scale_factors=cfg.projector_scale_factors,
        )

    def __call__(self, x: mx.array, *, capture_taps: bool = False) -> RFDETRFeaturePyramid:
        features = self.backbone(x, capture_taps=capture_taps)
        pyramid = self.projector(features)
        if capture_taps:
            features.extras["rfdetr_pyramid"] = [level.data for level in pyramid.levels]
        return pyramid
