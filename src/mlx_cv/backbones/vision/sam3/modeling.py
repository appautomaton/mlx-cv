"""SAM 3.1 image-mode ViT/VL backbone."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ....core.features import BackboneFeatures, FeatureMap, Layout, TokenLayout
from ....core.registry import register_backbone
from ..vit import AbsPosStrategy, ViTBackbone
from .config import SAM3ImageBackboneConfig

__all__ = ["SAM3ImageBackbone", "build_sam3_image"]


def _as_text_array(text_features: Any) -> mx.array:
    if hasattr(text_features, "language_features"):
        return text_features.language_features
    return mx.array(text_features)


def _mean_text_features(text_features: Any, *, batch: int) -> mx.array:
    """Return one text summary vector per image batch item."""

    arr = _as_text_array(text_features)
    if len(arr.shape) == 2:
        if arr.shape[0] == batch:
            return arr
        if arr.shape[0] == 1:
            return mx.broadcast_to(arr, (batch, arr.shape[-1]))
        raise ValueError(f"SAM3 text features batch {arr.shape[0]} does not match image batch {batch}")
    if len(arr.shape) != 3:
        raise ValueError(f"SAM3 text features must have rank 2 or 3, got {arr.shape}")

    language_mask = getattr(text_features, "language_mask", None)
    if arr.shape[1] == batch:
        seq_first = arr
    elif arr.shape[0] == batch:
        seq_first = arr.transpose(1, 0, 2)
    else:
        raise ValueError(f"SAM3 text features shape {arr.shape} is incompatible with image batch {batch}")

    if language_mask is None:
        return mx.mean(seq_first, axis=0)

    mask = mx.array(language_mask, dtype=mx.bool_)
    if mask.shape[0] != batch or mask.shape[1] != seq_first.shape[0]:
        raise ValueError(
            f"SAM3 language mask shape {mask.shape} is incompatible with text features {seq_first.shape}"
        )
    valid = (~mask).transpose(1, 0)[:, :, None].astype(seq_first.dtype)
    counts = mx.maximum(mx.sum(valid, axis=0), mx.array(1.0, dtype=seq_first.dtype))
    return mx.sum(seq_first * valid, axis=0) / counts


def _fuse_feature(feature: FeatureMap, delta: mx.array) -> FeatureMap:
    if feature.layout is Layout.BNC:
        data = feature.data + delta[:, None, :].astype(feature.data.dtype)
    elif feature.layout is Layout.BHWC:
        data = feature.data + delta[:, None, None, :].astype(feature.data.dtype)
    else:
        raise ValueError(f"SAM3 text fusion expects BNC/BHWC features, got {feature.layout}")
    return FeatureMap(
        data,
        layout=feature.layout,
        grid=feature.grid,
        stride=feature.stride,
        view_axis=feature.view_axis,
    )


class SAM3ImageBackbone(nn.Module):
    """Image-mode SAM3 backbone with a small text-to-vision fusion hook."""

    def __init__(self, cfg: SAM3ImageBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = ViTBackbone(
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=True,
            norm="layernorm",
            norm_eps=1e-6,
            final_norm_eps=1e-6,
            ffn="gelu",
            layerscale=False,
            position=AbsPosStrategy(cfg.pretrain_grid),
        )
        self.text_projection = nn.Linear(cfg.text_dim, cfg.embed_dim)
        self.fusion_norm = nn.LayerNorm(cfg.embed_dim)

    def __call__(
        self,
        image: mx.array,
        *,
        text_features: Any | None = None,
        capture_taps: bool = False,
    ) -> BackboneFeatures:
        return self.forward_features(image, text_features=text_features, capture_taps=capture_taps)

    def forward_features(
        self,
        image: mx.array,
        *,
        text_features: Any | None = None,
        capture_taps: bool = False,
    ) -> BackboneFeatures:
        features = self.vision.forward_features(
            image,
            intermediate_layers=self.cfg.out_layers,
            capture_taps=capture_taps,
        )
        if text_features is None:
            extras = dict(features.extras)
            extras["text_fused"] = False
            return replace(features, extras=extras)

        batch = features.patch_tokens.data.shape[0]
        summary = _mean_text_features(text_features, batch=batch)
        if summary.shape[-1] != self.cfg.text_dim:
            raise ValueError(f"SAM3 text feature width {summary.shape[-1]} does not match text_dim {self.cfg.text_dim}")
        delta = self.fusion_norm(self.text_projection(summary))
        patch_tokens = _fuse_feature(features.patch_tokens, delta)
        intermediates = [_fuse_feature(feature, delta) for feature in features.intermediates]
        cls_token = None if features.cls_token is None else features.cls_token + delta.astype(features.cls_token.dtype)
        storage_tokens = (
            None
            if features.storage_tokens is None
            else features.storage_tokens + delta[:, None, :].astype(features.storage_tokens.dtype)
        )
        extras = dict(features.extras)
        extras["text_fused"] = True
        extras["text_summary"] = summary
        return BackboneFeatures(
            patch_tokens=patch_tokens,
            cls_token=cls_token,
            storage_tokens=storage_tokens,
            token_layout=features.token_layout or TokenLayout.vit(n_storage=0),
            valid_mask=features.valid_mask,
            intermediates=intermediates,
            extras=extras,
        )


@register_backbone("sam3_image", kind="vision")
def build_sam3_image(config) -> SAM3ImageBackbone:
    cfg = config if isinstance(config, SAM3ImageBackboneConfig) else SAM3ImageBackboneConfig(**config)
    return SAM3ImageBackbone(cfg)
