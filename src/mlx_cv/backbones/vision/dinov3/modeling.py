"""DINOv3 ViT in MLX — a faithful port of the official PyTorch reference.

Ported from `references/dinov3` (`dinov3/models/vision_transformer.py` + layers):
patch-embed → prepend ``[cls, storage…]`` → 12× pre-norm SelfAttention blocks with
axial 2D-RoPE on the patch tokens (cls/storage skipped) → final LayerNorm → split
into cls / storage / patch tokens. No LayerScale (``layerscale_init=None``), plain
Mlp/GELU FFN, qkv bias on. The forward output satisfies the spine ``BackboneFeatures``
contract; numerical parity vs the reference is asserted in `tests/test_dinov3_parity.py`.

The reusable leaves (patch-embed, attention, FFN, RoPE) live in the shared
``backbones/layers`` families; this module wires DINOv3's config over them.

`core/` stays mlx-free: mlx lives only here, behind the ``[mlx]`` extra.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ....core.features import BackboneFeatures, FeatureMap, Layout, TokenLayout
from ....core.registry import register_backbone
from ...layers import PatchEmbed, TransformerBlock
from ...layers.position import rope_axial_periods, rope_axial_sincos
from .config import DINOv3Config

__all__ = ["DINOv3ViT", "build_dinov3"]


class DINOv3ViT(nn.Module):
    """DINOv3 vision transformer (MLX). ``__call__`` == ``forward_features``."""

    def __init__(self, cfg: DINOv3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.in_chans, cfg.embed_dim, cfg.patch_size)
        self.cls_token = mx.zeros((1, 1, cfg.embed_dim))
        if cfg.n_storage_tokens > 0:
            self.storage_tokens = mx.zeros((1, cfg.n_storage_tokens, cfg.embed_dim))
        self.periods = rope_axial_periods(cfg.head_dim, cfg.rope_base)
        self.blocks = [
            TransformerBlock(
                cfg.embed_dim, cfg.num_heads,
                mlp_ratio=cfg.ffn_ratio, qkv_bias=cfg.qkv_bias,
                norm="layernorm", norm_eps=cfg.layer_norm_eps,
                ffn="gelu", layerscale=False,   # DINOv3: no LayerScale
            )
            for _ in range(cfg.depth)
        ]
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)

    def __call__(self, x: mx.array) -> BackboneFeatures:
        return self.forward_features(x)

    def forward_features(self, x: mx.array, *, capture_taps: bool = False) -> BackboneFeatures:
        cfg = self.cfg
        taps: dict[str, mx.array] = {}
        patches, (hp, wp) = self.patch_embed(x)              # (B, P, D)
        b, _, d = patches.shape
        cls = mx.broadcast_to(self.cls_token, (b, 1, d))
        parts = [cls]
        if cfg.n_storage_tokens > 0:
            parts.append(mx.broadcast_to(self.storage_tokens, (b, cfg.n_storage_tokens, d)))
        parts.append(patches)
        tokens = mx.concatenate(parts, axis=1)               # (B, 1+R+P, D)
        if capture_taps:
            taps["patch_embed"] = tokens
        sin, cos = rope_axial_sincos(self.periods, hp, wp)
        if capture_taps:
            taps["rope_sincos"] = mx.stack([sin, cos])
        n_prefix = 1 + cfg.n_storage_tokens
        z = tokens
        for i, blk in enumerate(self.blocks):
            z = blk(z, rope=(sin, cos), n_prefix=n_prefix)
            if capture_taps:
                taps[f"block_{i:02d}"] = z
        z_norm = self.norm(z)
        cls_out = z_norm[:, 0]
        storage_out = z_norm[:, 1:n_prefix]
        patch_out = z_norm[:, n_prefix:]
        if capture_taps:
            taps["norm"] = z_norm
            taps["cls"] = cls_out
            taps["storage"] = storage_out
            taps["patch"] = patch_out
        extras: dict = {"x_prenorm": z}
        if capture_taps:
            extras["taps"] = taps
        return BackboneFeatures(
            patch_tokens=FeatureMap(patch_out, layout=Layout.BNC, grid=(hp, wp), stride=cfg.patch_size),
            cls_token=cls_out,
            storage_tokens=storage_out if cfg.n_storage_tokens > 0 else None,
            token_layout=TokenLayout.vit(n_storage=cfg.n_storage_tokens),
            extras=extras,
        )


@register_backbone("dinov3", kind="vision")
def build_dinov3(config) -> DINOv3ViT:
    """Registry builder: a `parity.fixtures` config dict or a `DINOv3Config`."""
    cfg = config if isinstance(config, DINOv3Config) else DINOv3Config.from_dict(config)
    return DINOv3ViT(cfg)
