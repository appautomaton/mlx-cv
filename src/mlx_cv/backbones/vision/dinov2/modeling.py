"""DINOv2 (with registers) ViT in MLX — a thin config-binding over `ViTBackbone`.

The generalization proof for Phase 2: DINOv2 is a *second* real ViT config that
reuses the shared families with **no new block code**. It differs from DINOv3 on
exactly two parameterized axes — learned-absolute (interpolated) pos-emb via
`AbsPosStrategy` instead of RoPE, and LayerScale on — plus dims/patch/registers.
Registers are carried as the assembly's storage tokens, so token order is
``[cls, register…, patch]`` and they receive no positional embedding.

This module defines **no** attention/block/mlp/rope of its own — it subclasses
`ViTBackbone` (keeping param paths top-level) and wires the config. Weight
conversion + numerical parity are deferred to the phase that consumes DINOv2
weights (Phase 3, Depth Anything V3); this phase proves structural instantiation.

`core/` stays mlx-free: mlx lives only here, behind the ``[mlx]`` extra.
"""

from __future__ import annotations

from ....core.registry import register_backbone
from ..vit import AbsPosStrategy, ViTBackbone
from .config import DINOv2Config

__all__ = ["DINOv2ViT", "build_dinov2"]


class DINOv2ViT(ViTBackbone):
    """DINOv2-with-registers vision transformer (MLX) = `ViTBackbone` + DINOv2 config."""

    def __init__(self, cfg: DINOv2Config) -> None:
        super().__init__(
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            n_storage=cfg.n_register_tokens,   # registers ride the storage-token slot
            mlp_ratio=cfg.ffn_ratio,
            qkv_bias=cfg.qkv_bias,
            norm="layernorm",
            norm_eps=cfg.layer_norm_eps,
            ffn="gelu",
            layerscale=True,                   # DINOv2: LayerScale on
            layerscale_init=cfg.layerscale_init,
            position=AbsPosStrategy(cfg.pretrain_grid),
        )
        self.cfg = cfg


@register_backbone("dinov2", kind="vision")
def build_dinov2(config) -> DINOv2ViT:
    """Registry builder: a config dict (HF `dinov2_with_registers`) or a `DINOv2Config`."""
    cfg = config if isinstance(config, DINOv2Config) else DINOv2Config.from_dict(config)
    return DINOv2ViT(cfg)
