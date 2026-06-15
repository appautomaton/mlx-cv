"""DINOv3 ViT in MLX — a thin config-binding over the shared `ViTBackbone`.

Ported from `references/dinov3` (`dinov3/models/vision_transformer.py` + layers):
patch-embed → prepend ``[cls, storage…]`` → 12× pre-norm SelfAttention blocks with
axial 2D-RoPE on the patch tokens (cls/storage skipped) → final LayerNorm → split
into cls / storage / patch tokens. No LayerScale (``layerscale_init=None``), plain
Mlp/GELU FFN, qkv bias on. The forward output satisfies the spine ``BackboneFeatures``
contract; numerical parity vs the reference is asserted in `tests/test_dinov3_parity.py`.

The whole ViT body — assembly, token order, RoPE seam, ``capture_taps`` taps — lives
in the shared `ViTBackbone` family; ``DINOv3ViT`` *subclasses* it and only wires the
DINOv3 config (RoPE strategy, no LayerScale, GELU FFN). Subclassing keeps the param
tree at top level on the instance (``cls_token``, ``storage_tokens``, ``periods``,
``patch_embed.*``, ``blocks.*``, ``norm.*``) so the converted weights load unchanged.

`core/` stays mlx-free: mlx lives only here, behind the ``[mlx]`` extra.
"""

from __future__ import annotations

from ....core.registry import register_backbone
from ..vit import RoPEStrategy, ViTBackbone
from .config import DINOv3Config

__all__ = ["DINOv3ViT", "build_dinov3"]


class DINOv3ViT(ViTBackbone):
    """DINOv3 vision transformer (MLX) = `ViTBackbone` bound to a DINOv3 config.

    ``__call__`` == ``forward_features`` (inherited). Carries no bespoke assembly:
    the config selects the RoPE position strategy, GELU FFN, LayerNorm, and no
    LayerScale — everything else is the shared family.
    """

    def __init__(self, cfg: DINOv3Config) -> None:
        super().__init__(
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            n_storage=cfg.n_storage_tokens,
            mlp_ratio=cfg.ffn_ratio,
            qkv_bias=cfg.qkv_bias,
            norm="layernorm",
            norm_eps=cfg.layer_norm_eps,
            ffn="gelu",
            layerscale=False,                 # DINOv3: no LayerScale
            position=RoPEStrategy(cfg.rope_base),
        )
        self.cfg = cfg


@register_backbone("dinov3", kind="vision")
def build_dinov3(config) -> DINOv3ViT:
    """Registry builder: a `parity.fixtures` config dict or a `DINOv3Config`."""
    cfg = config if isinstance(config, DINOv3Config) else DINOv3Config.from_dict(config)
    return DINOv3ViT(cfg)
