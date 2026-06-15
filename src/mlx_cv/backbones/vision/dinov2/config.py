"""DINOv2 (with registers) ViT config — the knobs the shared `ViTBackbone` needs."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DINOv2Config"]


@dataclass(frozen=True)
class DINOv2Config:
    """Architecture config for a DINOv2-with-registers vision transformer.

    Differs from DINOv3 on exactly the parameterized axes the families expose:
    learned-absolute (interpolated) pos-emb instead of RoPE, LayerScale on, and
    ``patch_size`` 14. ``pretrain_grid`` is the pos-emb table side (``image_size //
    patch_size``); the table is bicubic-interpolated to the runtime grid.
    """

    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int = 14
    in_chans: int = 3
    n_register_tokens: int = 4
    pretrain_grid: int = 37          # 518 // 14 for with-registers checkpoints
    ffn_ratio: float = 4.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6
    final_norm_eps: float = 1e-5
    layerscale_init: float = 1.0

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @classmethod
    def from_dict(cls, d: dict) -> "DINOv2Config":
        """Build from an HF ``dinov2_with_registers`` config dict (`references/rf-detr/...`)."""
        patch = d.get("patch_size", 14)
        return cls(
            embed_dim=d["hidden_size"],
            depth=d["num_hidden_layers"],
            num_heads=d["num_attention_heads"],
            patch_size=patch,
            in_chans=d.get("num_channels", 3),
            n_register_tokens=d.get("num_register_tokens", 4),
            pretrain_grid=d.get("image_size", 518) // patch,
            ffn_ratio=d.get("mlp_ratio", 4.0),
            qkv_bias=d.get("qkv_bias", True),
            layer_norm_eps=d.get("layer_norm_eps", 1e-6),
            final_norm_eps=d.get("final_norm_eps", 1e-5),
            layerscale_init=d.get("layerscale_value", 1.0),
        )
