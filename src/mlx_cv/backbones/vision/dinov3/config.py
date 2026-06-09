"""DINOv3 ViT config (the knobs the MLX port needs to instantiate)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DINOv3Config"]


@dataclass(frozen=True)
class DINOv3Config:
    """Architecture config for a DINOv3 vision transformer.

    Defaults follow the official ``vit_small`` factory (LayerNorm eps 1e-6, plain
    Mlp/GELU FFN, RoPE base 100, no LayerScale). ``n_storage_tokens`` is the number
    of register/storage tokens prepended after the cls token.
    """

    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int = 16
    in_chans: int = 3
    n_storage_tokens: int = 0
    ffn_ratio: float = 4.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6
    rope_base: float = 100.0

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @classmethod
    def from_dict(cls, d: dict) -> "DINOv3Config":
        """Build from a `parity.fixtures` config dict (DINOV3_VARIANT / _FIXTURE_CONFIG)."""
        return cls(
            embed_dim=d["embed_dim"],
            depth=d["depth"],
            num_heads=d["num_heads"],
            patch_size=d.get("patch_size", 16),
            in_chans=d.get("in_chans", 3),
            n_storage_tokens=d.get("n_storage_tokens", 0),
            ffn_ratio=d.get("ffn_ratio", 4.0),
            rope_base=d.get("pos_embed_rope_base", 100.0),
        )
