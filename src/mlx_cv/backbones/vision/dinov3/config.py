"""DINOv3 ViT config (the knobs the MLX port needs to instantiate)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["DINOv3Config"]


@dataclass(frozen=True)
class DINOv3Config:
    """Architecture config for a DINOv3 vision transformer.

    The base defaults preserve the repository's original parity fixture
    (LayerNorm eps 1e-6, plain Mlp/GELU FFN, RoPE base 100, no LayerScale).
    Production consumers can enable LayerScale explicitly; EoMT-DINOv3 does.
    ``n_storage_tokens`` is the number of register/storage tokens prepended after
    the cls token.
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
    layerscale_init: float | None = None

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @classmethod
    def from_dict(cls, d: dict) -> "DINOv3Config":
        """Build from a `parity.fixtures` config dict (DINOV3_VARIANT / _FIXTURE_CONFIG)."""
        return cls(
            embed_dim=d.get("embed_dim", d.get("hidden_size")),
            depth=d.get("depth", d.get("num_hidden_layers")),
            num_heads=d.get("num_heads", d.get("num_attention_heads")),
            patch_size=d.get("patch_size", 16),
            in_chans=d.get("in_chans", d.get("num_channels", 3)),
            n_storage_tokens=d.get("n_storage_tokens", d.get("num_register_tokens", 0)),
            ffn_ratio=d.get(
                "ffn_ratio",
                d.get("intermediate_size", 4 * d.get("embed_dim", d.get("hidden_size")))
                / d.get("embed_dim", d.get("hidden_size")),
            ),
            qkv_bias=d.get(
                "qkv_bias",
                bool(d.get("query_bias", True) or d.get("key_bias", False) or d.get("value_bias", True)),
            ),
            layer_norm_eps=d.get("layer_norm_eps", 1e-6),
            rope_base=d.get("rope_base", d.get("pos_embed_rope_base", d.get("rope_theta", 100.0))),
            layerscale_init=d.get("layerscale_init", d.get("layerscale_value")),
        )

    def to_dict(self) -> dict:
        """Return the normalized JSON-serializable architecture configuration."""

        return asdict(self)
