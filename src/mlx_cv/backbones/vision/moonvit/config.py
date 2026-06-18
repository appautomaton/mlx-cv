"""MoonViT-SO-400M config for LocateAnything's vision backbone.

This module is intentionally mlx-free. ``models.locateanything`` imports it for
config-only tests, so package-root imports must not pull in MLX modeling code or
register the backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MoonViTConfig"]


@dataclass
class MoonViTConfig:
    """MoonViT-SO-400M native-resolution vision encoder."""

    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 14
    num_channels: int = 3
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    merge_kernel_size: tuple[int, int] = (2, 2)

    @property
    def embed_dim(self) -> int:
        return self.hidden_size

    @property
    def depth(self) -> int:
        return self.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def spatial_merge_size(self) -> int:
        return self.merge_kernel_size[0]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MoonViTConfig":
        """Build from a reference-style config dict."""
        data = dict(d)
        if "vision_config" in data:
            data = dict(data["vision_config"])
        if "merge_kernel_size" in data:
            data["merge_kernel_size"] = tuple(data["merge_kernel_size"])
        allowed = {name for name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})
