"""MoonViT-SO-400M vision backbone package.

Package-root imports are mlx-free so LocateAnything config/decode/convert can
import ``MoonViTConfig`` without requiring the MLX extra. Import
``mlx_cv.backbones.vision.moonvit.modeling`` once the MLX model exists; that
submodule owns concrete registration.
"""

from __future__ import annotations

from .config import MoonViTConfig

__all__ = [
    "Learnable2DInterpPosEmb",
    "MoonViTBackbone",
    "MoonViTConfig",
    "MoonViTEncoderLayer",
    "MoonViTMLP",
    "MoonViTPatchEmbed",
    "MOONVIT_CONVERT_RULES",
    "Rope2DPosEmb",
    "apply_rope",
    "bicubic_interpolate",
    "build_moonvit_so400m",
    "convert_moonvit_state_dict",
    "cu_seqlens_from_grid_hws",
    "load_moonvit_weights",
    "make_block_attention_mask",
    "patch_merger",
]


def __getattr__(name: str):
    modeling_exports = {
        "Learnable2DInterpPosEmb",
        "MoonViTBackbone",
        "MoonViTEncoderLayer",
        "MoonViTMLP",
        "MoonViTPatchEmbed",
        "Rope2DPosEmb",
        "apply_rope",
        "bicubic_interpolate",
        "build_moonvit_so400m",
        "cu_seqlens_from_grid_hws",
        "make_block_attention_mask",
        "patch_merger",
    }
    convert_exports = {
        "MOONVIT_CONVERT_RULES",
        "convert_moonvit_state_dict",
        "load_moonvit_weights",
    }
    if name in modeling_exports:
        from . import modeling

        return getattr(modeling, name)
    if name in convert_exports:
        from . import convert

        return getattr(convert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
