"""MoonViT-SO-400M vision backbone package.

Package-root imports are mlx-free so LocateAnything config/decode/convert can
import ``MoonViTConfig`` without requiring the MLX extra. Import
``mlx_cv.backbones.vision.moonvit.modeling`` once the MLX model exists; that
submodule owns concrete registration.
"""

from __future__ import annotations

from .config import MoonViTConfig

__all__ = [
    "MoonViTConfig",
]
