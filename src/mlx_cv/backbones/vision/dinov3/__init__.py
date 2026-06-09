"""DINOv3 vision backbone (MLX). Importing this module self-registers it.

Registration is decorator-driven (`build_dinov3` -> ``register_backbone("dinov3",
kind="vision")``), so adding DINOv3 touches **no** ``core/`` file — the proof that
the spine generalizes. `core/` stays mlx-free; mlx is imported only here.
"""

from __future__ import annotations

from .config import DINOv3Config
from .convert import convert_dinov3_state_dict, load_dinov3_weights
from .modeling import DINOv3ViT, build_dinov3

__all__ = [
    "DINOv3Config", "DINOv3ViT", "build_dinov3",
    "convert_dinov3_state_dict", "load_dinov3_weights",
]
