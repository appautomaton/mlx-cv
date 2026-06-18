"""DINOv2 (with registers) vision backbone (MLX). Importing this self-registers it.

The Phase-2 generalization proof: a second real ViT config built entirely from
the shared `backbones/layers` + `ViTBackbone` families — no new block code, no
``core/`` edit. Registration is decorator-driven (`build_dinov2` ->
``register_backbone("dinov2", kind="vision")``).
"""

from __future__ import annotations

from .convert import DINOV2_CONVERT_RULES, convert_dinov2_state_dict, load_dinov2_weights
from .anyview import DA3AnyViewDINOv2, DA3Attention, DA3TransformerBlock, select_reference_view
from .config import DA3AnyViewDINOv2Config, DINOv2Config
from .modeling import DINOv2ViT, build_dinov2

__all__ = [
    "DA3AnyViewDINOv2",
    "DA3AnyViewDINOv2Config",
    "DA3Attention",
    "DA3TransformerBlock",
    "DINOV2_CONVERT_RULES",
    "DINOv2Config",
    "DINOv2ViT",
    "build_dinov2",
    "convert_dinov2_state_dict",
    "load_dinov2_weights",
    "select_reference_view",
]
