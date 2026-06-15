"""Weight conversion: DA3 vendored DINOv2 ``state_dict`` -> MLX param tree."""

from __future__ import annotations

import numpy as np

from ....hub.convert import (
    Drop,
    PrefixRename,
    Rename,
    TransposePattern,
    convert_state_dict,
    load_into,
)
from .modeling import DINOv2ViT

__all__ = ["DINOV2_CONVERT_RULES", "convert_dinov2_state_dict", "load_dinov2_weights"]


DINOV2_CONVERT_RULES = [
    PrefixRename("pretrained.", ""),
    Drop("mask_token"),
    Rename("pos_embed", "pos_embed.table"),
    Rename("register_tokens", "storage_tokens"),
    TransposePattern("patch_embed.proj.weight", (0, 2, 3, 1), ndim=4),
]


def convert_dinov2_state_dict(state: dict[str, np.ndarray]):
    """Map a DA3 DINOv2 ``state_dict`` to ``[(mlx_path, array)]`` via the shared engine."""
    return convert_state_dict(state, DINOV2_CONVERT_RULES)


def load_dinov2_weights(model: DINOv2ViT, weights_path) -> DINOv2ViT:
    """Load a minted DINOv2 ``*_weights.npz`` into ``model`` in place; returns it."""
    npz = np.load(weights_path)
    return load_into(model, {k: npz[k] for k in npz.files}, DINOV2_CONVERT_RULES)
