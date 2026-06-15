"""Weight conversion: official DINOv3 ``state_dict`` -> MLX param tree.

DINOv3's three fixes, expressed as declarative rules over the shared
`hub.convert` engine:

* ``patch_embed.proj.weight``: PyTorch conv ``(O, in, kH, kW)`` -> mlx ``(O, kH, kW, in)``.
* ``rope_embed.periods`` -> ``periods`` (we hold it as a top-level buffer, not a submodule).
* ``mask_token`` dropped (masked-pretraining only; eval forward computes ``cls + 0*mask_token``).

Linear weights are ``(out, in)`` in both frameworks, so they pass through unchanged.
"""

from __future__ import annotations

import numpy as np

from ....hub.convert import Drop, Rename, Transpose, convert_state_dict, load_into
from .modeling import DINOv3ViT

__all__ = ["DINOV3_CONVERT_RULES", "convert_dinov3_state_dict", "load_dinov3_weights"]

DINOV3_CONVERT_RULES = [
    Drop("mask_token"),
    Rename("rope_embed.periods", "periods"),
    Transpose("patch_embed.proj.weight", (0, 2, 3, 1)),     # (O,in,kH,kW) -> (O,kH,kW,in)
]


def convert_dinov3_state_dict(state: dict[str, np.ndarray]):
    """Map a reference ``state_dict`` to ``[(mlx_path, array)]`` via the shared engine."""
    return convert_state_dict(state, DINOV3_CONVERT_RULES)


def load_dinov3_weights(model: DINOv3ViT, weights_path) -> DINOv3ViT:
    """Load a minted ``*_weights.npz`` into ``model`` in place; returns it."""
    npz = np.load(weights_path)
    return load_into(model, {k: npz[k] for k in npz.files}, DINOV3_CONVERT_RULES)
