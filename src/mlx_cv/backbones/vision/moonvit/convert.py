"""Weight conversion/loading for standalone MoonViT fixtures."""

from __future__ import annotations

import numpy as np

from ....hub.convert import Drop, PrefixRename, Rename, Transpose, convert_state_dict, load_into
from .modeling import MoonViTBackbone

__all__ = ["MOONVIT_CONVERT_RULES", "convert_moonvit_state_dict", "load_moonvit_weights"]


MOONVIT_CONVERT_RULES = [
    Drop("__versions_json__"),
    Drop("__config_json__"),
    PrefixRename("encoder.blocks.", "blocks."),
    Rename("encoder.final_layernorm.weight", "final_layernorm.weight"),
    Rename("encoder.final_layernorm.bias", "final_layernorm.bias"),
    Transpose("patch_embed.proj.weight", (0, 2, 3, 1)),
]


def convert_moonvit_state_dict(state: dict[str, np.ndarray]):
    """Map a standalone reference MoonViT ``state_dict`` to local MLX paths."""
    return convert_state_dict(state, MOONVIT_CONVERT_RULES)


def load_moonvit_weights(model: MoonViTBackbone, weights_path) -> MoonViTBackbone:
    """Load a minted MoonViT ``*_weights.npz`` into ``model`` in place; returns it."""
    npz = np.load(weights_path, allow_pickle=False)
    return load_into(model, {k: npz[k] for k in npz.files}, MOONVIT_CONVERT_RULES)
