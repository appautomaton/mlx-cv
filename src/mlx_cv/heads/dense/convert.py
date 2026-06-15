"""DPT dense-head state-dict conversion."""

from __future__ import annotations

import numpy as np

from ...hub.convert import TransposePattern, convert_state_dict, load_into
from .dpt import DPTHead

__all__ = ["DPT_CONVERT_RULES", "convert_dpt_state_dict", "load_dpt_weights"]


DPT_CONVERT_RULES = [
    TransposePattern("resize_layers.[01].weight", (1, 2, 3, 0), ndim=4),
    TransposePattern("projects.*.weight", (0, 2, 3, 1), ndim=4),
    TransposePattern("resize_layers.3.weight", (0, 2, 3, 1), ndim=4),
    TransposePattern("scratch.layer*_rn.weight", (0, 2, 3, 1), ndim=4),
    TransposePattern("scratch.refinenet*.resConfUnit*.conv*.weight", (0, 2, 3, 1), ndim=4),
    TransposePattern("scratch.refinenet*.out_conv.weight", (0, 2, 3, 1), ndim=4),
    TransposePattern("scratch.output_conv1.weight", (0, 2, 3, 1), ndim=4),
    TransposePattern("scratch.output_conv2.*.weight", (0, 2, 3, 1), ndim=4),
]


def convert_dpt_state_dict(state: dict[str, np.ndarray]):
    """Map a DA3 DPT ``state_dict`` to ``[(mlx_path, array)]`` via the shared engine."""
    return convert_state_dict(state, DPT_CONVERT_RULES)


def load_dpt_weights(model: DPTHead, weights_path) -> DPTHead:
    """Load a minted DPT ``*_weights.npz`` into ``model`` in place; returns it."""
    npz = np.load(weights_path)
    return load_into(model, {k: npz[k] for k in npz.files}, DPT_CONVERT_RULES)
