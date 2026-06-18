"""DPT and DA3 DualDPT dense-head state-dict conversion."""

from __future__ import annotations

import numpy as np

from ...hub.convert import TransposePattern, convert_state_dict, load_into
from .dpt import DPTHead

__all__ = [
    "DA3_DUALDPT_CONVERT_RULES",
    "DPT_CONVERT_RULES",
    "convert_da3_dualdpt_state_dict",
    "convert_dpt_state_dict",
    "load_dpt_weights",
]


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


DA3_DUALDPT_CONVERT_RULES = [
    *DPT_CONVERT_RULES,
    TransposePattern("scratch.output_conv1_aux.*.*.weight", (0, 2, 3, 1), ndim=4),
    TransposePattern("scratch.output_conv2_aux.*.0.weight", (0, 2, 3, 1), ndim=4),
    TransposePattern("scratch.output_conv2_aux.*.5.weight", (0, 2, 3, 1), ndim=4),
]


def convert_dpt_state_dict(state: dict[str, np.ndarray]):
    """Map a DA3 DPT ``state_dict`` to ``[(mlx_path, array)]`` via the shared engine."""
    return convert_state_dict(state, DPT_CONVERT_RULES)


def convert_da3_dualdpt_state_dict(state: dict[str, np.ndarray]):
    """Map upstream DA3 ``DualDPT`` tensors to the MLX module tree."""
    return convert_state_dict(state, DA3_DUALDPT_CONVERT_RULES)


def load_dpt_weights(model: DPTHead, weights_path) -> DPTHead:
    """Load a minted DPT ``*_weights.npz`` into ``model`` in place; returns it."""
    npz = np.load(weights_path)
    return load_into(model, {k: npz[k] for k in npz.files}, DPT_CONVERT_RULES)
