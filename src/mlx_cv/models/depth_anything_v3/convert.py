"""Depth Anything V3 monocular state-dict conversion."""

from __future__ import annotations

import numpy as np
from mlx.utils import tree_unflatten

import mlx.core as mx

from ...backbones.vision.dinov2.convert import convert_dinov2_state_dict
from ...heads.dense.convert import convert_dpt_state_dict
from .modeling import DepthAnythingV3Monocular

__all__ = ["convert_da3_monocular_state_dict", "load_da3_monocular_weights"]


def _strip_prefix(state: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def _prefix_items(prefix: str, items):
    return [(f"{prefix}{k}", v) for k, v in items]


def convert_da3_monocular_state_dict(state: dict[str, np.ndarray]):
    """Convert DA3 ``backbone.*`` and ``head.*`` weights into the MLX model tree."""
    backbone_state = _strip_prefix(state, "backbone.")
    head_state = _strip_prefix(state, "head.")
    items = []
    items.extend(_prefix_items("backbone.", convert_dinov2_state_dict(backbone_state)))
    items.extend(_prefix_items("head.", convert_dpt_state_dict(head_state)))
    return items


def load_da3_monocular_weights(model: DepthAnythingV3Monocular, weights_path) -> DepthAnythingV3Monocular:
    npz = np.load(weights_path)
    state = {k: npz[k] for k in npz.files}
    model.update(tree_unflatten(convert_da3_monocular_state_dict(state)))
    mx.eval(model.parameters())
    return model
