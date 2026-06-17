"""Depth Anything V3 state-dict conversion."""

from __future__ import annotations

import numpy as np
from mlx.utils import tree_unflatten

import mlx.core as mx

from ...backbones.vision.dinov2.convert import convert_dinov2_state_dict
from ...heads.dense.convert import convert_da3_dualdpt_state_dict, convert_dpt_state_dict
from .modeling import DepthAnythingV3Monocular, DepthAnythingV3MultiView

__all__ = [
    "convert_da3_monocular_state_dict",
    "convert_da3_multiview_state_dict",
    "load_da3_monocular_weights",
    "load_da3_multiview_weights",
]


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


def _identity_items(state: dict[str, np.ndarray]):
    return [(k, mx.array(v)) for k, v in state.items()]


def convert_da3_multiview_state_dict(state: dict[str, np.ndarray]):
    """Convert DA3 any-view backbone, DualDPT, and camera tensors into MLX paths."""
    backbone_state = _strip_prefix(state, "backbone.")
    head_state = _strip_prefix(state, "head.")
    cam_enc_state = _strip_prefix(state, "cam_enc.")
    cam_dec_state = _strip_prefix(state, "cam_dec.")
    items = []
    items.extend(_prefix_items("backbone.", convert_dinov2_state_dict(backbone_state)))
    items.extend(_prefix_items("head.", convert_da3_dualdpt_state_dict(head_state)))
    items.extend(_prefix_items("cam_enc.", _identity_items(cam_enc_state)))
    items.extend(_prefix_items("cam_dec.", _identity_items(cam_dec_state)))
    return items


def load_da3_monocular_weights(model: DepthAnythingV3Monocular, weights_path) -> DepthAnythingV3Monocular:
    npz = np.load(weights_path)
    state = {k: npz[k] for k in npz.files}
    model.update(tree_unflatten(convert_da3_monocular_state_dict(state)))
    mx.eval(model.parameters())
    return model


def load_da3_multiview_weights(model: DepthAnythingV3MultiView, weights_path) -> DepthAnythingV3MultiView:
    npz = np.load(weights_path)
    state = {k: npz[k] for k in npz.files}
    model.update(tree_unflatten(convert_da3_multiview_state_dict(state)))
    mx.eval(model.parameters())
    return model
