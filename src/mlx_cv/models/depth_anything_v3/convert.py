"""Depth Anything V3 state-dict conversion."""

from __future__ import annotations

import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import mlx.core as mx

from ...backbones.vision.dinov2.convert import convert_dinov2_state_dict
from ...heads.dense.convert import convert_da3_dualdpt_state_dict, convert_dpt_state_dict
from .modeling import DepthAnythingV3Monocular, DepthAnythingV3MultiView

__all__ = [
    "DA3_MULTIVIEW_DEFAULT_AUX_LAYERNORM_KEYS",
    "DA3_UNSUPPORTED_MULTIVIEW_PREFIXES",
    "convert_da3_monocular_state_dict",
    "convert_da3_multiview_state_dict",
    "load_da3_monocular_weights",
    "load_da3_multiview_weights",
]


DA3_UNSUPPORTED_MULTIVIEW_PREFIXES = (
    "model.da3.",
    "model.da3_metric.",
    "model.gs_head.",
    "model.gs_adapter.",
    "model.metric.",
    "model.nested.",
    "da3.",
    "da3_metric.",
    "gs_head.",
    "gs_adapter.",
    "metric.",
    "nested.",
)

DA3_MULTIVIEW_DEFAULT_AUX_LAYERNORM_KEYS = tuple(
    key
    for level_index in range(1, 4)
    for key in (
        f"head.scratch.output_conv2_aux.{level_index}.2.weight",
        f"head.scratch.output_conv2_aux.{level_index}.2.bias",
    )
)


def _strip_prefix(state: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def _strip_first_present_prefix(state: dict[str, np.ndarray], *prefixes: str) -> dict[str, np.ndarray]:
    for prefix in prefixes:
        stripped = _strip_prefix(state, prefix)
        if stripped:
            return stripped
    return {}


def _prefix_items(prefix: str, items):
    return [(f"{prefix}{k}", v) for k, v in items]


def _reject_unsupported_multiview_branches(state: dict[str, np.ndarray]) -> None:
    blocked = sorted(
        key for key in state
        if key.startswith(DA3_UNSUPPORTED_MULTIVIEW_PREFIXES)
    )
    if blocked:
        sample = ", ".join(repr(key) for key in blocked[:5])
        more = "" if len(blocked) <= 5 else f", and {len(blocked) - 5} more"
        raise ValueError(f"unsupported DA3 checkpoint branches for multiview depth/camera load: {sample}{more}")


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


def _with_default_aux_layernorm_items(items: list[tuple[str, mx.array]]) -> list[tuple[str, mx.array]]:
    """Inject DA3 aux LayerNorm defaults missing from the non-strict upstream checkpoint."""

    seen = {key for key, _ in items}
    out = list(items)
    for key in DA3_MULTIVIEW_DEFAULT_AUX_LAYERNORM_KEYS:
        if key in seen:
            continue
        if key.endswith(".weight"):
            out.append((key, mx.ones((32,), dtype=mx.float32)))
        elif key.endswith(".bias"):
            out.append((key, mx.zeros((32,), dtype=mx.float32)))
        else:  # pragma: no cover - guarded by the constant definition.
            raise AssertionError(f"unknown DA3 default aux LayerNorm key: {key}")
    return out


def _looks_like_raw_multiview_state(state: dict[str, np.ndarray]) -> bool:
    """Return true for upstream DA3 state dicts that still need conversion."""

    return any(key.startswith("model.") or key.startswith("backbone.pretrained.") for key in state)


def convert_da3_multiview_state_dict(state: dict[str, np.ndarray]):
    """Convert DA3 any-view backbone, DualDPT, and camera tensors into MLX paths."""
    _reject_unsupported_multiview_branches(state)
    backbone_state = _strip_first_present_prefix(state, "model.backbone.", "backbone.")
    head_state = _strip_first_present_prefix(state, "model.head.", "head.")
    cam_enc_state = _strip_first_present_prefix(state, "model.cam_enc.", "cam_enc.")
    cam_dec_state = _strip_first_present_prefix(state, "model.cam_dec.", "cam_dec.")
    items = []
    items.extend(_prefix_items("backbone.", convert_dinov2_state_dict(backbone_state)))
    items.extend(_prefix_items("head.", convert_da3_dualdpt_state_dict(head_state)))
    items.extend(_prefix_items("cam_enc.", _identity_items(cam_enc_state)))
    items.extend(_prefix_items("cam_dec.", _identity_items(cam_dec_state)))
    return _with_default_aux_layernorm_items(items)


def load_da3_monocular_weights(model: DepthAnythingV3Monocular, weights_path) -> DepthAnythingV3Monocular:
    npz = np.load(weights_path)
    state = {k: npz[k] for k in npz.files}
    model.update(tree_unflatten(convert_da3_monocular_state_dict(state)))
    mx.eval(model.parameters())
    return model


def _validate_converted_items(model, items, *, strict: bool) -> list[tuple[str, mx.array]]:
    seen: dict[str, mx.array] = {}
    duplicates: list[str] = []
    for key, value in items:
        if key in seen:
            duplicates.append(key)
        seen[key] = value
    if duplicates:
        sample = ", ".join(repr(key) for key in duplicates[:5])
        more = "" if len(duplicates) <= 5 else f", and {len(duplicates) - 5} more"
        raise ValueError(f"duplicate converted DA3 keys: {sample}{more}")
    if not strict:
        return list(seen.items())

    params = dict(tree_flatten(model.parameters()))
    missing = sorted(key for key in params if key not in seen)
    extra = sorted(key for key in seen if key not in params)
    mismatched = sorted(
        (key, tuple(params[key].shape), tuple(seen[key].shape))
        for key in params
        if key in seen and tuple(params[key].shape) != tuple(seen[key].shape)
    )
    if missing or extra or mismatched:
        parts = []
        if missing:
            parts.append(f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}")
        if extra:
            parts.append(f"extra={extra[:8]}{'...' if len(extra) > 8 else ''}")
        if mismatched:
            parts.append(f"shape_mismatch={mismatched[:8]}{'...' if len(mismatched) > 8 else ''}")
        raise ValueError("strict DA3 multiview load failed: " + "; ".join(parts))
    return list(seen.items())


def load_da3_multiview_weights(
    model: DepthAnythingV3MultiView,
    weights_path,
    *,
    strict: bool = False,
) -> DepthAnythingV3MultiView:
    npz = np.load(weights_path, allow_pickle=False)
    state = {k: npz[k] for k in npz.files}
    items = convert_da3_multiview_state_dict(state) if _looks_like_raw_multiview_state(state) else _identity_items(state)
    items = _with_default_aux_layernorm_items(items)
    items = _validate_converted_items(
        model,
        items,
        strict=strict,
    )
    model.update(tree_unflatten(items))
    mx.eval(model.parameters())
    return model
