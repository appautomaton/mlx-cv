"""Faithful SAM 3 detector checkpoint conversion (transformers-native keys).

Maps the real ``facebook/sam3`` ``detector_model.*`` state dict onto the faithful
MLX modules (:mod:`mlx_cv.models.sam3.real_vision` and siblings, added per slice),
so the converter is a mechanical prefix remap plus conv-weight layout fixes.

Kept separate from the reduced clean-room :mod:`mlx_cv.models.sam3.convert` (which
targets the reduced model's ``feature_extractor.*`` paths and rejects video keys) so
the working image/video path is untouched until the faithful modules pass parity.

Conv layout (torch NCHW -> MLX NHWC):
- ``nn.Conv2d`` weight ``[out, in, kH, kW]`` -> ``[out, kH, kW, in]`` = transpose (0, 2, 3, 1)
- ``nn.ConvTranspose2d`` weight ``[in, out, kH, kW]`` -> ``[out, kH, kW, in]`` = transpose (1, 2, 3, 0)
  (the FPN ``scale_layers`` are the only ConvTranspose2d tensors in the vision encoder).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import mlx.core as mx

from .convert import _load_weight_arrays
from .real_detr import Sam3DetrEncoder, Sam3DotProductScoring
from .real_geometry import Sam3GeometryEncoder
from .real_text import Sam3TextEncoder
from .real_vision import Sam3VisionModel

__all__ = [
    "remap_sam3_vision_real_key",
    "convert_reference_shape",
    "convert_sam3_vision_real_state_dict",
    "load_sam3_vision_real_weights",
    "remap_sam3_text_real_key",
    "convert_sam3_text_real_state_dict",
    "load_sam3_text_real_weights",
    "remap_sam3_detr_encoder_real_key",
    "remap_sam3_scoring_real_key",
    "remap_sam3_geometry_real_key",
    "convert_sam3_namespace_state_dict",
    "load_sam3_detr_encoder_real_weights",
    "load_sam3_scoring_real_weights",
    "load_sam3_geometry_real_weights",
]


def _conv_perm(local_key: str, ndim: int) -> tuple[int, ...] | None:
    """Return the torch->MLX axis permutation for a 4D conv weight, else ``None``."""

    if ndim != 4 or not local_key.endswith(".weight"):
        return None
    if ".scale_layers." in local_key:
        return (1, 2, 3, 0)  # ConvTranspose2d [in, out, kH, kW] -> [out, kH, kW, in]
    return (0, 2, 3, 1)  # Conv2d [out, in, kH, kW] -> [out, kH, kW, in]


def remap_sam3_vision_real_key(key: str) -> str | None:
    """Map a reference detector key to a faithful ``Sam3VisionModel`` param path.

    Returns ``None`` for keys outside the ``vision_encoder.*`` namespace (other
    subsystems are handled by later slices).
    """

    key = key.removeprefix("detector_model.")
    if not key.startswith("vision_encoder."):
        return None
    return key.removeprefix("vision_encoder.")


def convert_reference_shape(local_key: str, reference_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Apply the conv layout permutation to a reference shape (no data needed)."""

    perm = _conv_perm(local_key, len(reference_shape))
    if perm is None:
        return tuple(reference_shape)
    return tuple(reference_shape[axis] for axis in perm)


def convert_sam3_vision_real_state_dict(state: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Convert the ``vision_encoder.*`` tensors of a detector checkpoint to MLX paths."""

    items: list[tuple[str, np.ndarray]] = []
    for key, value in state.items():
        local = remap_sam3_vision_real_key(key)
        if local is None:
            continue
        arr = np.asarray(value)
        perm = _conv_perm(local, arr.ndim)
        if perm is not None:
            arr = np.transpose(arr, perm)
        items.append((local, np.ascontiguousarray(arr)))
    return items


def load_sam3_vision_real_weights(model: Sam3VisionModel, weights_path) -> Sam3VisionModel:
    """Load converted faithful vision-encoder weights, enforcing a 1:1 shape match."""

    state = _load_weight_arrays(weights_path)
    converted = convert_sam3_vision_real_state_dict(state)
    params = dict(tree_flatten(model.parameters()))

    converted_keys = {key for key, _ in converted}
    missing = sorted(set(params) - converted_keys)
    if missing:
        sample = ", ".join(repr(k) for k in missing[:5])
        more = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        raise ValueError(f"checkpoint is missing SAM3 vision params: {sample}{more}")

    for key, value in converted:
        if key not in params:
            raise ValueError(f"converted SAM3 vision key {key!r} is not present in the local model")
        if tuple(params[key].shape) != tuple(value.shape):
            raise ValueError(
                f"converted SAM3 vision key {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(params[key].shape)}"
            )

    model.update(tree_unflatten([(key, mx.array(value)) for key, value in converted]))
    mx.eval(model.parameters())
    return model


# --- text encoder (slice 3) ---------------------------------------------------
#
# All text tensors are Linear/Embedding/LayerNorm (2D/1D), so the converter is a
# pure prefix-preserving identity map with no layout transposes.


def remap_sam3_text_real_key(key: str) -> str | None:
    """Map a reference detector key to a faithful ``Sam3TextEncoder`` param path.

    Returns ``None`` for keys outside the ``text_encoder.*`` / ``text_projection.*``
    namespaces. The outer ``text_projection.*`` and the inner CLIP
    ``text_encoder.text_projection.*`` keep their distinct full paths.
    """

    key = key.removeprefix("detector_model.")
    if key.startswith("text_encoder.") or key.startswith("text_projection."):
        return key
    return None


def convert_sam3_text_real_state_dict(state: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Convert the text-tower tensors of a detector checkpoint to MLX paths."""

    items: list[tuple[str, np.ndarray]] = []
    for key, value in state.items():
        local = remap_sam3_text_real_key(key)
        if local is None:
            continue
        items.append((local, np.ascontiguousarray(np.asarray(value))))
    return items


def load_sam3_text_real_weights(model: Sam3TextEncoder, weights_path) -> Sam3TextEncoder:
    """Load converted faithful text-encoder weights, enforcing a 1:1 shape match."""

    state = _load_weight_arrays(weights_path)
    converted = convert_sam3_text_real_state_dict(state)
    params = dict(tree_flatten(model.parameters()))

    converted_keys = {key for key, _ in converted}
    missing = sorted(set(params) - converted_keys)
    if missing:
        sample = ", ".join(repr(k) for k in missing[:5])
        more = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        raise ValueError(f"checkpoint is missing SAM3 text params: {sample}{more}")

    for key, value in converted:
        if key not in params:
            raise ValueError(f"converted SAM3 text key {key!r} is not present in the local model")
        if tuple(params[key].shape) != tuple(value.shape):
            raise ValueError(
                f"converted SAM3 text key {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(params[key].shape)}"
            )

    model.update(tree_unflatten([(key, mx.array(value)) for key, value in converted]))
    mx.eval(model.parameters())
    return model


# --- DETR encoder / scoring / geometry (slice 4) ------------------------------
#
# Generic prefix-stripping remap shared by the slice-4 subsystems. Each maps a
# reference detector key onto its subsystem-local param path (the subsystem module
# is the namespace root), applying the conv layout fix where needed (only the
# geometry boxes_pool_project is a Conv2d).


def _strip(key: str, namespace: str) -> str | None:
    key = key.removeprefix("detector_model.")
    if key.startswith(namespace + "."):
        return key.removeprefix(namespace + ".")
    return None


def remap_sam3_detr_encoder_real_key(key: str) -> str | None:
    """Map a reference key to a faithful ``Sam3DetrEncoder`` param path."""

    return _strip(key, "detr_encoder")


def remap_sam3_scoring_real_key(key: str) -> str | None:
    """Map a reference key to a faithful ``Sam3DotProductScoring`` param path."""

    return _strip(key, "dot_product_scoring")


def remap_sam3_geometry_real_key(key: str) -> str | None:
    """Map a reference key to a faithful ``Sam3GeometryEncoder`` param path."""

    return _strip(key, "geometry_encoder")


def convert_sam3_namespace_state_dict(state, remap) -> list[tuple[str, np.ndarray]]:
    """Convert one subsystem's tensors with ``remap`` (+ conv layout fix)."""

    items: list[tuple[str, np.ndarray]] = []
    for key, value in state.items():
        local = remap(key)
        if local is None:
            continue
        arr = np.asarray(value)
        perm = _conv_perm(local, arr.ndim)
        if perm is not None:
            arr = np.transpose(arr, perm)
        items.append((local, np.ascontiguousarray(arr)))
    return items


def _load_with_remap(model, weights_path, remap, kind: str):
    """Load converted subsystem weights, enforcing a 1:1 shape match."""

    state = _load_weight_arrays(weights_path)
    converted = convert_sam3_namespace_state_dict(state, remap)
    params = dict(tree_flatten(model.parameters()))

    converted_keys = {key for key, _ in converted}
    missing = sorted(set(params) - converted_keys)
    if missing:
        sample = ", ".join(repr(k) for k in missing[:5])
        more = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        raise ValueError(f"checkpoint is missing SAM3 {kind} params: {sample}{more}")

    for key, value in converted:
        if key not in params:
            raise ValueError(f"converted SAM3 {kind} key {key!r} is not present in the local model")
        if tuple(params[key].shape) != tuple(value.shape):
            raise ValueError(
                f"converted SAM3 {kind} key {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(params[key].shape)}"
            )

    model.update(tree_unflatten([(key, mx.array(value)) for key, value in converted]))
    mx.eval(model.parameters())
    return model


def load_sam3_detr_encoder_real_weights(model: Sam3DetrEncoder, weights_path) -> Sam3DetrEncoder:
    return _load_with_remap(model, weights_path, remap_sam3_detr_encoder_real_key, "detr encoder")


def load_sam3_scoring_real_weights(model: Sam3DotProductScoring, weights_path) -> Sam3DotProductScoring:
    return _load_with_remap(model, weights_path, remap_sam3_scoring_real_key, "scoring")


def load_sam3_geometry_real_weights(model: Sam3GeometryEncoder, weights_path) -> Sam3GeometryEncoder:
    return _load_with_remap(model, weights_path, remap_sam3_geometry_real_key, "geometry")
