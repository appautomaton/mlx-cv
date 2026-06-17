"""SAM 3.1 image-mode checkpoint conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import mlx.core as mx

from .modeling import SAM3Model

__all__ = [
    "convert_sam3_state_dict",
    "inspect_sam3_video_state_dict",
    "load_sam3_weights",
    "remap_sam3_key",
]


_LOCAL_PREFIXES = (
    "text_encoder.",
    "feature_extractor.",
    "decoder.",
    "mask_decoder.",
)
_METADATA_KEYS = {"args", "config", "model_config", "__args_json__", "__config_json__", "__metadata_json__"}
_VIDEO_KEY_PARTS = (
    "video",
    "tracker",
    "track",
    "memory_encoder",
    "memory_attention",
    "temporal",
)
_VIDEO_GATE_KEY_PARTS = _VIDEO_KEY_PARTS + (
    "memory",
    "maskmem",
    "multiplex",
    "sam2_predictor",
    "detector",
    "obj_ptr",
)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _json_like(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.dtype.kind in ("U", "S"):
            value = "".join(str(x) for x in value.reshape(-1))
        else:
            return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        return value
    return None


def _contains_key_part_flag(obj: Any, parts: tuple[str, ...]) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            if any(part in lowered for part in parts) and bool(value):
                return True
            if _contains_key_part_flag(value, parts):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_contains_key_part_flag(item, parts) for item in obj)
    return False


def _contains_video_flag(obj: Any) -> bool:
    return _contains_key_part_flag(obj, _VIDEO_KEY_PARTS)


def _metadata_key_parts(obj: Any, parts: tuple[str, ...]) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            for part in parts:
                if part in lowered and bool(value):
                    found.add(part)
            found.update(_metadata_key_parts(value, parts))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.update(_metadata_key_parts(item, parts))
    return found


def _reject_unsupported_variant(state: dict[str, Any]) -> None:
    for key, value in state.items():
        lowered = key.lower()
        if any(part in lowered for part in _VIDEO_KEY_PARTS):
            raise ValueError(
                "SAM 3.1 video/tracker checkpoints are not supported by the image-mode loader; "
                f"found key {key!r}"
            )
        if key in _METADATA_KEYS:
            metadata = _json_like(value)
            if _contains_video_flag(metadata):
                raise ValueError(
                    "SAM 3.1 video/tracker checkpoints are not supported by the image-mode loader; "
                    f"metadata key {key!r} declares video/tracker state"
                )


def inspect_sam3_video_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Inspect a checkpoint-like mapping for SAM3 video/tracker key families.

    This is intentionally separate from the image-mode converter. It recognizes
    video/tracker/multiplex candidates without making them loadable through
    :func:`convert_sam3_state_dict`.
    """

    matched: dict[str, list[str]] = {}
    for key, value in state.items():
        lowered = key.lower()
        for part in _VIDEO_GATE_KEY_PARTS:
            if part in lowered:
                matched.setdefault(part, []).append(key)
        if key in _METADATA_KEYS:
            metadata = _json_like(value)
            for part in _metadata_key_parts(metadata, _VIDEO_GATE_KEY_PARTS):
                matched.setdefault(part, []).append(key)
    sample_keys = []
    for keys in matched.values():
        sample_keys.extend(keys[:2])
    return {
        "is_video_candidate": bool(matched),
        "matched_key_parts": sorted(matched),
        "sample_keys": sample_keys[:10],
    }


def _strip_reference_prefix(key: str) -> str:
    for prefix in ("module.", "model.", "sam.", "sam3."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _transpose_conv_if_reference(key: str, value: np.ndarray) -> np.ndarray:
    if value.ndim == 4 and key.endswith(".weight"):
        return np.transpose(value, (0, 2, 3, 1))
    return value


def remap_sam3_key(key: str) -> tuple[str | None, bool]:
    """Map one reference SAM3 image-mode key to the local MLX key."""

    if key.startswith("__") or key in _METADATA_KEYS:
        return None, False
    if key.startswith(_LOCAL_PREFIXES):
        return key, False

    stripped = _strip_reference_prefix(key)
    if stripped.startswith(_LOCAL_PREFIXES):
        return stripped, True
    key = stripped
    if key.startswith("image_encoder."):
        return f"feature_extractor.backbone.vision.{key.removeprefix('image_encoder.')}", True
    if key.startswith("image_backbone."):
        return f"feature_extractor.backbone.vision.{key.removeprefix('image_backbone.')}", True
    if key.startswith("vision_encoder."):
        return f"feature_extractor.backbone.vision.{key.removeprefix('vision_encoder.')}", True
    if key.startswith("neck."):
        return f"feature_extractor.neck.{key.removeprefix('neck.')}", True
    if key.startswith("prompt_encoder.text_encoder."):
        return f"text_encoder.{key.removeprefix('prompt_encoder.text_encoder.')}", True
    if key.startswith("text_encoder."):
        return key, True
    if key.startswith("image_decoder."):
        return f"decoder.{key.removeprefix('image_decoder.')}", True
    if key.startswith("mask_decoder."):
        return key, True
    return None, True


def _convert_value(mapped_key: str, value: Any, *, reference_layout: bool) -> np.ndarray:
    out = _to_numpy(value)
    if reference_layout:
        out = _transpose_conv_if_reference(mapped_key, out)
    return np.asarray(out)


def convert_sam3_state_dict(state: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Convert SAM3 image-mode weights into local MLX parameter paths."""

    _reject_unsupported_variant(state)
    items: list[tuple[str, np.ndarray]] = []
    unknown: list[str] = []
    for key, value in state.items():
        mapped, reference_layout = remap_sam3_key(key)
        if mapped is None:
            if key.startswith("__") or key in _METADATA_KEYS:
                continue
            unknown.append(key)
            continue
        items.append((mapped, _convert_value(mapped, value, reference_layout=reference_layout)))
    if unknown:
        sample = ", ".join(repr(k) for k in unknown[:5])
        more = "" if len(unknown) <= 5 else f", and {len(unknown) - 5} more"
        raise ValueError(f"unsupported SAM3 checkpoint keys: {sample}{more}")
    return items


def _load_weight_arrays(weights_path) -> dict[str, np.ndarray]:
    path = Path(weights_path)
    if path.suffix == ".npz":
        npz = np.load(path, allow_pickle=False)
        return {k: npz[k] for k in npz.files}
    if path.suffix == ".safetensors":
        return {k: np.array(v) for k, v in mx.load(str(path)).items()}
    raise ValueError(f"unsupported SAM3 weight format: {path}")


def load_sam3_weights(model: SAM3Model, weights_path) -> SAM3Model:
    """Load converted SAM3 image-mode weights from ``.npz`` or safetensors."""

    state = _load_weight_arrays(weights_path)
    converted = [(key, mx.array(value)) for key, value in convert_sam3_state_dict(state)]
    params = dict(tree_flatten(model.parameters()))
    for key, value in converted:
        if key not in params:
            raise ValueError(f"converted SAM3 key {key!r} is not present in the local model")
        if tuple(params[key].shape) != tuple(value.shape):
            raise ValueError(
                f"converted SAM3 key {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(params[key].shape)}"
            )
    model.update(tree_unflatten(converted))
    mx.eval(model.parameters())
    return model
