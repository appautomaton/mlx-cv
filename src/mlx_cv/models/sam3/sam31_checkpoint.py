"""Strict direct loading for final-layout SAM 3.1 MLX Safetensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

__all__ = [
    "SAM31_CHECKPOINT_METADATA",
    "SAM31CheckpointError",
    "load_sam31_detector_weights",
    "load_sam31_tracker_weights",
    "load_sam31_weights",
    "read_safetensors_metadata",
]


SAM31_CHECKPOINT_METADATA = {
    "format": "mlx-cv-sam3.1-v1",
    "architecture": "sam3.1-multiplex",
    "layout": "mlx-final",
    "dtype": "bfloat16",
}


class SAM31CheckpointError(ValueError):
    """Raised when a SAM 3.1 production checkpoint violates its contract."""


def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    """Read the small JSON header without importing the safetensors package."""

    path = Path(path)
    try:
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            if header_size <= 0 or header_size > 100_000_000:
                raise SAM31CheckpointError(f"invalid Safetensors header size: {header_size}")
            header = json.loads(handle.read(header_size))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SAM31CheckpointError(f"cannot read SAM 3.1 Safetensors header: {path}") from exc
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise SAM31CheckpointError("SAM 3.1 Safetensors metadata must be string pairs")
    return metadata


def _validate_metadata(metadata: dict[str, str]) -> None:
    errors = {
        key: (metadata.get(key), expected)
        for key, expected in SAM31_CHECKPOINT_METADATA.items()
        if metadata.get(key) != expected
    }
    if errors:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in errors.items()
        )
        raise SAM31CheckpointError(f"incompatible SAM 3.1 checkpoint metadata: {details}")


def _read_weights(path: str | Path) -> tuple[dict[str, mx.array], dict[str, str]]:
    path = Path(path)
    if path.suffix != ".safetensors":
        raise SAM31CheckpointError("SAM 3.1 production weights must use .safetensors")
    metadata = read_safetensors_metadata(path)
    _validate_metadata(metadata)
    weights = mx.load(str(path))
    if not isinstance(weights, dict):
        raise SAM31CheckpointError("SAM 3.1 Safetensors did not contain a tensor map")
    return weights, metadata


def _load_weights(model: Any, weights: dict[str, mx.array]) -> Any:

    params = dict(tree_flatten(model.parameters()))
    missing = sorted(set(params) - set(weights))
    unexpected = sorted(set(weights) - set(params))
    if missing or unexpected:
        raise SAM31CheckpointError(
            "SAM 3.1 parameter names do not match: "
            f"missing={missing[:5]!r}, unexpected={unexpected[:5]!r}"
        )
    for key, value in weights.items():
        if tuple(value.shape) != tuple(params[key].shape):
            raise SAM31CheckpointError(
                f"SAM 3.1 tensor {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(params[key].shape)}"
            )
        if value.dtype != mx.bfloat16:
            raise SAM31CheckpointError(
                f"SAM 3.1 tensor {key!r} has dtype {value.dtype}, expected bfloat16"
            )

    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def load_sam31_detector_weights(model: Any, path: str | Path) -> Any:
    """Load the detector subtree from the final combined BF16 checkpoint."""

    weights, metadata = _read_weights(path)
    if metadata.get("scope") == "multiplex":
        prefix = "detector."
        weights = {
            key.removeprefix(prefix): value
            for key, value in weights.items()
            if key.startswith(prefix)
        }
    elif metadata.get("scope") != "detector":
        raise SAM31CheckpointError("SAM 3.1 checkpoint has no detector scope")
    return _load_weights(model, weights)


def load_sam31_tracker_weights(model: Any, path: str | Path) -> Any:
    """Load the tracker subtree from the final combined BF16 checkpoint."""

    weights, metadata = _read_weights(path)
    if metadata.get("scope") != "multiplex":
        raise SAM31CheckpointError("SAM 3.1 checkpoint has no tracker scope")
    prefix = "tracker."
    selected = {
        key.removeprefix(prefix): value
        for key, value in weights.items()
        if key.startswith(prefix)
    }
    return _load_weights(model, selected)


def load_sam31_weights(model: Any, path: str | Path) -> Any:
    """Load the complete final-layout detector + tracker checkpoint."""

    weights, metadata = _read_weights(path)
    if metadata.get("scope") != "multiplex":
        raise SAM31CheckpointError("complete SAM 3.1 load requires multiplex scope")
    return _load_weights(model, weights)
