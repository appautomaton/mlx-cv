"""Local DA3 real-checkpoint capture helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..core.types import Result
from ..models.depth_anything_v3 import (
    DA3MultiViewConfig,
    DA3Processor,
    DA3ProcessorConfig,
    DepthAnythingV3MultiView,
    load_da3_multiview_weights,
)
from .fixtures import da3_multiview_fixed_images

__all__ = [
    "DA3LocalCapture",
    "build_da3_small_local_model",
    "capture_da3_small_local",
]


def _np(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def _eval_head_output(data: dict[str, Any]) -> None:
    values = []
    for value in data.values():
        if hasattr(value, "shape"):
            values.append(value)
    if values:
        mx.eval(*values)


def _comparable_aux_taps(
    model: DepthAnythingV3MultiView,
    raw_taps: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Derive upstream-comparable normalized aux features from local block taps."""

    n_prefix = 1 + model.backbone.n_storage
    aux: dict[str, np.ndarray] = {}
    for layer in model.cfg.backbone.out_layers:
        raw_key = f"anyview.block_{int(layer):02d}"
        value = raw_taps.get(raw_key)
        if value is None or not hasattr(value, "shape"):
            continue
        normalized = model.backbone.norm(value)
        normalized = normalized[:, :, n_prefix:]
        batch, views, tokens, channels = normalized.shape
        side = int(tokens**0.5)
        if batch != 1 or side * side != tokens:
            continue
        normalized = normalized.reshape(views, side, side, channels)
        mx.eval(normalized)
        aux[f"aux_feat_layer_{int(layer):02d}"] = _np(normalized)
    return aux


@dataclass(frozen=True)
class DA3LocalCapture:
    input_images: np.ndarray
    input_tensor: np.ndarray
    raw_depth: np.ndarray
    raw_confidence: np.ndarray
    raw_ray: np.ndarray
    raw_ray_confidence: np.ndarray
    pose_encoding: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    result: Result
    taps: dict[str, np.ndarray]

    def as_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "input_images": self.input_images,
            "input_tensor": self.input_tensor,
            "raw_depth": self.raw_depth,
            "raw_confidence": self.raw_confidence,
            "raw_ray": self.raw_ray,
            "raw_ray_confidence": self.raw_ray_confidence,
            "pose_encoding": self.pose_encoding,
            "extrinsics": self.extrinsics,
            "intrinsics": self.intrinsics,
        }
        arrays.update({f"tap.{key}": value for key, value in self.taps.items()})
        return arrays

    def summary(self) -> dict[str, Any]:
        return {
            "input_shape": list(self.input_images.shape),
            "input_tensor_shape": list(self.input_tensor.shape),
            "depth_shape": list(self.raw_depth.shape),
            "confidence_shape": list(self.raw_confidence.shape),
            "ray_shape": list(self.raw_ray.shape),
            "ray_confidence_shape": list(self.raw_ray_confidence.shape),
            "pose_encoding_shape": list(self.pose_encoding.shape),
            "extrinsics_shape": list(self.extrinsics.shape),
            "intrinsics_shape": list(self.intrinsics.shape),
            "result_depth_views": 0 if self.result.depth_views is None else len(self.result.depth_views),
            "tap_order": list(self.taps),
        }


def build_da3_small_local_model(weights_path: str | Path) -> DepthAnythingV3MultiView:
    """Build DA3-SMALL any-view model and strict-load converted local weights."""

    model = DepthAnythingV3MultiView(DA3MultiViewConfig.small())
    return load_da3_multiview_weights(model, weights_path, strict=True)


def capture_da3_small_local(
    model: DepthAnythingV3MultiView,
    *,
    images: np.ndarray | None = None,
    process_res: int = 112,
    capture_taps: bool = False,
) -> DA3LocalCapture:
    """Run local DA3-SMALL on the canonical fixed three-view input."""

    input_images = da3_multiview_fixed_images() if images is None else np.asarray(images)
    processor = DA3Processor(DA3ProcessorConfig(process_res=process_res, patch_size=model.cfg.backbone.patch_size))
    x, ctx = processor.preprocess(list(input_images))
    raw = model(x, capture_taps=capture_taps, reference_view_strategy="middle")
    _eval_head_output(raw.data)
    result = processor.postprocess(raw, ctx)

    raw_taps = raw.data.get("taps", {})
    taps = {
        key: _np(value)
        for key, value in raw_taps.items()
        if hasattr(value, "shape")
    }
    taps.update(_comparable_aux_taps(model, raw_taps))
    return DA3LocalCapture(
        input_images=np.asarray(input_images),
        input_tensor=_np(x),
        raw_depth=_np(raw["depth"]),
        raw_confidence=_np(raw["depth_conf"]),
        raw_ray=_np(raw["ray"]),
        raw_ray_confidence=_np(raw["ray_conf"]),
        pose_encoding=_np(raw["pose_encoding"]),
        extrinsics=_np(raw["extrinsics"]),
        intrinsics=_np(raw["intrinsics"]),
        result=result,
        taps=taps,
    )
