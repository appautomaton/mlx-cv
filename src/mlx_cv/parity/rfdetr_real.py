"""Local RF-DETR Nano real-checkpoint capture helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..core.geometry import SpatialTransform
from ..core.types import Result
from ..models.rfdetr import (
    RFDETRConfig,
    RFDETRModel,
    RFDETRProcessor,
    RFDETRProcessorConfig,
    load_rfdetr_weights,
)
from .fixtures import rfdetr_fixed_image

__all__ = [
    "RFDETRLocalCapture",
    "build_rfdetr_nano_local_model",
    "capture_rfdetr_nano_local",
    "preprocess_rfdetr_nano_image",
    "rfdetr_nano_image_size",
]

_RFDETR_NANO_MEAN = (0.485, 0.456, 0.406)
_RFDETR_NANO_STD = (0.229, 0.224, 0.225)


def _np(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def rfdetr_nano_image_size(cfg: RFDETRConfig | None = None) -> int:
    """Return the RF-DETR Nano square inference resolution."""

    cfg = RFDETRConfig.rfdetr_nano() if cfg is None else cfg
    return int(cfg.backbone.patch_size * cfg.backbone.pretrain_grid)


def _linear_resize_indices(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.float32(in_size / out_size)
    coord = (np.arange(out_size, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5)
    raw0 = np.floor(coord).astype(np.int64)
    raw1 = raw0 + 1
    weight1 = (coord - raw0.astype(np.float32)).astype(np.float32)
    return (
        np.clip(raw0, 0, in_size - 1),
        np.clip(raw1, 0, in_size - 1),
        weight1,
    )


def _resize_bilinear_nchw(x: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"RF-DETR Nano preprocessing expects NCHW input, got {x.shape}")
    _, _, height, width = x.shape
    out_h, out_w = size
    if out_h < height or out_w < width:
        raise ValueError(
            "RF-DETR Nano local capture only implements upstream tensor resize exactly "
            "for same-size or upsampled inputs; downsampling would require antialias filtering."
        )
    y0, y1, wy = _linear_resize_indices(height, out_h)
    x0, x1, wx = _linear_resize_indices(width, out_w)
    top = x[:, :, y0, :] * (1.0 - wy)[None, None, :, None] + x[:, :, y1, :] * wy[None, None, :, None]
    out = top[:, :, :, x0] * (1.0 - wx)[None, None, None, :] + top[:, :, :, x1] * wx[None, None, None, :]
    return out.astype(np.float32, copy=False)


def preprocess_rfdetr_nano_image(image: np.ndarray, image_size: int) -> np.ndarray:
    """Match upstream RF-DETR Nano tensor resize + normalize for local capture."""

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"RF-DETR Nano preprocessing expects an HWC RGB image, got {arr.shape}")
    x = arr.astype(np.float32) / np.float32(255.0)
    x = np.transpose(x, (2, 0, 1))[None]
    x = _resize_bilinear_nchw(x, (int(image_size), int(image_size)))
    mean = np.asarray(_RFDETR_NANO_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(_RFDETR_NANO_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return ((x - mean) / std).astype(np.float32, copy=False)


@dataclass(frozen=True)
class RFDETRLocalCapture:
    input_image: np.ndarray
    input_tensor: np.ndarray
    raw_logits: np.ndarray
    raw_boxes: np.ndarray
    result: Result
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    taps: dict[str, np.ndarray]
    tap_gaps: tuple[str, ...] = ()

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "input_image": self.input_image,
            "input_tensor": self.input_tensor,
            "raw_logits": self.raw_logits,
            "raw_boxes": self.raw_boxes,
            "boxes": self.boxes,
            "scores": self.scores,
            "class_ids": self.class_ids,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "input_shape": list(self.input_image.shape),
            "input_tensor_shape": list(self.input_tensor.shape),
            "raw_logits_shape": list(self.raw_logits.shape),
            "raw_boxes_shape": list(self.raw_boxes.shape),
            "boxes_shape": list(self.boxes.shape),
            "scores_shape": list(self.scores.shape),
            "class_ids_shape": list(self.class_ids.shape),
            "tap_order": list(self.taps),
            "tap_gaps": list(self.tap_gaps),
        }


def build_rfdetr_nano_local_model(weights_path: str | Path) -> RFDETRModel:
    """Build RF-DETR Nano and strict-load converted local weights."""

    model = RFDETRModel(RFDETRConfig.rfdetr_nano())
    return load_rfdetr_weights(model, weights_path, strict=True)


def capture_rfdetr_nano_local(
    model: RFDETRModel,
    *,
    image: np.ndarray | None = None,
    image_size: int | None = None,
) -> RFDETRLocalCapture:
    """Run local RF-DETR Nano on the canonical fixed source image."""

    cfg = model.cfg
    image_size = rfdetr_nano_image_size(cfg) if image_size is None else int(image_size)
    processor = RFDETRProcessor(
        RFDETRProcessorConfig(
            image_size=image_size,
            top_k=int(cfg.decoder.num_queries),
            score_threshold=0.0,
        )
    )
    input_image = rfdetr_fixed_image() if image is None else np.asarray(image)
    input_tensor = preprocess_rfdetr_nano_image(input_image, image_size)
    ctx = SpatialTransform.resize(tuple(input_image.shape[:2]), (image_size, image_size))
    raw = model(mx.array(input_tensor), capture_taps=True)
    mx.eval(raw.data)
    result = processor.postprocess(raw, ctx)
    if result.detections is None:
        raise RuntimeError("RF-DETR local capture did not produce detections")

    taps = {key: _np(value) for key, value in raw["taps"].items()}
    taps["result.boxes"] = np.asarray(result.detections.boxes, dtype=np.float64)
    taps["result.scores"] = np.asarray(result.detections.scores, dtype=np.float64)
    taps["result.class_ids"] = np.asarray(result.detections.class_ids, dtype=np.int64)

    return RFDETRLocalCapture(
        input_image=np.asarray(input_image),
        input_tensor=input_tensor,
        raw_logits=_np(raw["logits"]),
        raw_boxes=_np(raw["boxes"]),
        result=result,
        boxes=result.detections.boxes,
        scores=np.asarray(result.detections.scores, dtype=np.float64),
        class_ids=np.asarray(result.detections.class_ids, dtype=np.int64),
        taps=taps,
        tap_gaps=(
            "backbone.windowed_dinov2: final RFDETRModel capture exposes stable "
            "projector/decoder/head taps; per-block backbone taps are not propagated.",
        ),
    )
