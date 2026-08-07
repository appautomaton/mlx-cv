"""Image preprocessing and panoptic postprocessing for EoMT-DINOv3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from ...core.base import Processor
from ...core.features import HeadOutput
from ...core.image import load_image
from ...core.types import Masks, Result
from .config import EoMTDINOv3Config

__all__ = [
    "EoMTDINOv3Processor",
    "EoMTDINOv3ProcessorConfig",
    "EoMTDINOv3ProcessorContext",
]


@dataclass(frozen=True)
class EoMTDINOv3ProcessorConfig:
    image_size: int = 640
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    threshold: float = 0.8
    mask_threshold: float = 0.5
    overlap_mask_area_threshold: float = 0.8
    stuff_classes: tuple[int, ...] = tuple(range(80, 133))
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("EoMT processor image_size must be positive")
        if len(self.mean) != 3 or len(self.std) != 3 or any(value == 0 for value in self.std):
            raise ValueError("EoMT processor mean/std must contain three non-zero channels")
        for name, value in (
            ("threshold", self.threshold),
            ("mask_threshold", self.mask_threshold),
            ("overlap_mask_area_threshold", self.overlap_mask_area_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"EoMT processor {name} must be between 0 and 1")


@dataclass(frozen=True)
class EoMTDINOv3ProcessorContext:
    image_size: tuple[int, int]
    resized_size: tuple[int, int]
    model_size: tuple[int, int]


def _resize_axis_half_pixel(x: np.ndarray, out_size: int, axis: int) -> np.ndarray:
    in_size = x.shape[axis]
    if in_size == out_size:
        return x
    coordinates = (np.arange(out_size, dtype=np.float32) + 0.5) * (in_size / out_size) - 0.5
    lower_raw = np.floor(coordinates).astype(np.int64)
    upper_raw = lower_raw + 1
    lower = np.clip(lower_raw, 0, in_size - 1)
    upper = np.clip(upper_raw, 0, in_size - 1)
    weight = coordinates - lower_raw.astype(np.float32)
    left = np.take(x, lower, axis=axis)
    right = np.take(x, upper, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = out_size
    weight = weight.reshape(shape)
    return left * (1.0 - weight) + right * weight


def _resize_bilinear_nchw(x: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    x = _resize_axis_half_pixel(x, int(size[0]), axis=2)
    return _resize_axis_half_pixel(x, int(size[1]), axis=3)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


class EoMTDINOv3Processor(Processor):
    """One-image processor returning normalized NCHW input and panoptic `Result`."""

    def __init__(self, config: EoMTDINOv3ProcessorConfig | None = None) -> None:
        self.config = config or EoMTDINOv3ProcessorConfig()

    @classmethod
    def from_model_config(
        cls,
        model_config: EoMTDINOv3Config,
        **overrides,
    ) -> "EoMTDINOv3Processor":
        values = {
            "image_size": model_config.image_size,
            "stuff_classes": model_config.stuff_classes,
            "labels": model_config.labels,
        }
        values.update(overrides)
        return cls(EoMTDINOv3ProcessorConfig(**values))

    def preprocess(self, inputs: Any) -> tuple[dict[str, mx.array], EoMTDINOv3ProcessorContext]:
        image = inputs.get("image") if isinstance(inputs, dict) else inputs
        if image is None:
            raise ValueError("EoMTDINOv3Processor.preprocess requires an image")
        arr, image_size = load_image(image)
        height, width = image_size
        scale = min(self.config.image_size / height, self.config.image_size / width)
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))

        from PIL import Image

        resized = np.asarray(
            Image.fromarray(arr).resize((resized_width, resized_height), Image.BILINEAR)
        )
        canvas = np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
        canvas[:resized_height, :resized_width] = resized
        values = canvas.astype(np.float32) / 255.0
        mean = np.asarray(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.config.std, dtype=np.float32).reshape(1, 1, 3)
        values = np.transpose((values - mean) / std, (2, 0, 1))[None]
        context = EoMTDINOv3ProcessorContext(
            image_size=image_size,
            resized_size=(resized_height, resized_width),
            model_size=(self.config.image_size, self.config.image_size),
        )
        return {"pixel_values": mx.array(np.ascontiguousarray(values))}, context

    @staticmethod
    def _raw_dict(raw: Any) -> dict[str, Any]:
        if isinstance(raw, HeadOutput):
            return raw.data
        if isinstance(raw, dict):
            return raw
        data = getattr(raw, "data", None)
        if isinstance(data, dict):
            return data
        raise TypeError("EoMT postprocess expects HeadOutput or a tensor dictionary")

    def postprocess(self, raw: Any, ctx: EoMTDINOv3ProcessorContext) -> Result:
        data = self._raw_dict(raw)
        mask_logits = data.get("masks_queries_logits", data.get("mask_logits"))
        class_logits = data.get("class_queries_logits", data.get("class_logits"))
        if mask_logits is None or class_logits is None:
            raise ValueError("EoMT postprocess requires mask and class query logits")
        masks = np.asarray(mask_logits, dtype=np.float32)
        classes = np.asarray(class_logits, dtype=np.float32)
        if masks.ndim != 4 or classes.ndim != 3 or masks.shape[:2] != classes.shape[:2]:
            raise ValueError(
                "EoMT logits must have shapes (B,Q,H,W) and (B,Q,C+1), got "
                f"{masks.shape} and {classes.shape}"
            )
        if masks.shape[0] != 1:
            raise ValueError("EoMTDINOv3Processor currently postprocesses one image at a time")

        probabilities = _softmax(classes[0], axis=-1)
        pred_labels = probabilities.argmax(axis=-1)
        pred_scores = probabilities.max(axis=-1)
        num_classes = classes.shape[-1] - 1
        keep = (pred_labels != num_classes) & (pred_scores > self.config.threshold)
        kept_labels = pred_labels[keep].astype(np.int64)
        kept_scores = pred_scores[keep]
        kept_masks = masks[0, keep]

        segmentation = np.full(ctx.image_size, -1, dtype=np.int64)
        segments_info: list[dict[str, Any]] = []
        if len(kept_masks):
            resized = _resize_bilinear_nchw(kept_masks[None], ctx.model_size)[0]
            resized = resized[:, : ctx.resized_size[0], : ctx.resized_size[1]]
            resized = _resize_bilinear_nchw(resized[None], ctx.image_size)[0]
            mask_probs = 1.0 / (1.0 + np.exp(-resized))
            assignments = (kept_scores[:, None, None] * mask_probs).argmax(axis=0)
            stuff_memory: dict[int, int] = {}

            for query_index, class_id in enumerate(kept_labels):
                original_mask = mask_probs[query_index] >= self.config.mask_threshold
                assigned_mask = assignments == query_index
                final_mask = original_mask & assigned_mask
                original_area = int(original_mask.sum())
                assigned_area = int(assigned_mask.sum())
                final_area = int(final_mask.sum())
                if (
                    original_area == 0
                    or assigned_area == 0
                    or final_area == 0
                    or assigned_area / original_area <= self.config.overlap_mask_area_threshold
                ):
                    continue

                class_id_int = int(class_id)
                if class_id_int in self.config.stuff_classes and class_id_int in stuff_memory:
                    segmentation[final_mask] = stuff_memory[class_id_int]
                    continue

                segment_id = len(segments_info)
                if class_id_int in self.config.stuff_classes:
                    stuff_memory[class_id_int] = segment_id
                segmentation[final_mask] = segment_id
                segment = {
                    "id": segment_id,
                    "label_id": class_id_int,
                    "score": round(float(kept_scores[query_index]), 6),
                }
                if self.config.labels is not None:
                    if class_id_int >= len(self.config.labels):
                        raise ValueError(f"EoMT class id {class_id_int} has no configured label")
                    segment["label"] = self.config.labels[class_id_int]
                segments_info.append(segment)

        labels = None
        if self.config.labels is not None:
            labels = [str(segment["label"]) for segment in segments_info]
        return Result(
            image_size=ctx.image_size,
            masks=Masks(segmentation, kind="panoptic", labels=labels),
            metadata={
                "segments_info": segments_info,
                "threshold": self.config.threshold,
                "mask_threshold": self.config.mask_threshold,
                "overlap_mask_area_threshold": self.config.overlap_mask_area_threshold,
            },
        )
