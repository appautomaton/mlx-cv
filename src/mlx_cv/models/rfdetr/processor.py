"""RF-DETR image preprocessing and detection postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from ...core.base import Processor
from ...core.features import HeadOutput
from ...core.geometry import SpatialTransform
from ...core.image import load_image
from ...core.types import Detections, Result
from ...transforms.resize import Resize

__all__ = [
    "RFDETRProcessor",
    "RFDETRProcessorConfig",
    "RFDETRProcessorContext",
]


def _as_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    return (int(size[0]), int(size[1]))


@dataclass(frozen=True)
class RFDETRProcessorConfig:
    image_size: int | tuple[int, int] = 560
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    top_k: int = 300
    score_threshold: float = 0.0
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if min(_as_hw(self.image_size)) <= 0:
            raise ValueError("RF-DETR image_size must be positive")
        if self.top_k <= 0:
            raise ValueError("RF-DETR top_k must be positive")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("RF-DETR mean/std must each contain 3 channels")
        if any(s == 0 for s in self.std):
            raise ValueError("RF-DETR std values must be non-zero")

    @property
    def model_size(self) -> tuple[int, int]:
        return _as_hw(self.image_size)


@dataclass(frozen=True)
class RFDETRProcessorContext:
    transform: SpatialTransform
    image_size: tuple[int, int]
    model_size: tuple[int, int]


class RFDETRProcessor(Processor):
    """One-image RF-DETR processor returning NCHW tensors and `Result.detections`."""

    def __init__(self, config: RFDETRProcessorConfig | None = None) -> None:
        self.config = config or RFDETRProcessorConfig()

    def preprocess(self, inputs: Any) -> tuple[dict[str, mx.array], RFDETRProcessorContext]:
        image = inputs.get("image") if isinstance(inputs, dict) else inputs
        if image is None:
            raise ValueError("RFDETRProcessor.preprocess requires an image")
        arr, image_size = load_image(image)
        resized, transform = Resize(self.config.model_size)(arr)
        x = resized.astype(np.float32) / 255.0
        mean = np.asarray(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.config.std, dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None]
        ctx = RFDETRProcessorContext(
            transform=transform,
            image_size=image_size,
            model_size=self.config.model_size,
        )
        return {"pixel_values": mx.array(np.ascontiguousarray(x))}, ctx

    def _raw_dict(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, HeadOutput):
            return raw.data
        if isinstance(raw, dict):
            return raw
        data = getattr(raw, "data", None)
        if isinstance(data, dict):
            return data
        raise TypeError("RFDETRProcessor.postprocess expects HeadOutput or a dict of raw tensors")

    def _labels_for(self, class_ids: np.ndarray) -> list[str] | None:
        if self.config.labels is None:
            return None
        labels = []
        for class_id in class_ids:
            cid = int(class_id)
            if cid < 0 or cid >= len(self.config.labels):
                raise ValueError(f"RF-DETR class id {cid} has no configured label")
            labels.append(self.config.labels[cid])
        return labels

    def postprocess(self, raw: Any, ctx: RFDETRProcessorContext | SpatialTransform) -> Result:
        if isinstance(ctx, SpatialTransform):
            context = RFDETRProcessorContext(
                transform=ctx,
                image_size=ctx.orig_size,
                model_size=ctx.model_size or ctx.orig_size,
            )
        else:
            context = ctx

        data = self._raw_dict(raw)
        logits = data.get("pred_logits", data.get("logits"))
        boxes = data.get("pred_boxes", data.get("boxes"))
        if logits is None or boxes is None:
            raise ValueError("RF-DETR postprocess requires logits/pred_logits and boxes/pred_boxes")

        logits_np = np.asarray(logits, dtype=np.float64)
        boxes_np = np.asarray(boxes, dtype=np.float64)
        if logits_np.ndim != 3:
            raise ValueError(f"RF-DETR logits must have shape (B,Q,C), got {logits_np.shape}")
        if boxes_np.ndim != 3 or boxes_np.shape[:2] != logits_np.shape[:2] or boxes_np.shape[-1] != 4:
            raise ValueError(
                f"RF-DETR boxes must have shape (B,Q,4) matching logits, got {boxes_np.shape}"
            )
        if logits_np.shape[0] != 1:
            raise ValueError("RFDETRProcessor currently supports one image per Result")

        probs = 1.0 / (1.0 + np.exp(-logits_np[0]))
        flat = probs.reshape(-1)
        count = min(self.config.top_k, flat.size)
        order = np.argsort(flat)[::-1][:count]
        scores = flat[order]
        keep = scores >= self.config.score_threshold
        order = order[keep]
        scores = scores[keep]

        num_classes = logits_np.shape[-1]
        query_ids = (order // num_classes).astype(np.int64)
        class_ids = (order % num_classes).astype(np.int64)
        selected = boxes_np[0, query_ids] if len(query_ids) else np.zeros((0, 4), dtype=np.float64)

        cx, cy, w, h = [selected[:, i] if len(selected) else np.array([], dtype=np.float64) for i in range(4)]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)
        model_h, model_w = context.model_size
        scale = np.asarray([model_w, model_h, model_w, model_h], dtype=np.float64)
        model_boxes = xyxy * scale
        orig_boxes = context.transform.invert_boxes(model_boxes, clip=False)

        detections = Detections(
            orig_boxes,
            scores=scores,
            class_ids=class_ids,
            labels=self._labels_for(class_ids),
        )
        return Result(image_size=context.image_size, detections=detections)
