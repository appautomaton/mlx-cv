"""SAM 3.1 image preprocessing and mask postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from ...core.base import Processor
from ...core.features import HeadOutput
from ...core.geometry import SpatialTransform
from ...core.image import load_image
from ...core.types import Detections, Masks, Result
from ...transforms.resize import Resize
from .prompts import SAM3PreparedPrompt, prepare_sam3_prompt

__all__ = ["SAM3Processor", "SAM3ProcessorConfig", "SAM3ProcessorContext"]


def _as_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    return (int(size[0]), int(size[1]))


@dataclass(frozen=True)
class SAM3ProcessorConfig:
    image_size: int | tuple[int, int] = 1024
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    mask_threshold: float = 0.5
    score_threshold: float = 0.0
    top_k: int = 64
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if min(_as_hw(self.image_size)) <= 0:
            raise ValueError("SAM3 image_size must be positive")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("SAM3 mean/std must each contain 3 channels")
        if any(s == 0 for s in self.std):
            raise ValueError("SAM3 std values must be non-zero")
        if self.top_k <= 0:
            raise ValueError("SAM3 top_k must be positive")

    @property
    def model_size(self) -> tuple[int, int]:
        return _as_hw(self.image_size)


@dataclass(frozen=True)
class SAM3ProcessorContext:
    transform: SpatialTransform
    image_size: tuple[int, int]
    model_size: tuple[int, int]
    prompt: SAM3PreparedPrompt


def _prompt_from_inputs(inputs: dict[str, Any]) -> Any:
    if "prompt" in inputs:
        return inputs["prompt"]
    prompt_keys = (
        "text",
        "box",
        "boxes",
        "exemplar",
        "exemplar_image",
        "exemplar_boxes",
        "point",
        "points",
        "mask",
        "mask_prompt",
        "video",
        "video_state",
        "tracker",
        "tracker_state",
    )
    prompt = {key: inputs[key] for key in prompt_keys if key in inputs}
    return prompt or None


def _raw_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, HeadOutput):
        return raw.data
    if isinstance(raw, dict):
        return raw
    data = getattr(raw, "data", None)
    if isinstance(data, dict):
        return data
    raise TypeError("SAM3Processor.postprocess expects HeadOutput or a dict of raw tensors")


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _resize_mask_nearest(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    out_h, out_w = size
    if mask.shape == (out_h, out_w):
        return mask
    y_idx = np.minimum(np.floor(np.arange(out_h) * (mask.shape[0] / out_h)).astype(np.int64), mask.shape[0] - 1)
    x_idx = np.minimum(np.floor(np.arange(out_w) * (mask.shape[1] / out_w)).astype(np.int64), mask.shape[1] - 1)
    return mask[y_idx[:, None], x_idx[None, :]]


class SAM3Processor(Processor):
    """One-image SAM3 processor returning NCHW tensors and `Result.masks`."""

    def __init__(self, config: SAM3ProcessorConfig | None = None) -> None:
        self.config = config or SAM3ProcessorConfig()

    def preprocess(self, inputs: Any) -> tuple[dict[str, Any], SAM3ProcessorContext]:
        if isinstance(inputs, dict):
            image = inputs.get("image")
            prompt = _prompt_from_inputs(inputs)
        else:
            image = inputs
            prompt = None
        if image is None:
            raise ValueError("SAM3Processor.preprocess requires an image")

        arr, image_size = load_image(image)
        resized, transform = Resize(self.config.model_size)(arr)
        x = resized.astype(np.float32) / 255.0
        mean = np.asarray(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.config.std, dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None]
        prepared = (
            prompt
            if isinstance(prompt, SAM3PreparedPrompt)
            else prepare_sam3_prompt(prompt, transform=transform, model_size=self.config.model_size)
        )
        ctx = SAM3ProcessorContext(
            transform=transform,
            image_size=image_size,
            model_size=self.config.model_size,
            prompt=prepared,
        )
        return {"pixel_values": mx.array(np.ascontiguousarray(x)), "prompt": prepared}, ctx

    def _labels_for(self, class_ids: np.ndarray) -> list[str] | None:
        if self.config.labels is None:
            return None
        labels = []
        for class_id in class_ids:
            cid = int(class_id)
            if cid < 0 or cid >= len(self.config.labels):
                raise ValueError(f"SAM3 class id {cid} has no configured label")
            labels.append(self.config.labels[cid])
        return labels

    def postprocess(self, raw: Any, ctx: SAM3ProcessorContext | SpatialTransform) -> Result:
        if isinstance(ctx, SpatialTransform):
            context = SAM3ProcessorContext(
                transform=ctx,
                image_size=ctx.orig_size,
                model_size=ctx.model_size or ctx.orig_size,
                prompt=SAM3PreparedPrompt(texts=(), geometry=None),
            )
        else:
            context = ctx

        data = _raw_dict(raw)
        mask_logits = data.get("pred_masks", data.get("mask_logits"))
        if mask_logits is None:
            raise ValueError("SAM3 postprocess requires mask_logits/pred_masks")
        masks_np = np.asarray(mask_logits, dtype=np.float64)
        if masks_np.ndim != 4:
            raise ValueError(f"SAM3 mask logits must have shape (B,Q,H,W), got {masks_np.shape}")
        if masks_np.shape[0] != 1:
            raise ValueError("SAM3Processor currently supports one image per Result")

        scores_np = data.get("object_scores", data.get("scores"))
        if scores_np is None:
            scores = np.ones((masks_np.shape[1],), dtype=np.float64)
        else:
            scores_arr = np.asarray(scores_np, dtype=np.float64)
            if scores_arr.ndim == 2:
                scores = scores_arr[0]
            elif scores_arr.ndim == 1:
                scores = scores_arr
            else:
                raise ValueError(f"SAM3 scores must have shape (Q,) or (B,Q), got {scores_arr.shape}")
        if len(scores) != masks_np.shape[1]:
            raise ValueError("SAM3 scores length must match mask query count")

        labels_raw = data.get("labels", None)
        if labels_raw is None:
            class_ids = np.zeros((masks_np.shape[1],), dtype=np.int64)
        else:
            labels_arr = np.asarray(labels_raw, dtype=np.int64)
            class_ids = labels_arr[0] if labels_arr.ndim == 2 else labels_arr
        if len(class_ids) != masks_np.shape[1]:
            raise ValueError("SAM3 labels length must match mask query count")

        order = np.argsort(scores)[::-1][: min(self.config.top_k, len(scores))]
        keep = scores[order] >= self.config.score_threshold
        selected = order[keep]
        selected_scores = scores[selected]
        selected_class_ids = class_ids[selected]
        selected_labels = self._labels_for(selected_class_ids)

        instance_masks = []
        probs = _sigmoid_np(masks_np[0, selected])
        for prob in probs:
            model_mask = _resize_mask_nearest(prob >= self.config.mask_threshold, context.model_size)
            instance_masks.append(context.transform.invert_mask(model_mask.astype(np.uint8), fill=0).astype(bool))
        if instance_masks:
            mask_data = np.stack(instance_masks, axis=0)
        else:
            h, w = context.image_size
            mask_data = np.zeros((0, h, w), dtype=bool)
        masks = Masks(mask_data, kind="instance", labels=selected_labels)

        detections = None
        boxes = data.get("boxes", data.get("pred_boxes"))
        if boxes is not None:
            boxes_np = np.asarray(boxes, dtype=np.float64)
            if boxes_np.ndim != 3 or boxes_np.shape[0] != 1 or boxes_np.shape[1] != masks_np.shape[1] or boxes_np.shape[-1] != 4:
                raise ValueError(f"SAM3 boxes must have shape (1,Q,4) matching masks, got {boxes_np.shape}")
            selected_boxes = boxes_np[0, selected]
            if len(selected_boxes):
                cx, cy, bw, bh = [selected_boxes[:, i] for i in range(4)]
                xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=-1)
                model_h, model_w = context.model_size
                scale = np.asarray([model_w, model_h, model_w, model_h], dtype=np.float64)
                model_boxes = xyxy * scale
                orig_boxes = context.transform.invert_boxes(model_boxes, clip=True)
            else:
                orig_boxes = np.zeros((0, 4), dtype=np.float64)
            detections = Detections(
                orig_boxes,
                scores=selected_scores,
                class_ids=selected_class_ids,
                labels=selected_labels,
            )

        return Result(image_size=context.image_size, masks=masks, detections=detections)
