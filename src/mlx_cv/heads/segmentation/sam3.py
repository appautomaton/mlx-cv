"""SAM 3.1 PCS geometry prompt encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.geometry import SpatialTransform
from ...prompts import BoxPrompt, ExemplarPrompt, PointPrompt

__all__ = ["SAM3EncodedGeometryPrompt", "SAM3PCSPromptEncoder"]


@dataclass(frozen=True)
class SAM3EncodedGeometryPrompt:
    boxes_cxcywh: np.ndarray
    box_labels: np.ndarray
    exemplar_boxes_cxcywh: np.ndarray | None = None
    exemplar_labels: np.ndarray | None = None


def _xyxy_to_cxcywh_norm(boxes: np.ndarray, *, model_size: tuple[int, int]) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    h, w = model_size
    x0, y0, x1, y1 = boxes.T if len(boxes) else [np.array([], dtype=np.float64)] * 4
    out = np.stack(
        [
            (x0 + x1) * 0.5 / w,
            (y0 + y1) * 0.5 / h,
            (x1 - x0) / w,
            (y1 - y0) / h,
        ],
        axis=-1,
    )
    return np.clip(out, 0.0, 1.0)


class SAM3PCSPromptEncoder:
    """Encode SAM3 image-mode PCS boxes into normalized model-space prompts."""

    def __init__(self, model_size: tuple[int, int]) -> None:
        h, w = int(model_size[0]), int(model_size[1])
        if h <= 0 or w <= 0:
            raise ValueError("SAM3 PCS prompt encoder model_size must be positive")
        self.model_size = (h, w)

    def encode_boxes(
        self,
        prompt: BoxPrompt,
        transform: SpatialTransform,
        *,
        labels: np.ndarray | list[bool] | list[int] | None = None,
    ) -> SAM3EncodedGeometryPrompt:
        boxes = transform.apply_boxes(prompt.boxes)
        encoded = _xyxy_to_cxcywh_norm(boxes, model_size=self.model_size)
        if labels is None:
            label_arr = np.ones((encoded.shape[0],), dtype=np.bool_)
        else:
            label_arr = np.asarray(labels, dtype=np.bool_).reshape(-1)
            if len(label_arr) != encoded.shape[0]:
                raise ValueError(f"SAM3 box labels length {len(label_arr)} does not match {encoded.shape[0]} boxes")
        return SAM3EncodedGeometryPrompt(encoded, label_arr)

    def encode_exemplar(
        self,
        prompt: ExemplarPrompt,
        *,
        exemplar_transform: SpatialTransform | None = None,
    ) -> SAM3EncodedGeometryPrompt:
        if exemplar_transform is None:
            exemplar_transform = SpatialTransform.resize(prompt.image.shape[:2], self.model_size)
        boxes = exemplar_transform.apply_boxes(prompt.boxes)
        encoded = _xyxy_to_cxcywh_norm(boxes, model_size=self.model_size)
        labels = np.ones((encoded.shape[0],), dtype=np.bool_)
        return SAM3EncodedGeometryPrompt(
            boxes_cxcywh=np.zeros((0, 4), dtype=np.float64),
            box_labels=np.zeros((0,), dtype=np.bool_),
            exemplar_boxes_cxcywh=encoded,
            exemplar_labels=labels,
        )

    def encode(self, prompt, transform: SpatialTransform) -> SAM3EncodedGeometryPrompt:
        if isinstance(prompt, BoxPrompt):
            return self.encode_boxes(prompt, transform)
        if isinstance(prompt, ExemplarPrompt):
            return self.encode_exemplar(prompt)
        if isinstance(prompt, PointPrompt):
            raise NotImplementedError("SAM 3.1 PCS grounding does not support PointPrompt; interactive points are deferred")
        raise TypeError(f"unsupported SAM3 geometry prompt type: {type(prompt).__name__}")
