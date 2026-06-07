"""Pure numpy box ops: format conversion, IoU, NMS, clipping."""

from __future__ import annotations

import numpy as np

__all__ = ["box_convert", "box_iou", "nms", "clip_boxes"]

_FORMATS = ("xyxy", "xywh", "cxcywh")


def box_convert(boxes, in_fmt: str, out_fmt: str) -> np.ndarray:
    """Convert between ``xyxy`` / ``xywh`` / ``cxcywh`` ``(N, 4)`` boxes."""
    if in_fmt not in _FORMATS or out_fmt not in _FORMATS:
        raise ValueError(f"formats must be one of {_FORMATS}")
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if in_fmt == "xyxy":
        xyxy = b.copy()
    elif in_fmt == "xywh":
        xyxy = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)
    else:  # cxcywh
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
    if out_fmt == "xyxy":
        return xyxy
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    if out_fmt == "xywh":
        return np.stack([xyxy[:, 0], xyxy[:, 1], w, h], axis=1)
    return np.stack([xyxy[:, 0] + w / 2, xyxy[:, 1] + h / 2, w, h], axis=1)  # cxcywh


def box_iou(a, b) -> np.ndarray:
    """Pairwise IoU between ``a`` ``(N,4)`` and ``b`` ``(M,4)`` xyxy -> ``(N, M)``."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(boxes, scores, iou_threshold: float = 0.5) -> np.ndarray:
    """Greedy non-max suppression; returns kept indices (highest score first)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def clip_boxes(boxes, size_hw: tuple[int, int]) -> np.ndarray:
    """Clip ``xyxy`` boxes to image bounds ``(H, W)``."""
    h, w = size_hw
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, 0::2] = np.clip(b[:, 0::2], 0, w)
    b[:, 1::2] = np.clip(b[:, 1::2], 0, h)
    return b
