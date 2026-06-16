"""The output lingua franca — one ``Result`` container for every task.

All tasks return the same ``Result``; modalities are optional, composable fields
(a panoptic+depth model and a plain detector share one surface). See
ARCHITECTURE.md §5.1. Data is numpy-backed for interop (COCO / supervision);
tensor compute (mlx) stays in the model and is converted at the boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Detections", "Masks", "Keypoints", "Points", "DepthMap",
    "Embedding", "Tracks", "Result",
]


def _arr(x, dtype=np.float64):
    return None if x is None else np.asarray(x, dtype=dtype)


@dataclass
class Detections:
    """Axis-aligned boxes in ``xyxy`` pixel coords, with optional metadata.

    ``scores`` may be ``None`` (e.g. LocateAnything emits no per-box score, §16).
    """

    boxes: np.ndarray                      # (N, 4) xyxy
    scores: np.ndarray | None = None       # (N,)
    labels: list[str] | None = None        # (N,) class names
    class_ids: np.ndarray | None = None    # (N,) int
    track_ids: np.ndarray | None = None    # (N,) int

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        n = len(self.boxes)
        self.scores = _arr(self.scores)
        self.class_ids = _arr(self.class_ids, np.int64)
        self.track_ids = _arr(self.track_ids, np.int64)
        for name, val in (("scores", self.scores), ("class_ids", self.class_ids),
                          ("track_ids", self.track_ids), ("labels", self.labels)):
            if val is not None and len(val) != n:
                raise ValueError(f"Detections.{name} has length {len(val)}, expected {n}")

    def __len__(self) -> int:
        return len(self.boxes)


@dataclass
class Points:
    """Sparse 2D localization points (pointing / GUI) — *not* skeletal keypoints (§16)."""

    points: np.ndarray                     # (N, 2) xy
    scores: np.ndarray | None = None
    labels: list[str] | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        self.scores = _arr(self.scores)

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class Keypoints:
    """Named skeletons: ``(N, K, 2|3)`` (xy + optional confidence) for pose."""

    keypoints: np.ndarray
    skeleton: list[tuple[int, int]] | None = None
    names: list[str] | None = None

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float64)


@dataclass
class Masks:
    """Instance ``(N,H,W)`` or semantic/panoptic ``(H,W)`` masks."""

    data: np.ndarray
    kind: str = "instance"                 # instance | semantic | panoptic
    labels: list[str] | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.labels is not None and self.kind == "instance":
            if self.data.ndim != 3:
                raise ValueError(
                    "Masks.labels for instance masks requires data shape (N,H,W)"
                )
            n = self.data.shape[0]
            if len(self.labels) != n:
                raise ValueError(f"Masks.labels has length {len(self.labels)}, expected {n}")


@dataclass
class DepthMap:
    """Per-pixel depth ``(H, W)`` plus optional confidence; metric or relative."""

    depth: np.ndarray
    depth_conf: np.ndarray | None = None
    metric: bool = False
    units: str | None = None               # e.g. "m"
    focal_px: float | None = None

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth, dtype=np.float64)
        self.depth_conf = _arr(self.depth_conf)
        if self.depth_conf is not None and self.depth_conf.shape != self.depth.shape:
            raise ValueError(
                f"DepthMap.depth_conf shape {self.depth_conf.shape} must match depth shape {self.depth.shape}"
            )


@dataclass
class Embedding:
    """Feature vector ``(D,)`` or dense feature map ``(H, W, D)``."""

    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)


@dataclass
class Tracks:
    """Temporal identities for the current frame."""

    ids: np.ndarray                        # (N,) int
    frame_index: int | None = None

    def __post_init__(self) -> None:
        self.ids = np.asarray(self.ids, dtype=np.int64).reshape(-1)


@dataclass
class Result:
    """One container, optional fields. Every spatial field is in original-image coords."""

    image_size: tuple[int, int]            # original (H, W)
    detections: Detections | None = None
    masks: Masks | None = None
    keypoints: Keypoints | None = None
    points: Points | None = None
    depth: DepthMap | None = None
    embedding: Embedding | None = None
    tracks: Tracks | None = None

    def draw(self, image=None, **opts):
        """Annotate ``image`` with this result. Lands with the first model (viz/)."""
        raise NotImplementedError(
            "Result.draw() ships with the viz/ annotators in a later release; "
            "v0.0.2 is the spine scaffold (no models, no rendering)."
        )

    def to_coco(self, image_id: int = 0) -> dict:
        """COCO-style ``{image_id, annotations:[...]}`` from ``detections`` (bbox = xywh)."""
        anns: list[dict] = []
        d = self.detections
        if d is not None:
            for i in range(len(d)):
                x0, y0, x1, y1 = (float(v) for v in d.boxes[i])
                anns.append({
                    "image_id": image_id,
                    "id": i,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "score": None if d.scores is None else float(d.scores[i]),
                    "category_id": None if d.class_ids is None else int(d.class_ids[i]),
                    "category_name": None if d.labels is None else d.labels[i],
                    "iscrowd": 0,
                })
        return {"image_id": image_id, "annotations": anns}

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (arrays -> lists). Covers boxes/points in v0.0.2."""
        out: dict[str, Any] = {"image_size": list(self.image_size)}
        if self.detections is not None:
            d = self.detections
            out["detections"] = {
                "boxes": d.boxes.tolist(),
                "scores": None if d.scores is None else d.scores.tolist(),
                "labels": d.labels,
                "class_ids": None if d.class_ids is None else d.class_ids.tolist(),
                "track_ids": None if d.track_ids is None else d.track_ids.tolist(),
            }
        if self.masks is not None:
            m = self.masks
            out["masks"] = {
                "data": m.data.tolist(),
                "shape": list(m.data.shape),
                "kind": m.kind,
                "labels": m.labels,
            }
        if self.points is not None:
            p = self.points
            out["points"] = {
                "points": p.points.tolist(),
                "scores": None if p.scores is None else p.scores.tolist(),
                "labels": p.labels,
            }
        if self.depth is not None:
            d = self.depth
            out["depth"] = {
                "depth": d.depth.tolist(),
                "depth_conf": None if d.depth_conf is None else d.depth_conf.tolist(),
                "metric": d.metric,
                "units": d.units,
                "focal_px": d.focal_px,
            }
        return out

    def save(self, path) -> None:
        """Write :meth:`to_dict` as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
