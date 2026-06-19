"""Faithful SAM3 Object-Multiplex detector<->tracker association (slice 15).

Ports the single-device core of the research predictor's planning phase
(``references/sam3`` ``sam3_video_base._associate_det_trk_compilable`` + hotstart
keep-alive): many-to-one mask-IoU matching, new-object spawning from unmatched
high-score detections, and keep-alive removal of tracks that stay unmatched. These are
config-driven heuristics (no weights); numeric parity is the deferred out-of-sandbox
gate. No torch/transformers imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

__all__ = [
    "Sam3AssociationConfig",
    "AssociationResult",
    "mask_iou",
    "associate_detections",
    "Sam3TrackKeepAlive",
]


@dataclass(frozen=True)
class Sam3AssociationConfig:
    iou_threshold: float = 0.5  # det<->trk overlap above which a detection is "explained" by a track
    iou_threshold_trk: float = 0.5  # overlap above which a track counts as "matched"
    new_detection_score_threshold: float = 0.5  # min detection score to spawn a new object
    max_unmatched_frames: int = 3  # keep-alive: remove a track after this many consecutive unmatched frames


@dataclass
class AssociationResult:
    is_new_detection: mx.array  # [N_det] bool — spawn a new object
    track_is_unmatched: mx.array  # [M_trk] bool — nonempty track with no matching detection
    detection_to_track: mx.array  # [N_det] int — best-IoU track index per detection (-1 when no tracks)


def mask_iou(masks_a: mx.array, masks_b: mx.array) -> mx.array:
    """IoU between two sets of binary masks: ``[A,H,W]``, ``[B,H,W]`` -> ``[A,B]``."""
    a = (masks_a > 0).reshape(masks_a.shape[0], -1).astype(mx.float32)
    b = (masks_b > 0).reshape(masks_b.shape[0], -1).astype(mx.float32)
    intersection = a @ b.T
    area_a = a.sum(axis=1, keepdims=True)
    area_b = b.sum(axis=1, keepdims=True).T
    union = area_a + area_b - intersection
    return intersection / mx.maximum(union, mx.array(1e-6))


def associate_detections(
    detection_masks: mx.array,  # [N_det, H, W]
    detection_scores: mx.array,  # [N_det]
    track_masks: mx.array,  # [M_trk, H, W]
    config: Sam3AssociationConfig,
) -> AssociationResult:
    """Many-to-one mask-IoU association (faithful core of ``_associate_det_trk_compilable``).

    A track is *matched* if any detection overlaps it >= ``iou_threshold_trk``; a detection is
    *new* if its score clears the threshold and it overlaps no track >= ``iou_threshold``.
    """
    num_det = int(detection_masks.shape[0])
    num_trk = int(track_masks.shape[0])
    score_ok = detection_scores >= config.new_detection_score_threshold
    if num_trk == 0:
        return AssociationResult(
            is_new_detection=score_ok if num_det else mx.zeros((0,), dtype=mx.bool_),
            track_is_unmatched=mx.zeros((0,), dtype=mx.bool_),
            detection_to_track=mx.full((num_det,), -1, dtype=mx.int32) if num_det else mx.zeros((0,), dtype=mx.int32),
        )
    if num_det == 0:
        track_nonempty = (track_masks > 0).reshape(num_trk, -1).any(axis=1)
        return AssociationResult(
            is_new_detection=mx.zeros((0,), dtype=mx.bool_),
            track_is_unmatched=track_nonempty,  # no detections -> every nonempty track is unmatched
            detection_to_track=mx.zeros((0,), dtype=mx.int32),
        )
    iou = mask_iou(detection_masks, track_masks)  # [N_det, M_trk]
    track_matched = (iou >= config.iou_threshold_trk).any(axis=0)
    track_nonempty = (track_masks > 0).reshape(num_trk, -1).any(axis=1)
    track_is_unmatched = track_nonempty & (~track_matched)
    explained = (iou >= config.iou_threshold).any(axis=1)
    is_new_detection = score_ok & (~explained)
    detection_to_track = mx.argmax(iou, axis=1).astype(mx.int32)
    return AssociationResult(
        is_new_detection=is_new_detection,
        track_is_unmatched=track_is_unmatched,
        detection_to_track=detection_to_track,
    )


class Sam3TrackKeepAlive:
    """Per-object unmatched-frame counter; removes a track after ``max_unmatched_frames``."""

    def __init__(self, config: Sam3AssociationConfig):
        self.config = config
        self._unmatched: dict[int, int] = {}

    def update(self, object_id: int, unmatched: bool) -> bool:
        """Record one frame's match status; return True when the object should be removed."""
        if unmatched:
            self._unmatched[object_id] = self._unmatched.get(object_id, 0) + 1
        else:
            self._unmatched[object_id] = 0
        return self._unmatched[object_id] > self.config.max_unmatched_frames

    def drop(self, object_id: int) -> None:
        self._unmatched.pop(object_id, None)
