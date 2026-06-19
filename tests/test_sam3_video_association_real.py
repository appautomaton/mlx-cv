"""Slice 15: faithful SAM3 detector<->tracker association (matching, spawning, keep-alive).

Two layers, both weight-free and CI-safe (no detector run):
- the association *logic* — ``mask_iou`` / ``associate_detections`` / ``Sam3TrackKeepAlive`` —
  tested deterministically on synthetic masks;
- the dynamic ``Sam3VideoMultiObjectTracker`` — objects spawn from unmatched high-score
  detections (injected) and are tracked via per-object memory banks.

Numeric parity vs the research predictor is the deferred out-of-sandbox gate.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_cv.models.sam3.real_tracker_decoder import Sam3TrackerPromptEncoderConfig
from mlx_cv.models.sam3.real_video_association import (
    Sam3AssociationConfig,
    Sam3TrackKeepAlive,
    associate_detections,
    mask_iou,
)
from mlx_cv.models.sam3.real_video_config import Sam3TrackerVideoConfig
from mlx_cv.models.sam3.real_video_model import Sam3TrackerVideoModel
from mlx_cv.models.sam3.real_video_streaming import Sam3VideoMultiObjectTracker

CHANNELS = 256


def _block_mask(size, y0, y1, x0, x1):
    m = mx.zeros((size, size))
    idx_y = (mx.arange(size) >= y0) & (mx.arange(size) < y1)
    idx_x = (mx.arange(size) >= x0) & (mx.arange(size) < x1)
    return (idx_y[:, None] & idx_x[None, :]).astype(mx.float32)


# --- pure association logic ---------------------------------------------------


def test_mask_iou_identical_disjoint_and_partial():
    a = _block_mask(8, 0, 4, 0, 4)[None]
    same = _block_mask(8, 0, 4, 0, 4)[None]
    disjoint = _block_mask(8, 4, 8, 4, 8)[None]
    half = _block_mask(8, 0, 4, 0, 2)[None]  # 8 of 16 px -> IoU 8/16 = 0.5
    assert abs(float(mask_iou(a, same)[0, 0].item()) - 1.0) < 1e-5
    assert float(mask_iou(a, disjoint)[0, 0].item()) == 0.0
    assert abs(float(mask_iou(a, half)[0, 0].item()) - 0.5) < 1e-5


def test_associate_spawns_unmatched_high_score_and_flags_unmatched_track():
    det_masks = mx.stack([_block_mask(8, 0, 4, 0, 4), _block_mask(8, 5, 8, 5, 8)], axis=0)
    det_scores = mx.array([0.9, 0.9])
    track_masks = mx.stack([_block_mask(8, 0, 4, 0, 4), _block_mask(8, 6, 8, 0, 2)], axis=0)
    result = associate_detections(det_masks, det_scores, track_masks, Sam3AssociationConfig())
    # det 0 overlaps trk 0 -> explained (not new); det 1 overlaps nothing -> new
    assert [bool(v) for v in result.is_new_detection.tolist()] == [False, True]
    # trk 0 matched by det 0; trk 1 overlaps no detection -> unmatched
    assert [bool(v) for v in result.track_is_unmatched.tolist()] == [False, True]


def test_associate_low_score_detection_is_not_new():
    det_masks = _block_mask(8, 5, 8, 5, 8)[None]
    result = associate_detections(det_masks, mx.array([0.1]), mx.zeros((0, 8, 8)), Sam3AssociationConfig())
    assert bool(result.is_new_detection[0].item()) is False


def test_keep_alive_removes_after_threshold_and_resets_on_match():
    keep = Sam3TrackKeepAlive(Sam3AssociationConfig(max_unmatched_frames=2))
    assert keep.update(1, unmatched=True) is False  # count 1
    assert keep.update(1, unmatched=True) is False  # count 2
    assert keep.update(1, unmatched=True) is True  # count 3 > 2 -> remove
    keep2 = Sam3TrackKeepAlive(Sam3AssociationConfig(max_unmatched_frames=2))
    keep2.update(2, unmatched=True)
    keep2.update(2, unmatched=False)  # reset
    assert keep2.update(2, unmatched=True) is False  # back to count 1


# --- dynamic manager (injected detections) ------------------------------------


def _tiny_tracker(g: int = 4) -> Sam3TrackerVideoModel:
    config = Sam3TrackerVideoConfig(
        prompt_encoder=Sam3TrackerPromptEncoderConfig(image_size=g * 16, patch_size=16),
        memory_attention_rope_feat_sizes=(g, g),
    )
    tracker = Sam3TrackerVideoModel(config)
    mx.eval(tracker.parameters())
    return tracker


def _frame_features(g: int = 4, seed: int = 0):
    keys = mx.random.split(mx.random.key(seed), 4)
    return (
        mx.random.normal((1, g, g, CHANNELS), key=keys[0]),
        mx.random.normal((1, g, g, CHANNELS), key=keys[1]),
        [
            mx.random.normal((1, g * 4, g * 4, CHANNELS), key=keys[2]),
            mx.random.normal((1, g * 2, g * 2, CHANNELS), key=keys[3]),
        ],
    )


def test_manager_spawns_objects_from_detections():
    g = 4
    manager = Sam3VideoMultiObjectTracker(_tiny_tracker(g))
    det_masks = mx.stack(
        [_block_mask(g * 4, 0, 8, 0, 8), _block_mask(g * 4, 8, 16, 8, 16), _block_mask(g * 4, 0, 4, 0, 4)],
        axis=0,
    )
    det_scores = mx.array([0.9, 0.9, 0.1])  # third below threshold -> no spawn
    active = manager.step(_frame_features(g, seed=0), det_masks, det_scores)
    assert active == [1, 2]  # two high-score, unmatched-on-empty-state detections spawned


def test_manager_runs_clip_and_is_deterministic():
    g = 4
    det = _block_mask(g * 4, 2, 12, 2, 12)[None]
    scores = mx.array([0.9])

    def run():
        manager = Sam3VideoMultiObjectTracker(_tiny_tracker(g))
        history = []
        for i in range(3):
            history.append(list(manager.step(_frame_features(g, seed=i), det, scores)))
        return history

    first, second = run(), run()
    assert first[0] == [1]  # spawned on frame 0
    assert first == second  # deterministic active-id sequence


def test_manager_step_without_detections_does_not_crash():
    g = 4
    manager = Sam3VideoMultiObjectTracker(_tiny_tracker(g))
    manager.step(_frame_features(g, seed=0), _block_mask(g * 4, 2, 12, 2, 12)[None], mx.array([0.9]))
    empty = mx.zeros((0, g * 4, g * 4))
    active = manager.step(_frame_features(g, seed=1), empty, mx.zeros((0,)))
    assert isinstance(active, list)  # tracked existing object, no detections to match
