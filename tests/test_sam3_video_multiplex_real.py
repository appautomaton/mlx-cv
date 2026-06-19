"""Slice 14: faithful SAM3 Object-Multiplex batching (multiple objects per frame).

Weight-free structural verification that several seeded objects track together as the
batch dimension ``B`` through one ``track_step`` per frame — each object carries its own
memory bank + box prompt over the shared frame features. Injection-based (no detector),
so it runs on the Linux mlx[cpu] CI; numeric parity is the deferred out-of-sandbox gate.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_cv.models.sam3.real_tracker_decoder import Sam3TrackerPromptEncoderConfig
from mlx_cv.models.sam3.real_video_config import Sam3TrackerVideoConfig
from mlx_cv.models.sam3.real_video_model import Sam3TrackerVideoModel
from mlx_cv.models.sam3.real_video_streaming import Sam3VideoSession

CHANNELS = 256


def _tiny_tracker(g: int = 4) -> Sam3TrackerVideoModel:
    config = Sam3TrackerVideoConfig(
        prompt_encoder=Sam3TrackerPromptEncoderConfig(image_size=g * 16, patch_size=16),
        memory_attention_rope_feat_sizes=(g, g),
    )
    tracker = Sam3TrackerVideoModel(config)
    mx.eval(tracker.parameters())
    return tracker


def _inject_features(g: int = 4, seed: int = 0):
    keys = mx.random.split(mx.random.key(seed), 4)
    return (
        mx.random.normal((1, g, g, CHANNELS), key=keys[0]),
        mx.random.normal((1, g, g, CHANNELS), key=keys[1]),
        [
            mx.random.normal((1, g * 4, g * 4, CHANNELS), key=keys[2]),
            mx.random.normal((1, g * 2, g * 2, CHANNELS), key=keys[3]),
        ],
    )


def test_multiplex_two_objects_batch_through_one_track_step():
    g = 4
    session = Sam3VideoSession.from_tracker(_tiny_tracker(g))
    session.add_box_prompt(0, [4, 4, 28, 28], object_id=5)
    session.add_box_prompt(0, [20, 20, 60, 60], object_id=9)
    results = session.run_from_features([_inject_features(g, seed=i) for i in range(3)])
    assert [r.frame_index for r in results] == [0, 1, 2]
    for r in results:
        assert r.object_ids == [5, 9]  # stable, ordered
        assert r.masks.shape == (2, g * 4, g * 4)  # one mask row per object
        assert r.object_score_logits.shape == (2,)
        assert bool(mx.all(mx.isfinite(r.object_score_logits)).item())


def test_multiplex_scales_to_more_objects():
    g = 4
    session = Sam3VideoSession.from_tracker(_tiny_tracker(g))
    for i, oid in enumerate((3, 7, 11, 4)):
        session.add_box_prompt(0, [2 + i, 2 + i, 30 + i, 30 + i], object_id=oid)
    results = session.run_from_features([_inject_features(g, seed=i) for i in range(2)])
    for r in results:
        assert r.object_ids == [3, 7, 11, 4]
        assert r.masks.shape == (4, g * 4, g * 4)


def test_multiplex_rejects_mixed_seed_frames():
    session = Sam3VideoSession.from_tracker(_tiny_tracker())
    session.add_box_prompt(0, [4, 4, 28, 28], object_id=1)
    session.add_box_prompt(1, [4, 4, 28, 28], object_id=2)
    with pytest.raises(ValueError, match="seed frame"):
        session.run_from_features([_inject_features() for _ in range(2)])


def test_multiplex_is_deterministic():
    g = 4
    session = Sam3VideoSession.from_tracker(_tiny_tracker(g))
    session.add_box_prompt(0, [4, 4, 28, 28], object_id=1)
    session.add_box_prompt(0, [20, 20, 60, 60], object_id=2)
    frames = [_inject_features(g, seed=i) for i in range(3)]
    first = session.run_from_features(frames)
    second = session.run_from_features(frames)
    for a, b in zip(first, second):
        assert bool(mx.all(a.masks == b.masks).item())
        assert bool(mx.all(a.object_score_logits == b.object_score_logits).item())
