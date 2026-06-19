"""Slice 13: faithful SAM3 single-object streaming session (memory bank + propagation).

Weight-free structural verification of the per-object streaming loop layered on the
slice-12 ``track_step``:

- the propagation loop tracks one object across a clip from a seed box prompt, growing
  the memory bank and feeding it back through the memory-attention transformer (frames
  after the prompt exercise the memory-conditioned path);
- ``Sam3VideoModel.extract_tracker_features`` selects the SAM2 feature levels
  (image-embeddings ``g`` + ``[4g, 2g]`` high-res) with their position encodings;
- the end-to-end ``propagate`` (detector vision -> tracker neck -> streaming loop) runs on
  a reduced model;
- the loop is deterministic.

Numeric parity stays the deferred out-of-sandbox gate (parity-status ``sam3_video``).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

try:  # the Linux mlx[cpu] CI build aborts (SIGABRT, no Python error) on the full detector ViT
    _GPU_AVAILABLE = mx.metal.is_available()
except Exception:  # pragma: no cover - depends on the MLX build
    _GPU_AVAILABLE = False

from mlx_cv.models.sam3.real_config import (
    Sam3DetectorConfig,
    Sam3DETRDecoderConfig,
    Sam3DETREncoderConfig,
    Sam3GeometryEncoderConfig,
    Sam3MaskDecoderConfig,
    Sam3TextConfig,
    Sam3VisionConfig,
    Sam3ViTConfig,
)
from mlx_cv.models.sam3.real_tracker_decoder import Sam3TrackerPromptEncoderConfig
from mlx_cv.models.sam3.real_video_config import Sam3TrackerVideoConfig
from mlx_cv.models.sam3.real_video_model import Sam3TrackerVideoModel, Sam3VideoModel
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
        mx.random.normal((1, g, g, CHANNELS), key=keys[0]),  # image embeddings (grid g)
        mx.random.normal((1, g, g, CHANNELS), key=keys[1]),  # positional encoding
        [
            mx.random.normal((1, g * 4, g * 4, CHANNELS), key=keys[2]),  # 4g-res
            mx.random.normal((1, g * 2, g * 2, CHANNELS), key=keys[3]),  # 2g-res
        ],
    )


def _reduced_video(*, fpn_hidden_size: int, tracker_g: int) -> Sam3VideoModel:
    vit = Sam3ViTConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        image_size=56, patch_size=14, window_size=4, global_attn_indexes=(1,), pretrain_image_size=28,
    )
    vision = Sam3VisionConfig(backbone=vit, fpn_hidden_size=fpn_hidden_size, scale_factors=(4.0, 2.0, 1.0, 0.5))
    text = Sam3TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, projection_dim=16,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=12,
    )
    detector = Sam3DetectorConfig(
        vision=vision,
        text=text,
        geometry_encoder=Sam3GeometryEncoderConfig(hidden_size=32, num_layers=1, num_attention_heads=4, intermediate_size=64),
        detr_encoder=Sam3DETREncoderConfig(hidden_size=32, num_layers=1, num_attention_heads=4, intermediate_size=64),
        detr_decoder=Sam3DETRDecoderConfig(hidden_size=32, num_layers=2, num_queries=5, num_attention_heads=4, intermediate_size=64),
        mask_decoder=Sam3MaskDecoderConfig(hidden_size=32, num_upsampling_stages=2, num_attention_heads=4),
    )
    tracker = Sam3TrackerVideoConfig(
        prompt_encoder=Sam3TrackerPromptEncoderConfig(image_size=tracker_g * 16, patch_size=16),
        memory_attention_rope_feat_sizes=(tracker_g, tracker_g),
    )
    return Sam3VideoModel(detector_config=detector, tracker_config=tracker)


def test_streaming_loop_single_object_tracks_across_frames():
    g = 4
    session = Sam3VideoSession.from_tracker(_tiny_tracker(g))
    session.add_box_prompt(0, [8, 8, 40, 40], object_id=7)
    results = session.run_from_features([_inject_features(g, seed=i) for i in range(3)])
    assert [r.frame_index for r in results] == [0, 1, 2]
    assert all(r.object_ids == [7] for r in results)  # stable single-object id
    for r in results:
        assert r.masks.shape == (1, g * 4, g * 4)
        assert r.masks.dtype == mx.bool_
        assert r.object_score_logits.shape == (1,)
        assert bool(mx.all(mx.isfinite(r.object_score_logits)).item())


def test_streaming_loop_is_deterministic():
    g = 4
    session = Sam3VideoSession.from_tracker(_tiny_tracker(g))
    session.add_box_prompt(0, [8, 8, 40, 40], object_id=1)
    frames = [_inject_features(g, seed=i) for i in range(3)]
    first = session.run_from_features(frames)
    second = session.run_from_features(frames)
    for a, b in zip(first, second):
        assert bool(mx.all(a.masks == b.masks).item())
        assert bool(mx.all(a.object_score_logits == b.object_score_logits).item())


def test_run_from_features_requires_prompt():
    session = Sam3VideoSession.from_tracker(_tiny_tracker())
    with pytest.raises(ValueError, match="prompt"):
        session.run_from_features([_inject_features()])


def test_extract_tracker_features_selects_sam2_levels():
    model = _reduced_video(fpn_hidden_size=32, tracker_g=2)
    mx.eval(model.parameters())
    image_embeddings, image_pos, high_res = model.extract_tracker_features(mx.random.normal((1, 3, 56, 56)))
    mx.eval(image_embeddings, image_pos, *high_res)
    assert image_embeddings.shape == (1, 2, 2, 32)  # L3 == g
    assert image_pos.shape == (1, 2, 2, 32)
    assert high_res[0].shape == (1, 8, 8, 32)  # L1 == 4g
    assert high_res[1].shape == (1, 4, 4, 32)  # L2 == 2g
    assert bool(mx.all(mx.isfinite(image_embeddings)).item())


@pytest.mark.skipif(
    not _GPU_AVAILABLE,
    reason="end-to-end runs the full detector ViT at fpn=256, which SIGABRTs on the Linux "
    "mlx[cpu] CI build; the loop + level selection are covered on CPU by the injection and "
    "extraction tests above, and the full chain is numerically gated out-of-sandbox",
)
def test_end_to_end_propagate_reduced_model():
    model = _reduced_video(fpn_hidden_size=CHANNELS, tracker_g=2)
    mx.eval(model.parameters())
    session = Sam3VideoSession(model)
    session.add_box_prompt(0, [4, 4, 28, 28], object_id=3)
    frames = [mx.random.normal((1, 3, 56, 56), key=mx.random.key(i)) for i in range(3)]
    results = session.propagate(frames)
    assert [r.frame_index for r in results] == [0, 1, 2]
    assert all(r.object_ids == [3] for r in results)
    for r in results:
        assert r.masks.shape == (1, 8, 8)  # 4g == 8
        assert bool(mx.all(mx.isfinite(r.object_score_logits)).item())
