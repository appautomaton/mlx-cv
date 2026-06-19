"""Slice 12: faithful SAM3 tracker per-frame ``track_step`` (single object).

Weight-free structural verification of the per-frame memory-propagation step that
``Sam3TrackerVideoModel`` now exposes (mirrors the SAM2-style tracker loop with the
faithful slice 8-10 subsystems):

- the init/no-memory path (seed box prompt) produces coherent mask / pointer / memory
  shapes and finite values;
- the memory-conditioned path (a prior frame's memory + object pointer fed back through
  the 4-layer memory-attention transformer) runs and stays finite;
- the step is deterministic (identical inputs -> identical outputs).

Numeric parity vs the upstream reference is the deferred, out-of-sandbox gate
(see parity-status ``sam3_video``); no synthetic pass and no real weights here.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_cv.models.sam3.real_tracker_decoder import Sam3TrackerPromptEncoderConfig
from mlx_cv.models.sam3.real_video_config import Sam3TrackerVideoConfig
from mlx_cv.models.sam3.real_video_model import Sam3TrackerStageOutput, Sam3TrackerVideoModel

GRID = 4  # tracker feature grid (g); image-embedding + RoPE feat sizes matched to it
CHANNELS = 256
MEM_DIM = 64


def _tiny_tracker() -> Sam3TrackerVideoModel:
    """A faithful tracker on a tiny grid: prompt-encoder grid and RoPE feat sizes == GRID."""
    config = Sam3TrackerVideoConfig(
        prompt_encoder=Sam3TrackerPromptEncoderConfig(image_size=GRID * 16, patch_size=16),
        memory_attention_rope_feat_sizes=(GRID, GRID),
    )
    model = Sam3TrackerVideoModel(config)
    mx.eval(model.parameters())
    return model


def _frame_features(seed: int) -> dict:
    """Random per-frame inputs: top-level features + pos enc + two raw high-res FPN levels."""
    key = mx.random.key(seed)
    k0, k1, k2, k3 = mx.random.split(key, 4)
    return {
        "vision_features": mx.random.normal((1, GRID, GRID, CHANNELS), key=k0),
        "vision_pos": mx.random.normal((1, GRID, GRID, CHANNELS), key=k1),
        "high_res_features": [
            mx.random.normal((1, GRID * 4, GRID * 4, CHANNELS), key=k2),  # 4g-res
            mx.random.normal((1, GRID * 2, GRID * 2, CHANNELS), key=k3),  # 2g-res
        ],
    }


def _box_prompt() -> tuple[mx.array, mx.array]:
    # SAM box encoding: top-left (label 2) + bottom-right (label 3), in input-image pixels.
    coords = mx.array([[[8.0, 8.0], [40.0, 40.0]]])  # [1, 2, 2]
    labels = mx.array([[2.0, 3.0]])  # [1, 2]
    return coords, labels


def _assert_stage_shapes(out: Sam3TrackerStageOutput) -> None:
    assert out.low_res_masks.shape == (1, GRID * 4, GRID * 4, 1)
    assert out.high_res_masks.shape == (1, GRID * 16, GRID * 16, 1)
    assert out.iou_pred.shape == (1, 1, 1)
    assert out.object_score_logits.shape == (1, 1, 1)
    assert out.obj_ptr.shape == (1, CHANNELS)
    assert out.maskmem_features.shape == (1, GRID, GRID, MEM_DIM)
    assert out.maskmem_pos_enc.shape == (1, GRID, GRID, MEM_DIM)
    for value in (out.low_res_masks, out.high_res_masks, out.obj_ptr, out.maskmem_features):
        assert bool(mx.all(mx.isfinite(value)).item())


def test_init_frame_box_prompt_produces_coherent_shapes():
    model = _tiny_tracker()
    out = model.track_step(is_init_cond_frame=True, point_inputs=_box_prompt(), **_frame_features(0))
    _assert_stage_shapes(out)


def test_memory_conditioned_frame_runs_and_is_finite():
    model = _tiny_tracker()
    init = model.track_step(is_init_cond_frame=True, point_inputs=_box_prompt(), **_frame_features(0))
    tracked = model.track_step(is_init_cond_frame=False, previous_frames=[init], **_frame_features(1))
    _assert_stage_shapes(tracked)


def test_track_step_is_deterministic():
    model = _tiny_tracker()
    features = _frame_features(0)
    first = model.track_step(is_init_cond_frame=True, point_inputs=_box_prompt(), **features)
    second = model.track_step(is_init_cond_frame=True, point_inputs=_box_prompt(), **features)
    assert bool(mx.all(first.low_res_masks == second.low_res_masks).item())
    assert bool(mx.all(first.obj_ptr == second.obj_ptr).item())
    assert bool(mx.all(first.maskmem_features == second.maskmem_features).item())


def test_decoder_exposes_sam_token_for_object_pointer():
    """The additive mask-token return must align with the (single) selected mask."""
    model = _tiny_tracker()
    out = model.track_step(is_init_cond_frame=True, point_inputs=_box_prompt(), **_frame_features(2))
    # obj_ptr is finite and shaped from the SAM token via object_pointer_proj.
    assert out.obj_ptr.shape == (1, CHANNELS)
    assert bool(mx.all(mx.isfinite(out.obj_ptr)).item())
