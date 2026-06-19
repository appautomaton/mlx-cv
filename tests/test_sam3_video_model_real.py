"""Slice 11: faithful SAM3 video model assembly (Sam3VideoModel, 1797 tensors).

Weight-free verification:
- the full ``Sam3VideoModel`` parameter tree maps 1:1 onto the entire ``video.json``
  reference (detector_model + tracker_model + tracker_neck), with conv transposes;
- the full video converter + loader round-trip a reduced synthetic checkpoint
  (exercises detector, tracker, neck + all conv layouts);
- ``get_vision_features_for_tracker`` (detector vision encoder -> tracker FPN neck)
  runs and produces coherent shapes.

The per-frame streaming / association loop (``Sam3VideoModel.forward`` over an
inference session) is the remaining numerically-gated piece and runs out-of-sandbox
with the gated checkpoint (see parity-status ``sam3_video``). No synthetic pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

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
from mlx_cv.models.sam3.real_convert import (
    convert_reference_shape,
    load_sam3_video_real_weights,
    remap_sam3_video_real_key,
)
from mlx_cv.models.sam3.real_video_config import Sam3TrackerVideoConfig
from mlx_cv.models.sam3.real_video_model import Sam3VideoModel

REPO = Path(__file__).resolve().parents[1]
VIDEO_KEYS = REPO / ".agent/work/2026-06-18-sam3-real-architecture-port/reference-key-shapes/video.json"


def _reduced_video() -> Sam3VideoModel:
    vit = Sam3ViTConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        image_size=56, patch_size=14, window_size=4, global_attn_indexes=(1,), pretrain_image_size=28,
    )
    vision = Sam3VisionConfig(backbone=vit, fpn_hidden_size=32, scale_factors=(4.0, 2.0, 1.0, 0.5))
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
    # Tracker keeps default (256) dims; we only structurally round-trip it at full size.
    return Sam3VideoModel(detector_config=detector, tracker_config=Sam3TrackerVideoConfig())


# --- structural 1:1 map (full config) -----------------------------------------


def test_video_model_param_tree_is_1797_keys():
    params = {k for k, _ in tree_flatten(Sam3VideoModel().parameters())}
    assert len(params) == 1797


def test_video_model_maps_1to1_against_full_reference():
    params = {k: tuple(v.shape) for k, v in tree_flatten(Sam3VideoModel().parameters())}
    reference = json.loads(VIDEO_KEYS.read_text())
    assert len(reference) == 1797

    expected = {}
    for ref_key, ref_shape in reference.items():
        local = remap_sam3_video_real_key(ref_key)
        assert local is not None, ref_key
        expected[local] = convert_reference_shape(local, tuple(ref_shape))

    assert set(params) == set(expected)
    mismatched = {k: (params[k], expected[k]) for k in params if params[k] != expected[k]}
    assert not mismatched, mismatched


def test_video_model_has_three_top_level_namespaces():
    params = {k for k, _ in tree_flatten(Sam3VideoModel().parameters())}
    assert any(k.startswith("detector_model.") for k in params)
    assert any(k.startswith("tracker_model.") for k in params)
    assert any(k.startswith("tracker_neck.") for k in params)
    # Tracker scalar embeddings are present.
    assert "tracker_model.no_memory_embedding" in params
    assert "tracker_model.memory_temporal_positional_encoding" in params


# --- full video converter + loader round-trip (reduced) -----------------------


def test_video_loader_round_trips_reduced_checkpoint(tmp_path):
    model = _reduced_video()
    params = dict(tree_flatten(model.parameters()))

    state: dict[str, np.ndarray] = {}
    for local, value in params.items():
        arr = np.zeros(tuple(value.shape), np.float32)
        if arr.ndim == 4 and local.endswith(".weight"):
            inv = (3, 0, 1, 2) if (".scale_layers." in local or "upscale_conv" in local) else (0, 3, 1, 2)
            arr = np.transpose(arr, inv)
        state[local] = arr

    path = tmp_path / "video.npz"
    np.savez(path, **state)
    loaded = load_sam3_video_real_weights(model, path)
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_video_loader_rejects_incomplete(tmp_path):
    model = _reduced_video()
    params = dict(tree_flatten(model.parameters()))
    state = {}
    for local, value in list(params.items())[:-1]:
        arr = np.zeros(tuple(value.shape), np.float32)
        if arr.ndim == 4 and local.endswith(".weight"):
            inv = (3, 0, 1, 2) if (".scale_layers." in local or "upscale_conv" in local) else (0, 3, 1, 2)
            arr = np.transpose(arr, inv)
        state[local] = arr
    path = tmp_path / "video.npz"
    np.savez(path, **state)
    with pytest.raises(ValueError, match="missing SAM3 video params"):
        load_sam3_video_real_weights(model, path)


# --- tracker feature path forward ---------------------------------------------


def test_get_vision_features_for_tracker_runs():
    model = _reduced_video()
    mx.eval(model.parameters())
    pixel_values = mx.random.normal((1, 3, 56, 56))  # 4x4 patch grid -> 16 tokens
    fpn = model.get_vision_features_for_tracker(pixel_values)
    mx.eval(*fpn)
    # tracker_neck reuses the detector neck: 4 FPN levels over the 4x4 grid.
    expected = [(16, 16), (8, 8), (4, 4), (2, 2)]
    assert len(fpn) == 4
    for level, (height, width) in enumerate(expected):
        assert tuple(fpn[level].shape) == (1, height, width, model.detector_model.config.vision.fpn_hidden_size)
        assert bool(mx.all(mx.isfinite(fpn[level])).item())
