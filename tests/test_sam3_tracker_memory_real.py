"""Slice 8: faithful SAM3 video tracker — neck + memory encoder (62 tensors).

Weight-free verification:
- the MLX ``Sam3TrackerMemoryEncoder`` (40) and the reused ``Sam3VisionNeck`` as
  ``tracker_neck`` (22) map 1:1 onto the ``video.json`` reference keys, with conv
  layout transposes;
- the converters + loaders round-trip synthetic checkpoints;
- reduced-size forwards run (mask downsampling + feature fusion + CX blocks; FPN).

Real numeric parity runs out-of-sandbox with the gated checkpoint (slice 11).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.models.sam3.real_config import Sam3VisionConfig
from mlx_cv.models.sam3.real_convert import (
    convert_reference_shape,
    load_sam3_memory_encoder_real_weights,
    load_sam3_tracker_neck_real_weights,
    remap_sam3_memory_encoder_real_key,
    remap_sam3_tracker_neck_real_key,
)
from mlx_cv.models.sam3.real_video import Sam3TrackerMemoryEncoder
from mlx_cv.models.sam3.real_video_config import Sam3TrackerVideoConfig
from mlx_cv.models.sam3.real_vision import Sam3VisionNeck

REPO = Path(__file__).resolve().parents[1]
VIDEO_KEYS = REPO / ".agent/work/2026-06-18-sam3-real-architecture-port/reference-key-shapes/video.json"


def _reference(prefix: str) -> dict[str, list[int]]:
    keys = json.loads(VIDEO_KEYS.read_text())
    return {k: v for k, v in keys.items() if k.startswith(prefix)}


def _param_shapes(model) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}


def _assert_maps_1to1(model, prefix, remap, count):
    params = _param_shapes(model)
    reference = _reference(prefix)
    assert len(reference) == count, (prefix, len(reference))
    expected = {}
    for ref_key, ref_shape in reference.items():
        local = remap(ref_key)
        assert local is not None, ref_key
        expected[local] = convert_reference_shape(local, tuple(ref_shape))
    assert set(params) == set(expected)
    mismatched = {k: (params[k], expected[k]) for k in params if params[k] != expected[k]}
    assert not mismatched, mismatched


# --- structural 1:1 maps ------------------------------------------------------


def test_memory_encoder_maps_1to1():
    _assert_maps_1to1(
        Sam3TrackerMemoryEncoder(Sam3TrackerVideoConfig()),
        "tracker_model.memory_encoder.",
        remap_sam3_memory_encoder_real_key,
        40,
    )


def test_tracker_neck_maps_1to1():
    _assert_maps_1to1(
        Sam3VisionNeck(Sam3VisionConfig()), "tracker_neck.", remap_sam3_tracker_neck_real_key, 22
    )


def test_memory_encoder_conv_transposes():
    # Depthwise CX conv: torch [C, 1, 7, 7] -> MLX [C, 7, 7, 1].
    assert convert_reference_shape("memory_fuser.layers.0.depthwise_conv.weight", (256, 1, 7, 7)) == (256, 7, 7, 1)
    # Mask downsampler first conv: torch [4, 1, 3, 3] -> MLX [4, 3, 3, 1].
    assert convert_reference_shape("mask_downsampler.layers.0.conv.weight", (4, 1, 3, 3)) == (4, 3, 3, 1)
    # pointwise convs are Linear -> untouched.
    assert convert_reference_shape("memory_fuser.layers.0.pointwise_conv1.weight", (1024, 256)) == (1024, 256)


# --- converter + loader round-trip --------------------------------------------


def _write_npz(tmp_path: Path, state: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, **state)
    return path


def _synthetic(model, prefix: str) -> dict[str, np.ndarray]:
    state = {}
    for local, value in dict(tree_flatten(model.parameters())).items():
        arr = np.zeros(tuple(value.shape), np.float32)
        if arr.ndim == 4 and local.endswith(".weight"):
            inv = (3, 0, 1, 2) if ".scale_layers." in local else (0, 3, 1, 2)
            arr = np.transpose(arr, inv)
        state[f"{prefix}{local}"] = arr
    return state


def test_memory_encoder_loader_round_trips(tmp_path):
    model = Sam3TrackerMemoryEncoder(Sam3TrackerVideoConfig())
    params = dict(tree_flatten(model.parameters()))
    state = _synthetic(model, "tracker_model.memory_encoder.")
    state["detector_model.vision_encoder.backbone.layer_norm.weight"] = np.zeros((1024,), np.float32)
    loaded = load_sam3_memory_encoder_real_weights(model, _write_npz(tmp_path, state))
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_tracker_neck_loader_round_trips(tmp_path):
    model = Sam3VisionNeck(Sam3VisionConfig())
    params = dict(tree_flatten(model.parameters()))
    state = _synthetic(model, "tracker_neck.")
    loaded = load_sam3_tracker_neck_real_weights(model, _write_npz(tmp_path, state))
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_memory_encoder_loader_rejects_incomplete(tmp_path):
    model = Sam3TrackerMemoryEncoder(Sam3TrackerVideoConfig())
    full = _synthetic(model, "tracker_model.memory_encoder.")
    partial = dict(list(full.items())[:-1])
    with pytest.raises(ValueError, match="missing SAM3 memory encoder params"):
        load_sam3_memory_encoder_real_weights(model, _write_npz(tmp_path, partial))


# --- reduced forwards ---------------------------------------------------------


def test_memory_encoder_forward_shapes():
    config = Sam3TrackerVideoConfig()
    model = Sam3TrackerMemoryEncoder(config)
    mx.eval(model.parameters())

    batch, h, w = 1, 4, 4
    vision = mx.random.normal((batch, h, w, config.memory_encoder_hidden_size))
    masks = mx.random.normal((batch, h * 16, w * 16, 1))  # downsampled by total_stride=16
    out = model(vision, masks)
    mx.eval(out.vision_features, out.vision_pos_enc)
    assert tuple(out.vision_features.shape) == (batch, h, w, config.memory_encoder_output_channels)
    assert tuple(out.vision_pos_enc.shape) == (batch, h, w, config.memory_encoder_output_channels)
    assert bool(mx.all(mx.isfinite(out.vision_features)).item())


def test_tracker_neck_forward_shapes():
    config = Sam3VisionConfig()
    model = Sam3VisionNeck(config)
    mx.eval(model.parameters())
    spatial = mx.random.normal((1, 8, 8, config.backbone.hidden_size))
    fpn_hidden, fpn_pos = model(spatial)
    mx.eval(*fpn_hidden, *fpn_pos)
    expected = [(32, 32), (16, 16), (8, 8), (4, 4)]
    for level, (height, width) in enumerate(expected):
        assert tuple(fpn_hidden[level].shape) == (1, height, width, config.fpn_hidden_size)
