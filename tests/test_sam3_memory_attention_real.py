"""Slice 9: faithful SAM3 memory attention + object-pointer projection (112 tensors).

Weight-free verification:
- ``Sam3TrackerMemoryAttention`` (106) and the ``object_pointer_proj``
  ``Sam3TrackerFeedForward`` (6) map 1:1 onto the ``video.json`` reference keys
  (no convs -> identity prefix maps);
- the converters + loaders round-trip synthetic checkpoints;
- reduced-size forwards run (RoPE self + image cross-attention; the FFN heads).

The remaining tracker scalar embeddings (no_memory_*, object pointers, temporal /
occlusion encodings, mask_downsample) are top-level ``tracker_model`` parameters and
are assembled + covered by the full 1797-tensor structural test in slice 11. Real
numeric parity runs out-of-sandbox (slice 11).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.models.sam3.real_convert import (
    load_sam3_memory_attention_real_weights,
    load_sam3_object_pointer_proj_real_weights,
    remap_sam3_memory_attention_real_key,
    remap_sam3_object_pointer_proj_real_key,
)
from mlx_cv.models.sam3.real_video import Sam3TrackerFeedForward, Sam3TrackerMemoryAttention
from mlx_cv.models.sam3.real_video_config import Sam3TrackerVideoConfig

REPO = Path(__file__).resolve().parents[1]
VIDEO_KEYS = REPO / ".agent/work/2026-06-18-sam3-real-architecture-port/reference-key-shapes/video.json"


def _reference(prefix: str) -> dict[str, list[int]]:
    keys = json.loads(VIDEO_KEYS.read_text())
    return {k: v for k, v in keys.items() if k.startswith(prefix)}


def _param_shapes(model) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}


def _reduced_config() -> Sam3TrackerVideoConfig:
    return Sam3TrackerVideoConfig(memory_attention_rope_feat_sizes=(4, 4), memory_attention_num_layers=2)


# --- structural 1:1 maps ------------------------------------------------------


def test_memory_attention_maps_1to1():
    params = _param_shapes(Sam3TrackerMemoryAttention(Sam3TrackerVideoConfig()))
    reference = _reference("tracker_model.memory_attention.")
    assert len(reference) == 106
    expected = {remap_sam3_memory_attention_real_key(k): tuple(v) for k, v in reference.items()}
    assert set(params) == set(expected)
    assert all(params[k] == expected[k] for k in params), {
        k: (params[k], expected[k]) for k in params if params[k] != expected[k]
    }


def test_object_pointer_proj_maps_1to1():
    params = _param_shapes(Sam3TrackerFeedForward(256, 256, 256, 3))
    reference = _reference("tracker_model.object_pointer_proj.")
    assert len(reference) == 6
    expected = {remap_sam3_object_pointer_proj_real_key(k): tuple(v) for k, v in reference.items()}
    assert set(params) == set(expected)
    assert all(params[k] == expected[k] for k in params)


def test_cross_attn_image_projects_from_64_dim_memory():
    params = _param_shapes(Sam3TrackerMemoryAttention(Sam3TrackerVideoConfig()))
    # Memory keys/values are the 64-channel memory-encoder output -> in_features 64.
    assert params["layers.0.cross_attn_image.k_proj.weight"] == (256, 64)
    assert params["layers.0.cross_attn_image.v_proj.weight"] == (256, 64)
    assert params["layers.0.self_attn.k_proj.weight"] == (256, 256)


# --- converter + loader round-trip --------------------------------------------


def _write_npz(tmp_path: Path, state: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, **state)
    return path


def test_memory_attention_loader_round_trips(tmp_path):
    model = Sam3TrackerMemoryAttention(Sam3TrackerVideoConfig())
    params = dict(tree_flatten(model.parameters()))
    state = {f"tracker_model.memory_attention.{k}": np.zeros(tuple(v.shape), np.float32) for k, v in params.items()}
    state["detector_model.vision_encoder.backbone.layer_norm.weight"] = np.zeros((1024,), np.float32)
    loaded = load_sam3_memory_attention_real_weights(model, _write_npz(tmp_path, state))
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_object_pointer_proj_loader_round_trips(tmp_path):
    model = Sam3TrackerFeedForward(256, 256, 256, 3)
    params = dict(tree_flatten(model.parameters()))
    state = {f"tracker_model.object_pointer_proj.{k}": np.zeros(tuple(v.shape), np.float32) for k, v in params.items()}
    loaded = load_sam3_object_pointer_proj_real_weights(model, _write_npz(tmp_path, state))
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_memory_attention_loader_rejects_incomplete(tmp_path):
    model = Sam3TrackerMemoryAttention(Sam3TrackerVideoConfig())
    params = dict(tree_flatten(model.parameters()))
    state = {
        f"tracker_model.memory_attention.{k}": np.zeros(tuple(v.shape), np.float32)
        for k, v in list(params.items())[:-1]
    }
    with pytest.raises(ValueError, match="missing SAM3 memory attention params"):
        load_sam3_memory_attention_real_weights(model, _write_npz(tmp_path, state))


# --- reduced forwards ---------------------------------------------------------


def test_memory_attention_forward_runs():
    config = _reduced_config()
    model = Sam3TrackerMemoryAttention(config)
    mx.eval(model.parameters())

    seq = config.memory_attention_rope_feat_sizes[0] * config.memory_attention_rope_feat_sizes[1]  # 16
    batch, mem_dim = 1, config.memory_encoder_output_channels  # 64
    vision = mx.random.normal((seq, batch, config.memory_attention_hidden_size))
    vision_pos = mx.random.normal((seq, batch, config.memory_attention_hidden_size))
    memory = mx.random.normal((seq, batch, mem_dim))
    memory_pos = mx.random.normal((seq, batch, mem_dim))

    out = model(vision, memory, vision_pos, memory_pos)
    mx.eval(out)
    assert tuple(out.shape) == (1, batch, seq, config.memory_attention_hidden_size)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_object_pointer_proj_forward_runs():
    model = Sam3TrackerFeedForward(256, 256, 256, 3)
    mx.eval(model.parameters())
    out = model(mx.random.normal((2, 256)))
    mx.eval(out)
    assert tuple(out.shape) == (2, 256)
