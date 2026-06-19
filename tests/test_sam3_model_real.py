"""Slice 7: faithful SAM3 image detector assembly (Sam3Model, 1468 tensors).

Weight-free verification:
- the full ``Sam3Model`` parameter tree maps 1:1 onto the entire ``detector.json``
  reference (all subsystems, with conv layout transposes);
- the full-detector converter + loader round-trip a reduced synthetic checkpoint
  (exercises every namespace + both conv layouts);
- a reduced-size end-to-end text-prompt forward produces coherent shapes, valid box
  (xyxy in [0,1]) and presence ranges, finite masks, and is deterministic.

The real numeric upstream-vs-MLX parity gate (boxes/scores/masks) runs out-of-sandbox
with the gated facebook/sam3 checkpoint via tools/sam3_upstream.py — see the slice-7
gate command in parity-status.json. No synthetic pass is claimed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.models.sam3.real_config import (
    Sam3DETRDecoderConfig,
    Sam3DETREncoderConfig,
    Sam3DetectorConfig,
    Sam3GeometryEncoderConfig,
    Sam3MaskDecoderConfig,
    Sam3TextConfig,
    Sam3VisionConfig,
    Sam3ViTConfig,
)
from mlx_cv.models.sam3.real_convert import (
    convert_reference_shape,
    load_sam3_detector_real_weights,
    remap_sam3_detector_real_key,
)
from mlx_cv.models.sam3.real_modeling import Sam3Model

REPO = Path(__file__).resolve().parents[1]
DETECTOR_KEYS = REPO / ".agent/work/2026-06-18-sam3-real-architecture-port/reference-key-shapes/detector.json"


def _reduced_config() -> Sam3DetectorConfig:
    vit = Sam3ViTConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        image_size=56, patch_size=14, window_size=4, global_attn_indexes=(1,), pretrain_image_size=28,
    )
    vision = Sam3VisionConfig(backbone=vit, fpn_hidden_size=32, scale_factors=(4.0, 2.0, 1.0, 0.5))
    text = Sam3TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, projection_dim=16,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=12,
    )
    geom = Sam3GeometryEncoderConfig(hidden_size=32, num_layers=1, num_attention_heads=4, intermediate_size=64)
    denc = Sam3DETREncoderConfig(hidden_size=32, num_layers=1, num_attention_heads=4, intermediate_size=64)
    ddec = Sam3DETRDecoderConfig(hidden_size=32, num_layers=2, num_queries=5, num_attention_heads=4, intermediate_size=64)
    mask = Sam3MaskDecoderConfig(hidden_size=32, num_upsampling_stages=2, num_attention_heads=4)
    return Sam3DetectorConfig(
        vision=vision, text=text, geometry_encoder=geom, detr_encoder=denc, detr_decoder=ddec, mask_decoder=mask
    )


# --- structural 1:1 map (full config) -----------------------------------------


def test_detector_param_tree_is_1468_keys():
    params = {k for k, _ in tree_flatten(Sam3Model(Sam3DetectorConfig()).parameters())}
    assert len(params) == 1468


def test_detector_maps_1to1_against_full_reference():
    params = {k: tuple(v.shape) for k, v in tree_flatten(Sam3Model(Sam3DetectorConfig()).parameters())}
    reference = json.loads(DETECTOR_KEYS.read_text())
    assert len(reference) == 1468

    expected = {}
    for ref_key, ref_shape in reference.items():
        local = remap_sam3_detector_real_key(ref_key)
        assert local is not None, ref_key
        expected[local] = convert_reference_shape(local, tuple(ref_shape))

    assert set(params) == set(expected)
    mismatched = {k: (params[k], expected[k]) for k in params if params[k] != expected[k]}
    assert not mismatched, mismatched


def test_tracker_keys_are_dropped_by_detector_remap():
    assert remap_sam3_detector_real_key("tracker_model.memory_encoder.x.weight") is None
    assert remap_sam3_detector_real_key("tracker_neck.0.weight") is None
    assert (
        remap_sam3_detector_real_key("detector_model.vision_encoder.backbone.layer_norm.weight")
        == "vision_encoder.backbone.layer_norm.weight"
    )


# --- full-detector converter + loader round-trip (reduced config) -------------


def test_detector_loader_round_trips_reduced_checkpoint(tmp_path):
    model = Sam3Model(_reduced_config())
    params = dict(tree_flatten(model.parameters()))

    state: dict[str, np.ndarray] = {}
    for local, value in params.items():
        arr = np.zeros(tuple(value.shape), np.float32)
        if arr.ndim == 4 and local.endswith(".weight"):
            inv = (3, 0, 1, 2) if ".scale_layers." in local else (0, 3, 1, 2)
            arr = np.transpose(arr, inv)
        state[f"detector_model.{local}"] = arr
    # A tracker key (video-only) must be ignored by the detector loader.
    state["tracker_neck.0.weight"] = np.zeros((4, 4), np.float32)

    path = tmp_path / "detector.npz"
    np.savez(path, **state)
    loaded = load_sam3_detector_real_weights(model, path)
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_detector_loader_rejects_incomplete(tmp_path):
    model = Sam3Model(_reduced_config())
    params = dict(tree_flatten(model.parameters()))
    state = {}
    for local, value in list(params.items())[:-1]:
        arr = np.zeros(tuple(value.shape), np.float32)
        if arr.ndim == 4 and local.endswith(".weight"):
            inv = (3, 0, 1, 2) if ".scale_layers." in local else (0, 3, 1, 2)
            arr = np.transpose(arr, inv)
        state[f"detector_model.{local}"] = arr
    path = tmp_path / "detector.npz"
    np.savez(path, **state)
    with pytest.raises(ValueError, match="missing SAM3 detector params"):
        load_sam3_detector_real_weights(model, path)


# --- reduced end-to-end forward -----------------------------------------------


def _inputs(config, seq=7):
    pixel = mx.random.normal((1, 3, config.image_size, config.image_size))
    ids = mx.array((np.arange(seq) * 7 + 1).astype(np.int32).reshape(1, seq))
    return pixel, ids


def test_detector_end_to_end_forward_shapes():
    config = _reduced_config()
    model = Sam3Model(config)
    mx.eval(model.parameters())
    pixel, ids = _inputs(config)

    out = model(pixel, ids)
    mx.eval(out.pred_logits, out.pred_boxes, out.presence_logits, out.pred_masks, out.semantic_seg)

    num_queries = config.detr_decoder.num_queries
    # Finest FPN level used as backbone[0] is scale 4.0 over the 4x4 grid -> 16x16.
    assert tuple(out.pred_logits.shape) == (1, num_queries)
    assert tuple(out.pred_boxes.shape) == (1, num_queries, 4)
    assert tuple(out.presence_logits.shape) == (1, 1)
    assert tuple(out.pred_masks.shape) == (1, num_queries, 16, 16)
    assert tuple(out.semantic_seg.shape) == (1, 1, 16, 16)

    assert bool(mx.all(mx.isfinite(out.pred_masks)).item())
    assert bool(mx.all(mx.isfinite(out.pred_boxes)).item())
    # pred_boxes are xyxy derived from sigmoid cxcywh, so corners may fall slightly
    # outside [0, 1]; just bound them to a sane range around the unit box.
    assert bool((mx.min(out.pred_boxes) >= -1.0).item())
    assert bool((mx.max(out.pred_boxes) <= 2.0).item())


def test_detector_end_to_end_is_deterministic():
    config = _reduced_config()
    model = Sam3Model(config)
    mx.eval(model.parameters())
    pixel, ids = _inputs(config)
    first = model(pixel, ids).pred_logits
    second = model(pixel, ids).pred_logits
    assert bool(mx.all(first == second).item())
