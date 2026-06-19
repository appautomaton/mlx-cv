"""Slice 5: faithful SAM3 DETR decoder (6-layer, 200-query, 247 tensors).

Weight-free verification:
- the MLX ``Sam3DetrDecoder`` parameter tree maps 1:1 onto the ``detr_decoder.*``
  reference keys (committed ``detector.json``);
- the converter + loader round-trip a synthetic checkpoint;
- a reduced-size forward runs end-to-end with coherent shapes, valid box/presence
  ranges, the box-RPB single-level path active, and is deterministic.

Real numeric tap parity runs out-of-sandbox with the gated checkpoint (slice 7).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.models.sam3.real_config import Sam3DETRDecoderConfig
from mlx_cv.models.sam3.real_convert import (
    load_sam3_decoder_real_weights,
    remap_sam3_decoder_real_key,
)
from mlx_cv.models.sam3.real_decoder import Sam3DetrDecoder

REPO = Path(__file__).resolve().parents[1]
DETECTOR_KEYS = REPO / ".agent/work/2026-06-18-sam3-real-architecture-port/reference-key-shapes/detector.json"


def _reference() -> dict[str, list[int]]:
    keys = json.loads(DETECTOR_KEYS.read_text())
    return {k: v for k, v in keys.items() if k.startswith("detr_decoder.")}


def _param_shapes(model) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}


def _reduced() -> Sam3DETRDecoderConfig:
    return Sam3DETRDecoderConfig(
        hidden_size=16, num_layers=2, num_queries=5, num_attention_heads=4, intermediate_size=32
    )


# --- structural 1:1 map -------------------------------------------------------


def test_decoder_param_tree_is_247_keys():
    assert len(_param_shapes(Sam3DetrDecoder(Sam3DETRDecoderConfig()))) == 247


def test_decoder_maps_1to1():
    params = _param_shapes(Sam3DetrDecoder(Sam3DETRDecoderConfig()))
    reference = _reference()
    assert len(reference) == 247

    expected = {}
    for ref_key, ref_shape in reference.items():
        local = remap_sam3_decoder_real_key(ref_key)
        assert local is not None, ref_key
        expected[local] = tuple(ref_shape)  # decoder has no convs -> no transposes

    assert set(params) == set(expected)
    mismatched = {k: (params[k], expected[k]) for k in params if params[k] != expected[k]}
    assert not mismatched, mismatched


def test_decoder_special_shapes():
    params = _param_shapes(Sam3DetrDecoder(Sam3DETRDecoderConfig()))
    assert params["query_embed.weight"] == (200, 256)
    assert params["reference_points.weight"] == (200, 4)
    assert params["presence_token.weight"] == (1, 256)
    assert params["box_head.layer3.weight"] == (4, 256)
    assert params["presence_head.layer3.weight"] == (1, 256)
    assert params["box_rpb_embed_x.layer2.weight"] == (8, 256)  # one bias per attention head
    assert params["ref_point_head.layer1.weight"] == (256, 512)  # 2*hidden -> hidden


# --- converter + loader round-trip --------------------------------------------


def _write_npz(tmp_path: Path, state: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, **state)
    return path


def _synthetic_checkpoint(model) -> dict[str, np.ndarray]:
    return {
        f"detector_model.detr_decoder.{local}": np.zeros(tuple(v.shape), np.float32)
        for local, v in dict(tree_flatten(model.parameters())).items()
    }


def test_converter_loader_round_trips(tmp_path):
    model = Sam3DetrDecoder(Sam3DETRDecoderConfig())
    params = dict(tree_flatten(model.parameters()))
    state = _synthetic_checkpoint(model)
    state["detector_model.vision_encoder.backbone.layer_norm.weight"] = np.zeros((1024,), np.float32)
    loaded = load_sam3_decoder_real_weights(model, _write_npz(tmp_path, state))
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_loader_rejects_incomplete_checkpoint(tmp_path):
    model = Sam3DetrDecoder(Sam3DETRDecoderConfig())
    partial = dict(list(_synthetic_checkpoint(model).items())[:-1])
    with pytest.raises(ValueError, match="missing SAM3 detr decoder params"):
        load_sam3_decoder_real_weights(model, _write_npz(tmp_path, partial))


# --- reduced-size forward -----------------------------------------------------


def _inputs(config, batch=2, height=3, width=3, seq=4):
    num_tokens = height * width
    vision = mx.random.normal((batch, num_tokens, config.hidden_size))
    text = mx.random.normal((batch, seq, config.hidden_size))
    vision_pos = mx.random.normal((batch, num_tokens, config.hidden_size))
    return vision, text, vision_pos, [(height, width)]


def test_decoder_forward_shapes_and_ranges():
    config = _reduced()
    model = Sam3DetrDecoder(config)
    mx.eval(model.parameters())
    vision, text, vision_pos, spatial = _inputs(config)

    out = model(vision, text, vision_pos, spatial_shapes=spatial)
    mx.eval(out.intermediate_hidden_states, out.reference_boxes, out.presence_logits)

    assert tuple(out.intermediate_hidden_states.shape) == (config.num_layers, 2, config.num_queries, config.hidden_size)
    assert tuple(out.reference_boxes.shape) == (config.num_layers, 2, config.num_queries, 4)
    assert tuple(out.presence_logits.shape) == (config.num_layers, 2, 1)

    assert bool((mx.min(out.reference_boxes) >= 0.0).item())
    assert bool((mx.max(out.reference_boxes) <= 1.0).item())
    assert bool((mx.max(mx.abs(out.presence_logits)) <= 10.0).item())
    assert bool(mx.all(mx.isfinite(out.intermediate_hidden_states)).item())


def test_decoder_runs_without_rpb_when_multilevel():
    # Multi-level spatial_shapes disables the single-level box RPB path.
    config = _reduced()
    model = Sam3DetrDecoder(config)
    mx.eval(model.parameters())
    vision, text, vision_pos, _ = _inputs(config)
    out = model(vision, text, vision_pos, spatial_shapes=[(3, 3), (2, 2)])
    mx.eval(out.intermediate_hidden_states)
    assert tuple(out.intermediate_hidden_states.shape) == (config.num_layers, 2, config.num_queries, config.hidden_size)


def test_decoder_is_deterministic():
    config = _reduced()
    model = Sam3DetrDecoder(config)
    mx.eval(model.parameters())
    vision, text, vision_pos, spatial = _inputs(config)
    first = model(vision, text, vision_pos, spatial_shapes=spatial).intermediate_hidden_states
    second = model(vision, text, vision_pos, spatial_shapes=spatial).intermediate_hidden_states
    assert bool(mx.all(first == second).item())
