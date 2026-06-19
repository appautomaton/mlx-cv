"""Slice 4: faithful SAM3 DETR encoder + geometry encoder + scoring (260 tensors).

Weight-free verification, mirroring the earlier real-subsystem tests:
- each subsystem's MLX parameter tree maps 1:1 onto its reference namespace in the
  committed ``detector.json`` (detr_encoder 156, geometry_encoder 94,
  dot_product_scoring 10);
- the converter + loader round-trip a synthetic checkpoint (incl. the geometry
  ``boxes_pool_project`` Conv2d layout transpose);
- reduced-size forwards run for the critical-path subsystems (DETR encoder, scoring)
  and the geometry transformer stack; the geometry box-prompt entry point raises a
  precise not-yet-ported error (it is off the text-prompt gate path).

Real numeric tap parity runs out-of-sandbox with the gated checkpoint (slice 7).
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
    Sam3GeometryEncoderConfig,
)
from mlx_cv.models.sam3.real_convert import (
    convert_reference_shape,
    convert_sam3_namespace_state_dict,
    load_sam3_detr_encoder_real_weights,
    load_sam3_geometry_real_weights,
    load_sam3_scoring_real_weights,
    remap_sam3_detr_encoder_real_key,
    remap_sam3_geometry_real_key,
    remap_sam3_scoring_real_key,
)
from mlx_cv.models.sam3.real_detr import Sam3DetrEncoder, Sam3DotProductScoring
from mlx_cv.models.sam3.real_geometry import Sam3GeometryEncoder

REPO = Path(__file__).resolve().parents[1]
DETECTOR_KEYS = REPO / ".agent/work/2026-06-18-sam3-real-architecture-port/reference-key-shapes/detector.json"


def _reference(namespace: str) -> dict[str, list[int]]:
    keys = json.loads(DETECTOR_KEYS.read_text())
    return {k: v for k, v in keys.items() if k.startswith(namespace + ".")}


def _param_shapes(model) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}


def _assert_namespace_maps_1to1(model, namespace, remap, expected_count):
    params = _param_shapes(model)
    reference = _reference(namespace)
    assert len(reference) == expected_count, (namespace, len(reference))

    expected: dict[str, tuple[int, ...]] = {}
    for ref_key, ref_shape in reference.items():
        local = remap(ref_key)
        assert local is not None, ref_key
        expected[local] = convert_reference_shape(local, tuple(ref_shape))

    assert set(params) == set(expected)
    mismatched = {k: (params[k], expected[k]) for k in params if params[k] != expected[k]}
    assert not mismatched, mismatched


# --- structural 1:1 parameter maps --------------------------------------------


def test_detr_encoder_maps_1to1():
    _assert_namespace_maps_1to1(
        Sam3DetrEncoder(Sam3DETREncoderConfig()), "detr_encoder", remap_sam3_detr_encoder_real_key, 156
    )


def test_scoring_maps_1to1():
    _assert_namespace_maps_1to1(
        Sam3DotProductScoring(Sam3DETRDecoderConfig()), "dot_product_scoring", remap_sam3_scoring_real_key, 10
    )


def test_geometry_maps_1to1():
    _assert_namespace_maps_1to1(
        Sam3GeometryEncoder(Sam3GeometryEncoderConfig()), "geometry_encoder", remap_sam3_geometry_real_key, 94
    )


def test_geometry_boxes_pool_project_is_conv_transposed():
    # boxes_pool_project is a Conv2d: torch [out, in, kH, kW] -> MLX [out, kH, kW, in].
    assert convert_reference_shape("boxes_pool_project.weight", (256, 256, 7, 7)) == (256, 7, 7, 256)


def test_remaps_are_namespace_scoped():
    assert remap_sam3_detr_encoder_real_key("geometry_encoder.final_proj.weight") is None
    assert remap_sam3_scoring_real_key("detr_encoder.layers.0.mlp.fc1.weight") is None
    assert (
        remap_sam3_geometry_real_key("detector_model.geometry_encoder.cls_embed.weight")
        == "cls_embed.weight"
    )


# --- converter + loader round-trip --------------------------------------------


def _write_npz(tmp_path: Path, state: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, **state)
    return path


def _synthetic_checkpoint(model, namespace: str) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for local, value in dict(tree_flatten(model.parameters())).items():
        arr = np.zeros(tuple(value.shape), np.float32)
        if arr.ndim == 4 and local.endswith(".weight"):  # Conv2d -> inverse of (0,2,3,1)
            arr = np.transpose(arr, (0, 3, 1, 2))
        state[f"detector_model.{namespace}.{local}"] = arr
    return state


@pytest.mark.parametrize(
    ("ctor", "namespace", "remap", "loader", "count"),
    [
        (lambda: Sam3DetrEncoder(Sam3DETREncoderConfig()), "detr_encoder", remap_sam3_detr_encoder_real_key, load_sam3_detr_encoder_real_weights, 156),
        (lambda: Sam3DotProductScoring(Sam3DETRDecoderConfig()), "dot_product_scoring", remap_sam3_scoring_real_key, load_sam3_scoring_real_weights, 10),
        (lambda: Sam3GeometryEncoder(Sam3GeometryEncoderConfig()), "geometry_encoder", remap_sam3_geometry_real_key, load_sam3_geometry_real_weights, 94),
    ],
)
def test_converter_loader_round_trips(tmp_path, ctor, namespace, remap, loader, count):
    model = ctor()
    params = dict(tree_flatten(model.parameters()))
    state = _synthetic_checkpoint(model, namespace)
    state["detector_model.vision_encoder.backbone.layer_norm.weight"] = np.zeros((1024,), np.float32)

    converted = convert_sam3_namespace_state_dict(state, remap)
    assert len(converted) == count
    assert {k for k, _ in converted} == set(params)

    loaded = loader(model, _write_npz(tmp_path, state))
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_loader_rejects_incomplete_checkpoint(tmp_path):
    model = Sam3DetrEncoder(Sam3DETREncoderConfig())
    full = _synthetic_checkpoint(model, "detr_encoder")
    partial = dict(list(full.items())[:-1])
    with pytest.raises(ValueError, match="missing SAM3 detr encoder params"):
        load_sam3_detr_encoder_real_weights(model, _write_npz(tmp_path, partial))


# --- reduced-size forwards ----------------------------------------------------


def test_detr_encoder_forward_shapes():
    config = Sam3DETREncoderConfig(hidden_size=16, num_layers=2, num_attention_heads=4, intermediate_size=32)
    model = Sam3DetrEncoder(config)
    mx.eval(model.parameters())

    batch = 2
    levels = [mx.random.normal((batch, 4, 4, 16)), mx.random.normal((batch, 2, 2, 16))]
    pos = [mx.random.normal((batch, 4, 4, 16)), mx.random.normal((batch, 2, 2, 16))]
    text = mx.random.normal((batch, 5, 16))
    out = model(levels, text, pos)
    mx.eval(out.last_hidden_state)
    assert tuple(out.last_hidden_state.shape) == (batch, 4 * 4 + 2 * 2, 16)
    assert bool(mx.all(mx.isfinite(out.last_hidden_state)).item())


def test_scoring_forward_shapes_and_clamp():
    config = Sam3DETRDecoderConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4)
    model = Sam3DotProductScoring(config)
    mx.eval(model.parameters())

    num_layers, batch, num_queries, seq = 3, 2, 4, 5
    decoder_hidden = mx.random.normal((num_layers, batch, num_queries, 16))
    text = mx.random.normal((batch, seq, 16))
    scores = model(decoder_hidden, text)
    mx.eval(scores)
    assert tuple(scores.shape) == (num_layers, batch, num_queries, 1)
    assert bool((mx.max(mx.abs(scores)) <= 12.0).item())


def test_geometry_transformer_stack_runs():
    config = Sam3GeometryEncoderConfig(hidden_size=16, num_layers=2, num_attention_heads=4, intermediate_size=32)
    model = Sam3GeometryEncoder(config)
    mx.eval(model.parameters())

    batch, num_prompts, num_vision = 2, 3, 9
    prompt = mx.random.normal((batch, num_prompts, 16))
    vision = mx.random.normal((batch, num_vision, 16))
    vision_pos = mx.random.normal((batch, num_vision, 16))
    out = model.run_layers(prompt, vision, vision_pos)
    mx.eval(out)
    assert tuple(out.shape) == (batch, num_prompts, 16)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_geometry_box_prompt_forward_is_honest_not_yet_ported():
    model = Sam3GeometryEncoder(Sam3GeometryEncoderConfig(hidden_size=16, num_layers=1))
    with pytest.raises(NotImplementedError, match="roi_align"):
        model("box_embeddings", "box_mask", "box_labels", "img_feats")
