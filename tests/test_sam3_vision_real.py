"""Slice 2: faithful SAM3 vision encoder (windowed-RoPE ViT + FPN, 538 tensors).

Weight-free verification:
- the MLX ``Sam3VisionModel`` parameter tree maps 1:1 onto the real
  ``vision_encoder.*`` keys (committed ``detector.json`` shape spec), with the conv
  layout transposes applied;
- the converter + loader round-trip a full synthetic checkpoint (built from the
  model's own param shapes in torch layout) with an exact shape match;
- a reduced-size forward runs end-to-end and produces coherent shapes (tiling,
  windowed/global attention alternation, and all four FPN levels exercised).

The real numeric parity tap runs out-of-sandbox with the gated checkpoint (slice 7).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.models.sam3.real_config import Sam3ViTConfig, Sam3VisionConfig
from mlx_cv.models.sam3.real_convert import (
    convert_reference_shape,
    convert_sam3_vision_real_state_dict,
    load_sam3_vision_real_weights,
    remap_sam3_vision_real_key,
)
from mlx_cv.models.sam3.real_vision import Sam3VisionModel

REPO = Path(__file__).resolve().parents[1]
DETECTOR_KEYS = REPO / ".agent/work/2026-06-18-sam3-real-architecture-port/reference-key-shapes/detector.json"


def _reference_vision_shapes() -> dict[str, list[int]]:
    keys = json.loads(DETECTOR_KEYS.read_text())
    return {k: v for k, v in keys.items() if k.startswith("vision_encoder.")}


def _model_param_shapes(model: Sam3VisionModel) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}


def _reduced_config() -> Sam3VisionConfig:
    backbone = Sam3ViTConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=112,  # 8x8 patches
        patch_size=14,
        window_size=4,
        global_attn_indexes=(1, 3),
        pretrain_image_size=28,  # 2x2 pretrain grid -> forces position-embedding tiling
    )
    return Sam3VisionConfig(backbone=backbone, fpn_hidden_size=16, scale_factors=(4.0, 2.0, 1.0, 0.5))


# --- structural 1:1 parameter map ---------------------------------------------


def test_vision_param_tree_is_538_keys():
    model = Sam3VisionModel(Sam3VisionConfig())
    params = _model_param_shapes(model)
    assert len(params) == 538


def test_vision_param_tree_matches_reference_keys_and_shapes():
    model = Sam3VisionModel(Sam3VisionConfig())
    params = _model_param_shapes(model)
    reference = _reference_vision_shapes()
    assert len(reference) == 538

    expected: dict[str, tuple[int, ...]] = {}
    for ref_key, ref_shape in reference.items():
        local = remap_sam3_vision_real_key(ref_key)
        assert local is not None, ref_key
        expected[local] = convert_reference_shape(local, tuple(ref_shape))

    assert set(params) == set(expected)
    mismatched = {k: (params[k], expected[k]) for k in params if params[k] != expected[k]}
    assert not mismatched, mismatched


def test_conv_weight_layout_transposes():
    # Conv2d patch projection: torch [out, in, kH, kW] -> MLX [out, kH, kW, in].
    assert convert_reference_shape(
        "backbone.embeddings.patch_embeddings.projection.weight", (1024, 3, 14, 14)
    ) == (1024, 14, 14, 3)
    # ConvTranspose2d FPN scale layer: torch [in, out, kH, kW] -> MLX [out, kH, kW, in].
    assert convert_reference_shape(
        "neck.fpn_layers.0.scale_layers.0.weight", (1024, 512, 2, 2)
    ) == (512, 2, 2, 1024)
    # Linear / norm / position-embedding tensors are not transposed.
    assert convert_reference_shape("backbone.layers.0.mlp.fc1.weight", (4736, 1024)) == (4736, 1024)
    assert convert_reference_shape("backbone.embeddings.position_embeddings", (1, 576, 1024)) == (1, 576, 1024)


def test_non_vision_keys_are_ignored_by_remap():
    assert remap_sam3_vision_real_key("text_encoder.text_model.encoder.layers.0.mlp.fc1.weight") is None
    assert remap_sam3_vision_real_key("detr_decoder.layers.0.self_attn.q_proj.weight") is None
    # An optional detector_model. prefix is stripped before matching.
    assert (
        remap_sam3_vision_real_key("detector_model.vision_encoder.backbone.layer_norm.weight")
        == "backbone.layer_norm.weight"
    )


# --- converter + loader round-trip (synthetic full checkpoint) ----------------


def test_converter_loader_round_trips_synthetic_checkpoint(tmp_path):
    model = Sam3VisionModel(Sam3VisionConfig())
    params = dict(tree_flatten(model.parameters()))

    # Build a reference checkpoint in torch layout by inverse-transposing each param.
    inverse = {(0, 2, 3, 1): (0, 3, 1, 2), (1, 2, 3, 0): (3, 0, 1, 2)}
    state: dict[str, np.ndarray] = {}
    for local, value in params.items():
        arr = np.zeros(tuple(value.shape), dtype=np.float32)
        if arr.ndim == 4 and local.endswith(".weight"):
            perm = (1, 2, 3, 0) if ".scale_layers." in local else (0, 2, 3, 1)
            arr = np.transpose(arr, inverse[perm])
        state[f"detector_model.vision_encoder.{local}"] = arr
    # A foreign-namespace key must be ignored by the vision converter.
    state["detector_model.text_encoder.text_model.final_layer_norm.weight"] = np.zeros((1024,), np.float32)

    converted = convert_sam3_vision_real_state_dict(state)
    assert len(converted) == 538
    assert {k for k, _ in converted} == set(params)
    for key, value in converted:
        assert tuple(value.shape) == tuple(params[key].shape)

    loaded = load_sam3_vision_real_weights(model, _write_npz(tmp_path, state))
    reloaded = dict(tree_flatten(loaded.parameters()))
    assert set(reloaded) == set(params)


def _write_npz(tmp_path: Path, state: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, **state)
    return path


def test_loader_rejects_incomplete_checkpoint(tmp_path):
    model = Sam3VisionModel(Sam3VisionConfig())
    params = dict(tree_flatten(model.parameters()))
    state = {}
    for local, value in list(params.items())[:-1]:  # drop one tensor
        arr = np.zeros(tuple(value.shape), dtype=np.float32)
        if arr.ndim == 4 and local.endswith(".weight"):
            inv = (3, 0, 1, 2) if ".scale_layers." in local else (0, 3, 1, 2)
            arr = np.transpose(arr, inv)
        state[f"detector_model.vision_encoder.{local}"] = arr
    path = _write_npz(tmp_path, state)
    with pytest.raises(ValueError, match="missing SAM3 vision params"):
        load_sam3_vision_real_weights(model, path)


# --- reduced-size forward smoke -----------------------------------------------


def test_reduced_forward_produces_coherent_shapes():
    config = _reduced_config()
    model = Sam3VisionModel(config)
    mx.eval(model.parameters())

    batch, grid = 2, config.backbone.image_size // config.backbone.patch_size  # 8
    pixel_values = mx.zeros((batch, config.backbone.num_channels, config.backbone.image_size, config.backbone.image_size))
    out = model(pixel_values)
    mx.eval(out.last_hidden_state, *out.fpn_hidden_states, *out.fpn_position_encoding)

    assert tuple(out.last_hidden_state.shape) == (batch, grid * grid, config.backbone.hidden_size)
    assert bool(mx.all(mx.isfinite(out.last_hidden_state)).item())

    # Four FPN levels at scales (4.0, 2.0, 1.0, 0.5) of the 8x8 spatial grid.
    expected_spatial = [grid * 4, grid * 2, grid, grid // 2]
    for level, hidden_state in enumerate(out.fpn_hidden_states):
        side = expected_spatial[level]
        assert tuple(hidden_state.shape) == (batch, side, side, config.fpn_hidden_size)
        assert bool(mx.all(mx.isfinite(hidden_state)).item())

    for level, pos in enumerate(out.fpn_position_encoding):
        side = expected_spatial[level]
        assert tuple(pos.shape) == (batch, side, side, config.fpn_hidden_size)


def test_reduced_forward_is_deterministic():
    config = _reduced_config()
    model = Sam3VisionModel(config)
    mx.eval(model.parameters())
    pixel_values = mx.random.normal((1, 3, config.backbone.image_size, config.backbone.image_size))
    first = model(pixel_values).last_hidden_state
    second = model(pixel_values).last_hidden_state
    assert bool(mx.all(first == second).item())
