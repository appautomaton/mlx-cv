"""Slice 6: faithful SAM3 mask decoder (FPN pixel decoder + heads, 32 tensors).

Weight-free verification:
- the MLX ``Sam3MaskDecoder`` parameter tree maps 1:1 onto the ``mask_decoder.*``
  reference keys (committed ``detector.json``), with conv layout transposes;
- the converter + loader round-trip a synthetic checkpoint;
- a reduced-size forward runs end-to-end with coherent shapes (top-down FPN with
  nearest upsample + GroupNorm, instance/semantic Conv2d 1x1 heads, einsum masks),
  with and without prompt cross-attention, and is deterministic.

This completes the image detector subsystems (SPEC AC3). Real numeric tap parity
runs out-of-sandbox with the gated checkpoint (slice 7).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.models.sam3.real_config import Sam3MaskDecoderConfig
from mlx_cv.models.sam3.real_convert import (
    convert_reference_shape,
    load_sam3_mask_decoder_real_weights,
    remap_sam3_mask_decoder_real_key,
)
from mlx_cv.models.sam3.real_mask import Sam3MaskDecoder

REPO = Path(__file__).resolve().parents[1]
DETECTOR_KEYS = REPO / ".agent/work/2026-06-18-sam3-real-architecture-port/reference-key-shapes/detector.json"


def _reference() -> dict[str, list[int]]:
    keys = json.loads(DETECTOR_KEYS.read_text())
    return {k: v for k, v in keys.items() if k.startswith("mask_decoder.")}


def _param_shapes(model) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}


# --- structural 1:1 map -------------------------------------------------------


def test_mask_decoder_param_tree_is_32_keys():
    assert len(_param_shapes(Sam3MaskDecoder(Sam3MaskDecoderConfig()))) == 32


def test_mask_decoder_maps_1to1():
    params = _param_shapes(Sam3MaskDecoder(Sam3MaskDecoderConfig()))
    reference = _reference()
    assert len(reference) == 32

    expected = {}
    for ref_key, ref_shape in reference.items():
        local = remap_sam3_mask_decoder_real_key(ref_key)
        assert local is not None, ref_key
        expected[local] = convert_reference_shape(local, tuple(ref_shape))

    assert set(params) == set(expected)
    mismatched = {k: (params[k], expected[k]) for k in params if params[k] != expected[k]}
    assert not mismatched, mismatched


def test_mask_decoder_conv_transposes():
    # 1x1 instance / semantic heads and 3x3 pixel-decoder convs: torch NCHW -> MLX NHWC.
    assert convert_reference_shape("instance_projection.weight", (256, 256, 1, 1)) == (256, 1, 1, 256)
    assert convert_reference_shape("semantic_projection.weight", (1, 256, 1, 1)) == (1, 1, 1, 256)
    assert convert_reference_shape("pixel_decoder.conv_layers.0.weight", (256, 256, 3, 3)) == (256, 3, 3, 256)
    # GroupNorm tensors are 1D -> untouched.
    assert convert_reference_shape("pixel_decoder.norms.0.weight", (256,)) == (256,)


# --- converter + loader round-trip --------------------------------------------


def _write_npz(tmp_path: Path, state: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, **state)
    return path


def _synthetic_checkpoint(model) -> dict[str, np.ndarray]:
    state = {}
    for local, value in dict(tree_flatten(model.parameters())).items():
        arr = np.zeros(tuple(value.shape), np.float32)
        if arr.ndim == 4 and local.endswith(".weight"):  # Conv2d -> inverse of (0,2,3,1)
            arr = np.transpose(arr, (0, 3, 1, 2))
        state[f"detector_model.mask_decoder.{local}"] = arr
    return state


def test_converter_loader_round_trips(tmp_path):
    model = Sam3MaskDecoder(Sam3MaskDecoderConfig())
    params = dict(tree_flatten(model.parameters()))
    state = _synthetic_checkpoint(model)
    state["detector_model.vision_encoder.backbone.layer_norm.weight"] = np.zeros((1024,), np.float32)
    loaded = load_sam3_mask_decoder_real_weights(model, _write_npz(tmp_path, state))
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_loader_rejects_incomplete_checkpoint(tmp_path):
    model = Sam3MaskDecoder(Sam3MaskDecoderConfig())
    partial = dict(list(_synthetic_checkpoint(model).items())[:-1])
    with pytest.raises(ValueError, match="missing SAM3 mask decoder params"):
        load_sam3_mask_decoder_real_weights(model, _write_npz(tmp_path, partial))


# --- reduced-size forward -----------------------------------------------------


def _reduced_inputs(config, batch=2, num_queries=5, prompt_len=4):
    # Backbone features high -> low resolution; encoder tokens match the finest (last) level.
    backbone = [
        mx.random.normal((batch, 8, 8, config.hidden_size)),
        mx.random.normal((batch, 4, 4, config.hidden_size)),
        mx.random.normal((batch, 2, 2, config.hidden_size)),
    ]
    encoder = mx.random.normal((batch, 2 * 2, config.hidden_size))
    queries = mx.random.normal((batch, num_queries, config.hidden_size))
    prompt = mx.random.normal((batch, prompt_len, config.hidden_size))
    return backbone, encoder, queries, prompt


def test_mask_decoder_forward_shapes_with_prompt():
    config = Sam3MaskDecoderConfig(hidden_size=16, num_upsampling_stages=2, num_attention_heads=4)
    model = Sam3MaskDecoder(config)
    mx.eval(model.parameters())
    backbone, encoder, queries, prompt = _reduced_inputs(config)

    out = model(queries, backbone, encoder, prompt_features=prompt)
    mx.eval(out.pred_masks, out.semantic_seg)
    # Output spatial == finest backbone level (8x8).
    assert tuple(out.pred_masks.shape) == (2, 5, 8, 8)
    assert tuple(out.semantic_seg.shape) == (2, 1, 8, 8)
    assert bool(mx.all(mx.isfinite(out.pred_masks)).item())


def test_mask_decoder_forward_without_prompt():
    config = Sam3MaskDecoderConfig(hidden_size=16, num_upsampling_stages=2, num_attention_heads=4)
    model = Sam3MaskDecoder(config)
    mx.eval(model.parameters())
    backbone, encoder, queries, _ = _reduced_inputs(config)
    out = model(queries, backbone, encoder, prompt_features=None)
    mx.eval(out.pred_masks)
    assert tuple(out.pred_masks.shape) == (2, 5, 8, 8)


def test_mask_decoder_is_deterministic():
    config = Sam3MaskDecoderConfig(hidden_size=16, num_upsampling_stages=2, num_attention_heads=4)
    model = Sam3MaskDecoder(config)
    mx.eval(model.parameters())
    backbone, encoder, queries, prompt = _reduced_inputs(config)
    first = model(queries, backbone, encoder, prompt_features=prompt).pred_masks
    second = model(queries, backbone, encoder, prompt_features=prompt).pred_masks
    assert bool(mx.all(first == second).item())
