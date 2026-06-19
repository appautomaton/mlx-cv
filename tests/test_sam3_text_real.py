"""Slice 3: faithful SAM3 text encoder (CLIP tower + projections, 391 tensors).

Weight-free verification, mirroring tests/test_sam3_vision_real.py:
- the MLX ``Sam3TextEncoder`` parameter tree maps 1:1 onto the real
  ``text_encoder.*`` / ``text_projection.*`` keys (committed ``detector.json``);
- the converter + loader round-trip a full synthetic checkpoint;
- a reduced-size forward runs end-to-end with coherent shapes, is deterministic,
  and respects causality (a later token cannot change earlier hidden states).

The real numeric tap parity runs out-of-sandbox with the gated checkpoint (slice 7).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.models.sam3.real_config import Sam3TextConfig
from mlx_cv.models.sam3.real_convert import (
    convert_sam3_text_real_state_dict,
    load_sam3_text_real_weights,
    remap_sam3_text_real_key,
)
from mlx_cv.models.sam3.real_text import Sam3TextEncoder

REPO = Path(__file__).resolve().parents[1]
DETECTOR_KEYS = REPO / ".agent/work/2026-06-18-sam3-real-architecture-port/reference-key-shapes/detector.json"


def _reference_text_shapes() -> dict[str, list[int]]:
    keys = json.loads(DETECTOR_KEYS.read_text())
    return {
        k: v
        for k, v in keys.items()
        if k.startswith("text_encoder.") or k.startswith("text_projection.")
    }


def _model_param_shapes(model: Sam3TextEncoder) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}


def _reduced_config() -> Sam3TextConfig:
    return Sam3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        projection_dim=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=12,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
    )


# --- structural 1:1 parameter map ---------------------------------------------


def test_text_param_tree_is_391_keys():
    model = Sam3TextEncoder(Sam3TextConfig())
    assert len(_model_param_shapes(model)) == 391


def test_text_param_tree_matches_reference_keys_and_shapes():
    model = Sam3TextEncoder(Sam3TextConfig())
    params = _model_param_shapes(model)
    reference = _reference_text_shapes()
    assert len(reference) == 391

    expected: dict[str, tuple[int, ...]] = {}
    for ref_key, ref_shape in reference.items():
        local = remap_sam3_text_real_key(ref_key)
        assert local is not None, ref_key
        expected[local] = tuple(ref_shape)  # text tensors are never transposed

    assert set(params) == set(expected)
    mismatched = {k: (params[k], expected[k]) for k in params if params[k] != expected[k]}
    assert not mismatched, mismatched


def test_inner_and_outer_text_projection_shapes():
    params = _model_param_shapes(Sam3TextEncoder(Sam3TextConfig()))
    # Inner CLIP projection (no bias) and outer SAM3 projection (with bias) coexist.
    assert params["text_encoder.text_projection.weight"] == (512, 1024)
    assert params["text_projection.weight"] == (256, 1024)
    assert params["text_projection.bias"] == (256,)


def test_remap_ignores_non_text_keys():
    assert remap_sam3_text_real_key("vision_encoder.backbone.layer_norm.weight") is None
    assert remap_sam3_text_real_key("detr_decoder.layers.0.self_attn.q_proj.weight") is None
    assert (
        remap_sam3_text_real_key("detector_model.text_encoder.text_model.final_layer_norm.bias")
        == "text_encoder.text_model.final_layer_norm.bias"
    )


# --- converter + loader round-trip --------------------------------------------


def _write_npz(tmp_path: Path, state: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, **state)
    return path


def test_converter_loader_round_trips_synthetic_checkpoint(tmp_path):
    model = Sam3TextEncoder(Sam3TextConfig())
    params = dict(tree_flatten(model.parameters()))

    state = {f"detector_model.{local}": np.zeros(tuple(v.shape), np.float32) for local, v in params.items()}
    state["detector_model.vision_encoder.backbone.layer_norm.weight"] = np.zeros((1024,), np.float32)

    converted = convert_sam3_text_real_state_dict(state)
    assert len(converted) == 391
    assert {k for k, _ in converted} == set(params)

    loaded = load_sam3_text_real_weights(model, _write_npz(tmp_path, state))
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_loader_rejects_incomplete_checkpoint(tmp_path):
    model = Sam3TextEncoder(Sam3TextConfig())
    params = dict(tree_flatten(model.parameters()))
    state = {
        f"detector_model.{local}": np.zeros(tuple(v.shape), np.float32)
        for local, v in list(params.items())[:-1]  # drop one tensor
    }
    with pytest.raises(ValueError, match="missing SAM3 text params"):
        load_sam3_text_real_weights(model, _write_npz(tmp_path, state))


# --- reduced-size forward -----------------------------------------------------


def test_reduced_forward_produces_coherent_shapes():
    config = _reduced_config()
    model = Sam3TextEncoder(config, detr_hidden_size=8)
    mx.eval(model.parameters())

    batch, seq = 2, 7
    input_ids = mx.array(np.random.randint(0, config.vocab_size, size=(batch, seq)).astype(np.int32))
    out = model(input_ids)
    mx.eval(out.last_hidden_state, out.pooler_output, out.text_embeds)

    assert tuple(out.last_hidden_state.shape) == (batch, seq, config.hidden_size)
    assert tuple(out.pooler_output.shape) == (batch, seq, 8)
    assert tuple(out.text_embeds.shape) == (batch, config.projection_dim)
    assert bool(mx.all(mx.isfinite(out.pooler_output)).item())


def test_reduced_forward_is_deterministic():
    config = _reduced_config()
    model = Sam3TextEncoder(config, detr_hidden_size=8)
    mx.eval(model.parameters())
    input_ids = mx.array(np.random.randint(0, config.vocab_size, size=(1, 7)).astype(np.int32))
    first = model(input_ids).pooler_output
    second = model(input_ids).pooler_output
    assert bool(mx.all(first == second).item())


def test_attention_is_causal():
    config = _reduced_config()
    model = Sam3TextEncoder(config, detr_hidden_size=8)
    mx.eval(model.parameters())

    base = np.random.randint(0, config.vocab_size, size=(1, 7)).astype(np.int32)
    altered = base.copy()
    altered[0, -1] = (altered[0, -1] + 1) % config.vocab_size  # change only the last token

    first = model(mx.array(base)).last_hidden_state
    second = model(mx.array(altered)).last_hidden_state
    mx.eval(first, second)
    # Causality: positions before the changed last token are unaffected.
    assert bool(mx.allclose(first[:, :-1], second[:, :-1], atol=1e-5).item())
    # The changed position does differ (sanity that the edit took effect).
    assert not bool(mx.allclose(first[:, -1], second[:, -1], atol=1e-5).item())
