"""Slice 10: faithful SAM3 tracker prompt encoder + SAM2 mask decoder (145 tensors).

Weight-free verification:
- ``Sam3TrackerPromptEncoder`` (14) and ``Sam3TrackerMaskDecoder`` (131) map 1:1
  onto the ``video.json`` reference keys, with conv (incl. ConvTranspose ``upscale_conv``)
  layout transposes;
- the converters + loaders round-trip synthetic checkpoints;
- reduced-size forwards run (TwoWay transformer mask decoder; dense prompt path).

Real numeric parity runs out-of-sandbox via the streaming assembly (slice 11).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.models.sam3.real_convert import (
    convert_reference_shape,
    load_sam3_prompt_encoder_real_weights,
    load_sam3_tracker_mask_decoder_real_weights,
    remap_sam3_prompt_encoder_real_key,
    remap_sam3_tracker_mask_decoder_real_key,
)
from mlx_cv.models.sam3.real_tracker_decoder import Sam3TrackerMaskDecoder, Sam3TrackerPromptEncoder
from mlx_cv.models.sam3.real_video_config import Sam3TrackerMaskDecoderConfig, Sam3TrackerPromptEncoderConfig

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


def test_prompt_encoder_maps_1to1():
    _assert_maps_1to1(
        Sam3TrackerPromptEncoder(Sam3TrackerPromptEncoderConfig()),
        "tracker_model.prompt_encoder.",
        remap_sam3_prompt_encoder_real_key,
        14,
    )


def test_tracker_mask_decoder_maps_1to1():
    _assert_maps_1to1(
        Sam3TrackerMaskDecoder(Sam3TrackerMaskDecoderConfig()),
        "tracker_model.mask_decoder.",
        remap_sam3_tracker_mask_decoder_real_key,
        131,
    )


def test_upscale_conv_is_convtranspose_transposed():
    # upscale_conv* are ConvTranspose2d: torch [in, out, kH, kW] -> MLX [out, kH, kW, in].
    assert convert_reference_shape("upscale_conv1.weight", (256, 64, 2, 2)) == (64, 2, 2, 256)
    assert convert_reference_shape("upscale_conv2.weight", (64, 32, 2, 2)) == (32, 2, 2, 64)
    # conv_s0/s1 are plain Conv2d 1x1.
    assert convert_reference_shape("conv_s0.weight", (32, 256, 1, 1)) == (32, 1, 1, 256)


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
            inv = (3, 0, 1, 2) if ("upscale_conv" in local or ".scale_layers." in local) else (0, 3, 1, 2)
            arr = np.transpose(arr, inv)
        state[f"{prefix}{local}"] = arr
    return state


def test_prompt_encoder_loader_round_trips(tmp_path):
    model = Sam3TrackerPromptEncoder(Sam3TrackerPromptEncoderConfig())
    params = dict(tree_flatten(model.parameters()))
    state = _synthetic(model, "tracker_model.prompt_encoder.")
    loaded = load_sam3_prompt_encoder_real_weights(model, _write_npz(tmp_path, state))
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_tracker_mask_decoder_loader_round_trips(tmp_path):
    model = Sam3TrackerMaskDecoder(Sam3TrackerMaskDecoderConfig())
    params = dict(tree_flatten(model.parameters()))
    state = _synthetic(model, "tracker_model.mask_decoder.")
    state["detector_model.vision_encoder.backbone.layer_norm.weight"] = np.zeros((1024,), np.float32)
    loaded = load_sam3_tracker_mask_decoder_real_weights(model, _write_npz(tmp_path, state))
    assert set(dict(tree_flatten(loaded.parameters()))) == set(params)


def test_tracker_mask_decoder_loader_rejects_incomplete(tmp_path):
    model = Sam3TrackerMaskDecoder(Sam3TrackerMaskDecoderConfig())
    partial = dict(list(_synthetic(model, "tracker_model.mask_decoder.").items())[:-1])
    with pytest.raises(ValueError, match="missing SAM3 tracker mask decoder params"):
        load_sam3_tracker_mask_decoder_real_weights(model, _write_npz(tmp_path, partial))


# --- reduced forwards ---------------------------------------------------------


def test_prompt_encoder_dense_path():
    model = Sam3TrackerPromptEncoder(Sam3TrackerPromptEncoderConfig())
    mx.eval(model.parameters())
    dense = model(None, batch_size=2)
    mx.eval(dense)
    h = w = model.image_embedding_size[0]
    assert tuple(dense.shape) == (2, h, w, model.hidden_size)
    masks = mx.random.normal((1, 64, 64, 1))
    embedded = model(masks)
    mx.eval(embedded)
    assert embedded.shape[-1] == model.hidden_size


def test_tracker_mask_decoder_forward_shapes():
    config = Sam3TrackerMaskDecoderConfig()
    model = Sam3TrackerMaskDecoder(config)
    mx.eval(model.parameters())

    batch, pb, h, w, c = 1, 1, 8, 8, config.hidden_size
    image = mx.random.normal((batch, h, w, c))
    image_pe = mx.random.normal((batch, h, w, c))
    sparse = mx.random.normal((batch, pb, 2, c))
    dense = mx.random.normal((batch, h, w, c))
    feat_s0 = mx.random.normal((batch, 4 * h, 4 * w, c // 8))
    feat_s1 = mx.random.normal((batch, 2 * h, 2 * w, c // 4))

    out = model(image, image_pe, sparse, dense, [feat_s0, feat_s1], multimask_output=True)
    mx.eval(out.masks, out.iou_pred, out.object_score_logits)
    assert tuple(out.masks.shape) == (batch, pb, config.num_multimask_outputs, 4 * h, 4 * w)
    assert tuple(out.iou_pred.shape) == (batch, pb, config.num_multimask_outputs)
    assert tuple(out.object_score_logits.shape) == (batch, pb, 1)
    assert bool(mx.all(mx.isfinite(out.masks)).item())
