from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]


def _load_tool(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rfdetr_checkpoint = _load_tool("rfdetr_checkpoint", "tools/rfdetr_checkpoint.py")
rfdetr_convert = _load_tool("rfdetr_convert_checkpoint", "tools/rfdetr_convert_checkpoint.py")


def _converted_weights_for_real_forward(*, environ=None, cache_root=None) -> Path:
    required = rfdetr_checkpoint.required_gate_enabled(environ)
    try:
        converted = rfdetr_convert.resolve_rfdetr_nano_converted_weights(
            environ=environ,
            cache_root=cache_root,
            required=required,
        )
    except rfdetr_convert.RFDETRConversionDependencyError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    except (rfdetr_checkpoint.CheckpointError, rfdetr_convert.RFDETRConversionError) as exc:
        if required:
            pytest.fail(str(exc))
        raise
    if converted is None:
        if required:
            pytest.fail("RF-DETR Nano real forward skipped in required mode")
        pytest.skip("RF-DETR Nano checkpoint not configured")
    return converted


def _runtime_imports_for_real_forward():
    required = rfdetr_checkpoint.required_gate_enabled()
    try:
        from mlx_cv.core.types import Detections, Result
        from mlx_cv.parity import rfdetr_fixed_image, rfdetr_tap_order
        from mlx_cv.parity.rfdetr_real import (
            build_rfdetr_nano_local_model,
            capture_rfdetr_nano_local,
            rfdetr_nano_image_size,
        )
    except Exception as exc:
        if required:
            pytest.fail(f"RF-DETR Nano real forward requires the MLX runtime: {exc}")
        pytest.skip(f"RF-DETR Nano real forward requires the MLX runtime: {exc}")
    return (
        Detections,
        Result,
        build_rfdetr_nano_local_model,
        capture_rfdetr_nano_local,
        rfdetr_fixed_image,
        rfdetr_nano_image_size,
        rfdetr_tap_order,
    )


def _capture_or_skip():
    converted = _converted_weights_for_real_forward()
    required = rfdetr_checkpoint.required_gate_enabled()
    imports = _runtime_imports_for_real_forward()
    _, _, build_model, capture_local, *_ = imports
    try:
        model = build_model(converted)
        capture = capture_local(model)
    except Exception as exc:
        if required:
            pytest.fail(f"RF-DETR Nano local real forward failed: {exc}")
        raise
    return capture, imports


def _upstream_style_input_tensor_or_skip(image: np.ndarray, image_size: int) -> np.ndarray:
    required = rfdetr_checkpoint.required_gate_enabled()
    try:
        import torchvision.transforms.functional as tvf
        from PIL import Image
    except Exception as exc:
        if required:
            pytest.fail(f"RF-DETR Nano input tensor check requires torchvision and PIL: {exc}")
        pytest.skip(f"RF-DETR Nano input tensor check requires torchvision and PIL: {exc}")

    tensor = tvf.to_tensor(Image.fromarray(image))
    resized = tvf.resize(tensor, [image_size, image_size])
    normalized = tvf.normalize(resized, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return normalized.unsqueeze(0).numpy()


def test_optional_no_checkpoint_real_forward_skips_cleanly(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="checkpoint not configured"):
        _converted_weights_for_real_forward(environ={}, cache_root=tmp_path)


def test_required_no_checkpoint_real_forward_fails_instead_of_skipping(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="checkpoint is required but missing"):
        _converted_weights_for_real_forward(
            environ={rfdetr_checkpoint.RFDETR_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
        )


def test_required_missing_converted_real_forward_fails_instead_of_skipping(tmp_path):
    missing = tmp_path / "missing-rfdetr-nano.npz"

    with pytest.raises(pytest.fail.Exception, match="not a file"):
        _converted_weights_for_real_forward(
            environ={
                rfdetr_checkpoint.RFDETR_REQUIRED_GATE_ENV: "1",
                rfdetr_convert.RFDETR_NANO_CONVERTED_ENV: str(missing),
            },
            cache_root=tmp_path,
        )


def test_real_rfdetr_nano_local_forward_capture_runs_fixed_image():
    capture, imports = _capture_or_skip()
    Detections, Result, _, _, fixed_image_fn, nano_image_size, tap_order = imports
    fixed_image = fixed_image_fn()

    assert isinstance(capture.result, Result)
    assert isinstance(capture.result.detections, Detections)
    np.testing.assert_array_equal(capture.input_image, fixed_image)
    assert capture.input_image.shape == (28, 28, 3)
    assert capture.input_image.dtype == np.uint8

    assert nano_image_size() == 384
    assert capture.input_tensor.shape == (1, 3, 384, 384)
    assert capture.input_tensor.dtype == np.float32
    expected_input_tensor = _upstream_style_input_tensor_or_skip(fixed_image, 384)
    np.testing.assert_allclose(capture.input_tensor, expected_input_tensor, rtol=0.0, atol=1e-6)
    assert capture.result.image_size == fixed_image.shape[:2]

    assert capture.raw_logits.shape == (1, 300, 91)
    assert capture.raw_boxes.shape == (1, 300, 4)
    assert np.all(np.isfinite(capture.raw_logits))
    assert np.all(np.isfinite(capture.raw_boxes))

    detections = capture.result.detections
    assert len(detections) == 300
    assert capture.boxes.shape == (300, 4)
    assert capture.scores.shape == (300,)
    assert capture.class_ids.shape == (300,)
    assert detections.labels is None
    assert np.issubdtype(capture.class_ids.dtype, np.integer)
    assert np.all(capture.class_ids >= 0)
    assert np.all(capture.class_ids < 91)
    assert np.all(np.isfinite(capture.boxes))
    assert np.all(np.isfinite(capture.scores))

    with np.errstate(over="ignore"):
        probs = 1.0 / (1.0 + np.exp(-capture.raw_logits[0].astype(np.float64)))
    flat = probs.reshape(-1)
    order = np.argsort(flat)[::-1][:300]
    expected_scores = flat[order]
    expected_class_ids = (order % capture.raw_logits.shape[-1]).astype(np.int64)
    np.testing.assert_allclose(capture.scores, expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(capture.class_ids, expected_class_ids)

    expected_taps = tap_order(num_levels=1, num_layers=2, include_self_attention=True)
    assert list(capture.taps) == expected_taps
    assert capture.taps["projector.level_0"].shape == (1, 24, 24, 256)
    assert capture.taps["decoder.self_attention_0"].shape == (1, 300, 256)
    assert capture.taps["decoder.deformable_attention_0"].shape == (1, 300, 256)
    assert capture.taps["decoder.self_attention_1"].shape == (1, 300, 256)
    assert capture.taps["decoder.deformable_attention_1"].shape == (1, 300, 256)
    assert capture.taps["decoder.hidden_states"].shape == (1, 300, 256)
    assert capture.taps["head.logits"].shape == capture.raw_logits.shape
    assert capture.taps["head.boxes"].shape == capture.raw_boxes.shape
    assert capture.taps["result.boxes"].shape == (300, 4)
    assert capture.taps["result.scores"].shape == (300,)
    assert capture.taps["result.class_ids"].shape == (300,)
    assert capture.tap_gaps
