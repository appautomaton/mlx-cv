from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
_ATOL = 1e-4
_RTOL = 1e-4
_BOX_ATOL = 2e-4
_BOX_RTOL = 5e-4


def _load_tool(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rfdetr_checkpoint = _load_tool("rfdetr_checkpoint", "tools/rfdetr_checkpoint.py")
rfdetr_convert = _load_tool("rfdetr_convert_checkpoint", "tools/rfdetr_convert_checkpoint.py")
rfdetr_upstream = _load_tool("rfdetr_upstream", "tools/rfdetr_upstream.py")


def _checkpoint_or_skip(*, environ=None, cache_root=None):
    required = rfdetr_checkpoint.required_gate_enabled(environ)
    try:
        info = rfdetr_checkpoint.resolve_rfdetr_nano_checkpoint(
            environ=environ,
            cache_root=cache_root,
            required=required,
        )
    except rfdetr_checkpoint.CheckpointError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    if info is None:
        if required:
            pytest.fail("RF-DETR Nano upstream parity skipped in required mode")
        pytest.skip("RF-DETR Nano checkpoint not configured")
    return info, required


def _converted_weights_or_skip(*, environ=None, cache_root=None) -> Path:
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
        pytest.skip(str(exc))
    if converted is None:
        if required:
            pytest.fail("RF-DETR Nano converted weights skipped in required mode")
        pytest.skip("RF-DETR Nano checkpoint not configured")
    return converted


def _local_imports_or_skip():
    required = rfdetr_checkpoint.required_gate_enabled()
    try:
        from mlx_cv.parity import rfdetr_tap_order
        from mlx_cv.parity.rfdetr_real import build_rfdetr_nano_local_model, capture_rfdetr_nano_local
    except Exception as exc:
        if required:
            pytest.fail(f"RF-DETR Nano upstream parity requires the MLX runtime: {exc}")
        pytest.skip(f"RF-DETR Nano upstream parity requires the MLX runtime: {exc}")
    return build_rfdetr_nano_local_model, capture_rfdetr_nano_local, rfdetr_tap_order


def _reference_capture_or_skip(checkpoint):
    required = rfdetr_checkpoint.required_gate_enabled()
    try:
        return rfdetr_upstream.capture_rfdetr_nano_reference(checkpoint)
    except rfdetr_upstream.ReferenceDependencyError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    except Exception as exc:
        if required:
            pytest.fail(f"RF-DETR Nano upstream reference forward failed: {exc}")
        raise


def _local_capture_or_fail(converted: Path):
    build_model, capture_local, _ = _local_imports_or_skip()
    try:
        model = build_model(converted)
        return capture_local(model)
    except Exception as exc:
        pytest.fail(f"RF-DETR Nano local MLX forward failed: {exc}")


def _assert_close(name: str, got, expected, *, atol: float = _ATOL, rtol: float = _RTOL) -> None:
    got_arr = np.asarray(got)
    expected_arr = np.asarray(expected)
    assert got_arr.shape == expected_arr.shape, (
        f"{name} shape mismatch: got {got_arr.shape}, expected {expected_arr.shape}"
    )
    np.testing.assert_allclose(got_arr, expected_arr, atol=atol, rtol=rtol, err_msg=name)


def _assert_final_boxes_close(local, reference) -> None:
    height, width = local.input_image.shape[:2]
    scale = np.asarray([width, height, width, height], dtype=np.float64)
    # Final detections are in pixel units; normalize before tolerance checks so
    # the recorded tolerance remains below the plan's 1e-3 ceiling.
    _assert_close(
        "final boxes normalized",
        np.asarray(local.boxes, dtype=np.float64) / scale,
        np.asarray(reference.boxes, dtype=np.float64) / scale,
        atol=_BOX_ATOL,
        rtol=_BOX_RTOL,
    )


def _assert_local_taps_are_ordered_and_diagnostic(local) -> None:
    _, _, tap_order = _local_imports_or_skip()
    expected_taps = tap_order(num_levels=1, num_layers=2, include_self_attention=True)

    assert list(local.taps) == expected_taps
    assert local.tap_gaps == (
        "backbone.windowed_dinov2: final RFDETRModel capture exposes stable "
        "projector/decoder/head taps; per-block backbone taps are not propagated.",
    )
    assert local.taps["head.logits"].shape == local.raw_logits.shape
    assert local.taps["head.boxes"].shape == local.raw_boxes.shape
    assert local.taps["result.boxes"].shape == local.boxes.shape
    assert local.taps["result.scores"].shape == local.scores.shape
    assert local.taps["result.class_ids"].shape == local.class_ids.shape
    _assert_close("local tap head.logits", local.taps["head.logits"], local.raw_logits)
    _assert_close("local tap head.boxes", local.taps["head.boxes"], local.raw_boxes)
    _assert_close("local tap result.boxes", local.taps["result.boxes"], local.boxes)
    _assert_close("local tap result.scores", local.taps["result.scores"], local.scores)
    np.testing.assert_array_equal(local.taps["result.class_ids"], local.class_ids)
    for name, value in local.taps.items():
        assert np.all(np.isfinite(value)), f"local tap {name} contains non-finite values"


def test_optional_no_checkpoint_upstream_parity_skips_cleanly(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="checkpoint not configured"):
        _checkpoint_or_skip(environ={}, cache_root=tmp_path)


def test_rfdetr_nano_upstream_vs_mlx_real_checkpoint_parity(capsys):
    checkpoint, _ = _checkpoint_or_skip()
    converted = _converted_weights_or_skip()

    reference = _reference_capture_or_skip(checkpoint)
    local = _local_capture_or_fail(converted)

    evidence = checkpoint.evidence()
    assert str(checkpoint.path) in evidence
    assert checkpoint.md5 in evidence
    with capsys.disabled():
        print(evidence)

    np.testing.assert_array_equal(local.input_image, reference.input_image)
    _assert_close("raw logits", local.raw_logits, reference.raw_logits)
    _assert_close("raw boxes", local.raw_boxes, reference.raw_boxes, atol=_BOX_ATOL, rtol=_BOX_RTOL)
    _assert_final_boxes_close(local, reference)
    _assert_close("scores", local.scores, reference.scores)
    np.testing.assert_array_equal(local.class_ids, reference.class_ids)

    assert reference.tap_gaps == (
        "RF-DETR reference exposes final raw logits/boxes through the public model output; "
        "stable intermediate taps are not exposed without invasive hooks.",
    )
    _assert_local_taps_are_ordered_and_diagnostic(local)
