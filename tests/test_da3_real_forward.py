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


da3_checkpoint = _load_tool("da3_checkpoint", "tools/da3_checkpoint.py")
da3_convert_tool = _load_tool("da3_convert_checkpoint", "tools/da3_convert_checkpoint.py")


def _converted_weights_for_real_forward(*, environ=None, cache_root=None) -> Path:
    required = da3_checkpoint.required_gate_enabled(environ)
    try:
        converted = da3_convert_tool.resolve_da3_converted_weights(
            environ=environ,
            cache_root=cache_root,
            required=required,
        )
    except da3_convert_tool.DA3ConversionDependencyError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    except (da3_checkpoint.DA3CheckpointError, da3_convert_tool.DA3ConversionError) as exc:
        if required:
            pytest.fail(str(exc))
        raise
    if converted is None:
        if required:
            pytest.fail("DA3 real forward skipped in required mode")
        pytest.skip("DA3 checkpoint not configured")
    return converted


def _runtime_imports_for_real_forward():
    required = da3_checkpoint.required_gate_enabled()
    try:
        from mlx_cv.core.types import Result
        from mlx_cv.parity.fixtures import da3_multiview_fixed_images
        from mlx_cv.parity.da3_real import build_da3_small_local_model, capture_da3_small_local
    except Exception as exc:
        if required:
            pytest.fail(f"DA3 real forward requires the MLX runtime: {exc}")
        pytest.skip(f"DA3 real forward requires the MLX runtime: {exc}")
    return Result, build_da3_small_local_model, capture_da3_small_local, da3_multiview_fixed_images


def test_optional_no_checkpoint_real_forward_skips_cleanly(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="DA3 checkpoint not configured"):
        _converted_weights_for_real_forward(environ={}, cache_root=tmp_path)


def test_required_no_checkpoint_real_forward_fails_instead_of_skipping(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="checkpoint is required but missing"):
        _converted_weights_for_real_forward(
            environ={da3_checkpoint.DA3_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
        )


def test_required_missing_converted_real_forward_fails_instead_of_skipping(tmp_path):
    missing = tmp_path / "missing-da3-small.npz"

    with pytest.raises(pytest.fail.Exception, match="not a file"):
        _converted_weights_for_real_forward(
            environ={
                da3_checkpoint.DA3_REQUIRED_GATE_ENV: "1",
                da3_convert_tool.DA3_CONVERTED_ENV: str(missing),
            },
            cache_root=tmp_path,
        )


def test_real_da3_small_local_forward_runs_fixed_multiview_input():
    converted = _converted_weights_for_real_forward()
    Result, build_model, capture_local, fixed_images_fn = _runtime_imports_for_real_forward()
    model = build_model(converted)
    capture = capture_local(model)
    fixed_images = fixed_images_fn()

    assert isinstance(capture.result, Result)
    np.testing.assert_array_equal(capture.input_images, fixed_images)
    assert capture.input_images.shape == (3, 112, 112, 3)
    assert capture.input_tensor.shape == (1, 3, 3, 112, 112)
    assert capture.input_tensor.dtype == np.float32

    assert capture.raw_depth.shape == (1, 3, 112, 112)
    assert capture.raw_confidence.shape == (1, 3, 112, 112)
    assert capture.raw_ray.shape == (1, 3, 64, 64, 6)
    assert capture.raw_ray_confidence.shape == (1, 3, 64, 64)
    assert capture.pose_encoding.shape == (1, 3, 9)
    assert capture.extrinsics.shape == (1, 3, 3, 4)
    assert capture.intrinsics.shape == (1, 3, 3, 3)

    for arr in (
        capture.raw_depth,
        capture.raw_confidence,
        capture.raw_ray,
        capture.pose_encoding,
        capture.extrinsics,
        capture.intrinsics,
    ):
        assert np.all(np.isfinite(arr))
    # The upstream camera-decoder path deletes raw ray confidence before returning
    # public outputs; keep the local debug capture shape-valid without requiring
    # internal confidence overflow behavior to become a public contract.
    assert not np.any(np.isnan(capture.raw_ray_confidence))

    assert capture.result.image_size == (112, 112)
    assert capture.result.depth is capture.result.depth_views[0]
    assert len(capture.result.depth_views) == 3
    assert capture.result.depth_views[0].depth.shape == (112, 112)
    assert capture.result.depth_views[0].depth_conf.shape == (112, 112)
    assert capture.result.camera_geometry is not None
    assert capture.result.camera_geometry.convention == "w2c"
    assert capture.result.camera_geometry.extrinsics.shape == (3, 3, 4)
    assert capture.result.camera_geometry.intrinsics.shape == (3, 3, 3)

    summary = capture.summary()
    assert summary["depth_shape"] == [1, 3, 112, 112]
    assert summary["result_depth_views"] == 3
