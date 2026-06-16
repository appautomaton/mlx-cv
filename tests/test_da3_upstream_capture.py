from __future__ import annotations

import importlib.util
import os
import re
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
da3_upstream = _load_tool("da3_upstream", "tools/da3_upstream.py")


def _fake_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / da3_checkpoint.DA3_CHECKPOINT_FILENAME
    config_path = tmp_path / da3_checkpoint.DA3_CONFIG_FILENAME
    checkpoint_path.write_bytes(b"fake-da3-weights")
    config_path.write_bytes(b'{"model_name":"da3-small"}')
    return da3_upstream.DA3CheckpointInfo(
        model_id=da3_checkpoint.DA3_DEFAULT_MODEL_ID,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        checkpoint_sha256="weights-sha",
        config_sha256="config-sha",
        checkpoint_url="https://example.test/model.safetensors",
        config_url="https://example.test/config.json",
        revision="main",
        license_note="Apache-2.0",
        source="test",
    )


def _checkpoint_or_skip(*, environ: dict[str, str], cache_root: Path):
    required = da3_checkpoint.required_gate_enabled(environ)
    try:
        checkpoint = da3_checkpoint.resolve_da3_checkpoint(
            environ=environ,
            cache_root=cache_root,
            required=required,
        )
    except da3_checkpoint.DA3CheckpointError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    if checkpoint is None:
        if required:
            pytest.fail("DA3 checkpoint not configured")
        pytest.skip("DA3 checkpoint not configured")
    return checkpoint, required


def _capture_or_skip(checkpoint, *, required: bool):
    try:
        return da3_upstream.capture_da3_upstream_reference(checkpoint)
    except (da3_upstream.DA3ReferenceDependencyError, da3_upstream.DA3UpstreamCaptureError) as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def test_da3_fixed_multiview_input_is_three_same_size_ordered_views():
    images = da3_upstream.da3_multiview_fixed_images()

    assert images.shape == (3, 112, 112, 3)
    assert images.dtype == np.uint8
    assert images.flags.c_contiguous
    assert not np.array_equal(images[0], images[1])
    assert not np.array_equal(images[1], images[2])


def test_optional_no_checkpoint_skips_cleanly(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="DA3 checkpoint not configured"):
        _checkpoint_or_skip(environ={}, cache_root=tmp_path)


def test_required_missing_checkpoint_fails_instead_of_skipping(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="checkpoint is required but missing"):
        _checkpoint_or_skip(
            environ={da3_checkpoint.DA3_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
        )


def test_required_missing_reference_dependency_fails_instead_of_skipping(tmp_path, monkeypatch):
    checkpoint = _fake_checkpoint(tmp_path)

    def missing_reference():
        raise da3_upstream.DA3ReferenceDependencyError("missing upstream DA3 reference")

    monkeypatch.setattr(da3_upstream, "_import_reference", missing_reference)

    with pytest.raises(pytest.fail.Exception, match="missing upstream DA3 reference"):
        _capture_or_skip(checkpoint, required=True)


def test_mocked_upstream_capture_records_schema_and_taps(tmp_path, monkeypatch):
    checkpoint = _fake_checkpoint(tmp_path)
    fixed_images = da3_upstream.da3_multiview_fixed_images()

    class FakePrediction:
        processed_images = fixed_images.copy()
        depth = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
        conf = np.ones((3, 8, 8), dtype=np.float32)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
        aux = {
            "feat_layer_5": np.zeros((3, 8, 8, 4), dtype=np.float32),
            "feat_layer_7": np.ones((3, 8, 8, 4), dtype=np.float32),
        }

    def fake_import_reference():
        return object(), object()

    def fake_load_reference_model(_DepthAnything3, got_checkpoint):
        assert got_checkpoint == checkpoint
        return object()

    def fake_model_to_cpu_float32(model):
        return model

    def fake_run_upstream_float32(**kwargs):
        np.testing.assert_array_equal(kwargs["images"], fixed_images)
        assert kwargs["process_res"] == da3_upstream.DEFAULT_PROCESS_RES
        assert kwargs["process_res_method"] == da3_upstream.DEFAULT_PROCESS_RES_METHOD
        assert tuple(kwargs["export_feat_layers"]) == da3_upstream.DEFAULT_EXPORT_FEAT_LAYERS
        assert kwargs["ref_view_strategy"] == "middle"
        return FakePrediction(), {
            "device": "cpu",
            "dtype": "float32",
            "depthanything3_forward_autocast": "bypassed",
            "reference_selector_calls": [[1]],
            "torch_autocast_enabled": False,
        }

    monkeypatch.setattr(da3_upstream, "_import_reference", fake_import_reference)
    monkeypatch.setattr(da3_upstream, "_load_reference_model", fake_load_reference_model)
    monkeypatch.setattr(da3_upstream, "_model_to_cpu_float32", fake_model_to_cpu_float32)
    monkeypatch.setattr(da3_upstream, "_run_upstream_float32", fake_run_upstream_float32)

    capture = da3_upstream.capture_da3_upstream_reference(checkpoint)

    assert capture.input_images.shape == (3, 112, 112, 3)
    assert capture.processed_images.shape == (3, 112, 112, 3)
    assert capture.depth.shape == (3, 8, 8)
    assert capture.confidence.shape == (3, 8, 8)
    assert capture.extrinsics.shape == (3, 4, 4)
    assert capture.intrinsics.shape == (3, 3, 3)
    assert capture.selected_reference_index == 1
    assert capture.view_order == (0, 1, 2)
    assert set(capture.taps) == {"feat_layer_5", "feat_layer_7"}
    assert capture.autocast_policy["torch_autocast_enabled"] is False

    summary = capture.summary()
    assert summary["processed_image_shape"] == [3, 112, 112, 3]
    assert summary["depth_shape"] == [3, 8, 8]
    assert summary["confidence_shape"] == [3, 8, 8]
    assert summary["extrinsics_shape"] == [3, 4, 4]
    assert summary["intrinsics_shape"] == [3, 3, 3]
    assert summary["tap_shapes"]["feat_layer_5"] == [3, 8, 8, 4]

    arrays = capture.as_arrays()
    assert arrays["selected_reference_index"].shape == ()
    assert arrays["view_order"].tolist() == [0, 1, 2]
    assert arrays["tap.feat_layer_7"].shape == (3, 8, 8, 4)


def test_default_three_view_capture_requires_recorded_reference_selection(tmp_path, monkeypatch):
    checkpoint = _fake_checkpoint(tmp_path)
    fixed_images = da3_upstream.da3_multiview_fixed_images()

    class FakePrediction:
        processed_images = fixed_images.copy()
        depth = np.ones((3, 8, 8), dtype=np.float32)
        conf = np.ones((3, 8, 8), dtype=np.float32)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
        aux = {"feat_layer_5": np.zeros((3, 8, 8, 4), dtype=np.float32)}

    monkeypatch.setattr(da3_upstream, "_import_reference", lambda: (object(), object()))
    monkeypatch.setattr(da3_upstream, "_load_reference_model", lambda _DepthAnything3, _checkpoint: object())
    monkeypatch.setattr(da3_upstream, "_model_to_cpu_float32", lambda model: model)
    monkeypatch.setattr(
        da3_upstream,
        "_run_upstream_float32",
        lambda **_kwargs: (
            FakePrediction(),
            {
                "device": "cpu",
                "dtype": "float32",
                "depthanything3_forward_autocast": "bypassed",
                "torch_autocast_enabled": False,
            },
        ),
    )

    with pytest.raises(da3_upstream.DA3UpstreamCaptureError, match="reference-view selection"):
        da3_upstream.capture_da3_upstream_reference(checkpoint)


def test_runtime_package_sources_do_not_hard_import_da3_reference_dependencies():
    import_re = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w.]*)", re.MULTILINE)
    blocked = {
        "torch",
        "torchvision",
        "cv2",
        "huggingface_hub",
        "references",
        "depth_anything_3",
        "urllib",
    }
    for path in (REPO / "src" / "mlx_cv").rglob("*.py"):
        text = path.read_text()
        imports = {match.group(1).split(".", 1)[0] for match in import_re.finditer(text)}
        assert not (imports & blocked), path


def test_da3_upstream_capture_runs_real_checkpoint(capsys):
    checkpoint, required = _checkpoint_or_skip(environ=dict(os.environ), cache_root=None)
    capture = _capture_or_skip(checkpoint, required=required)

    da3_checkpoint.print_checkpoint_evidence(checkpoint)
    out = capsys.readouterr().out
    assert str(checkpoint.checkpoint_path) in out
    assert checkpoint.checkpoint_sha256 in out

    assert capture.input_images.shape == (3, 112, 112, 3)
    assert capture.processed_images.shape[0] == 3
    assert capture.depth.ndim == 3 and capture.depth.shape[0] == 3
    assert capture.confidence.shape == capture.depth.shape
    assert capture.extrinsics.shape[0] == 3
    assert capture.extrinsics.shape[1:] in ((3, 4), (4, 4))
    assert capture.intrinsics.shape == (3, 3, 3)
    assert capture.selected_reference_index == 1
    assert capture.autocast_policy["reference_selector_calls"] == [[1]]
    assert capture.view_order == (0, 1, 2)
    assert capture.taps
    assert capture.autocast_policy["dtype"] == "float32"
