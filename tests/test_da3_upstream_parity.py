from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
da3_upstream = _load_tool("da3_upstream", "tools/da3_upstream.py")
da3_demo = _load_tool("da3_demo", "tools/da3_demo.py")


def _checkpoint_or_skip(*, environ=None, cache_root=None):
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
            pytest.fail("DA3 upstream parity skipped in required mode")
        pytest.skip("DA3 checkpoint not configured")
    return checkpoint, required


def _converted_weights_or_skip(*, environ=None, cache_root=None) -> Path:
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
        pytest.skip(str(exc))
    if converted is None:
        if required:
            pytest.fail("DA3 local converted weights skipped in required mode")
        pytest.skip("DA3 checkpoint not configured")
    return converted


def _reference_capture_or_skip(checkpoint, *, required: bool, images: np.ndarray | None = None):
    try:
        return da3_upstream.capture_da3_upstream_reference(checkpoint, images=images)
    except (da3_upstream.DA3ReferenceDependencyError, da3_upstream.DA3UpstreamCaptureError) as exc:
        if required:
            pytest.fail(f"DA3 upstream capture failed in required mode: {exc}")
        pytest.skip(str(exc))
    except Exception as exc:
        if required:
            pytest.fail(f"DA3 upstream reference forward failed in required mode: {exc}")
        raise


def _local_capture_or_skip(converted: Path, reference, *, required: bool):
    try:
        return da3_demo.load_da3_local_capture(converted, images=reference.input_images)
    except Exception as exc:
        if required:
            pytest.fail(f"DA3 local load/forward failed in required mode: {exc}")
        pytest.skip(str(exc))


def _fake_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / da3_checkpoint.DA3_CHECKPOINT_FILENAME
    config_path = tmp_path / da3_checkpoint.DA3_CONFIG_FILENAME
    checkpoint_path.write_bytes(b"fake-da3-weights")
    config_path.write_bytes(b'{"model_name":"da3-small"}')
    return da3_checkpoint.DA3CheckpointInfo(
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


def _fake_capture_pair(*, drift: float = 0.0):
    depth = np.ones((3, 2, 2), dtype=np.float32)
    confidence = np.full((3, 2, 2), 0.5, dtype=np.float32)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
    taps = {
        "feat_layer_5": np.ones((3, 4, 8), dtype=np.float32),
        "feat_layer_7": np.ones((3, 4, 8), dtype=np.float32) * 2.0,
        "feat_layer_9": np.ones((3, 4, 8), dtype=np.float32) * 3.0,
        "feat_layer_11": np.ones((3, 4, 8), dtype=np.float32) * 4.0,
    }
    reference = SimpleNamespace(
        depth=depth,
        confidence=confidence,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        taps=taps,
        input_images=np.zeros((3, 112, 112, 3), dtype=np.uint8),
        selected_reference_index=1,
        view_order=(0, 1, 2),
        summary=lambda: {"side": "reference"},
    )
    local = SimpleNamespace(
        raw_depth=depth[None] + drift,
        raw_confidence=confidence[None],
        extrinsics=extrinsics[None],
        intrinsics=intrinsics[None],
        taps={
            "aux_feat_layer_05": taps["feat_layer_5"],
            "aux_feat_layer_07": taps["feat_layer_7"],
            "aux_feat_layer_09": taps["feat_layer_9"],
            "aux_feat_layer_11": taps["feat_layer_11"],
        },
        summary=lambda: {"side": "local"},
    )
    return reference, local


def _assert_demo_artifacts(artifacts: dict[str, str], *, view_count: int) -> None:
    assert Path(artifacts["camera_summary"]).is_file()
    assert Path(artifacts["parity_summary"]).is_file()
    for index in range(view_count):
        assert Path(artifacts[f"view_{index:02d}_input"]).is_file()
        assert Path(artifacts[f"view_{index:02d}_upstream_depth"]).is_file()
        assert Path(artifacts[f"view_{index:02d}_local_depth"]).is_file()
        assert Path(artifacts[f"view_{index:02d}_absdiff_depth"]).is_file()
    assert Path(artifacts["contact_sheet"]).is_file()
    assert Path(artifacts["readme"]).is_file()


def test_da3_parity_comparison_has_explicit_tolerances_and_selected_taps():
    reference, local = _fake_capture_pair()

    report = da3_demo.compare_da3_captures(reference, local)

    assert report["passed"] is True
    assert report["tolerances"]["depth"] == {"atol": 0.05, "rtol": 0.0}
    assert report["tolerances"]["confidence"] == {"atol": 0.075, "rtol": 0.0}
    assert report["selected_tap_pairs"] == [
        {"reference": "feat_layer_5", "local": "aux_feat_layer_05"},
        {"reference": "feat_layer_7", "local": "aux_feat_layer_07"},
        {"reference": "feat_layer_9", "local": "aux_feat_layer_09"},
        {"reference": "feat_layer_11", "local": "aux_feat_layer_11"},
    ]
    assert [field["name"] for field in report["fields"]] == [
        "depth",
        "confidence",
        "extrinsics",
        "intrinsics",
        "tap.feat_layer_5",
        "tap.feat_layer_7",
        "tap.feat_layer_9",
        "tap.feat_layer_11",
    ]


def test_da3_upstream_capture_accepts_two_square_rgb_views():
    images = np.zeros((2, 112, 112, 3), dtype=np.uint8)

    validated = da3_upstream._validate_fixed_views(images)

    assert validated.shape == (2, 112, 112, 3)


def test_da3_demo_loads_real_images_as_square_views():
    paths = [
        REPO / "references/Depth-Anything-3/assets/examples/SOH/000.png",
        REPO / "references/Depth-Anything-3/assets/examples/SOH/010.png",
    ]
    if not all(p.exists() for p in paths):
        pytest.skip("DA3 reference assets absent (references/ is gitignored, not in CI)")

    images = da3_demo.load_da3_demo_images(paths, image_size=112)

    assert images.shape == (2, 112, 112, 3)
    assert images.dtype == np.uint8
    assert not np.array_equal(images[0], images[1])


def test_da3_demo_loads_robot_video_as_square_views():
    pytest.importorskip("cv2")
    images = da3_demo.load_da3_demo_video_frames(
        REPO / "references/Depth-Anything-3/assets/examples/robot_unitree.mp4",
        image_size=112,
    )

    assert images.shape == (3, 112, 112, 3)
    assert images.dtype == np.uint8
    assert not np.array_equal(images[0], images[-1])


def test_da3_required_drift_beyond_tolerance_fails():
    reference, local = _fake_capture_pair(drift=0.1)
    report = da3_demo.compare_da3_captures(reference, local)

    assert report["passed"] is False
    with pytest.raises(da3_demo.DA3ParityError, match="depth"):
        da3_demo.raise_for_failed_parity(report)


def test_da3_missing_selected_tap_fails_required_comparison():
    reference, local = _fake_capture_pair()
    del reference.taps["feat_layer_7"]

    with pytest.raises(da3_demo.DA3ParityError, match="feat_layer_7"):
        da3_demo.compare_da3_captures(reference, local)


def test_required_no_checkpoint_upstream_parity_fails_instead_of_skipping(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="checkpoint is required but missing"):
        _checkpoint_or_skip(
            environ={da3_checkpoint.DA3_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
        )


def test_required_missing_upstream_capture_fails_instead_of_skipping(tmp_path, monkeypatch):
    checkpoint = _fake_checkpoint(tmp_path)

    def missing_capture(_checkpoint, **_kwargs):
        raise da3_upstream.DA3UpstreamCaptureError("missing upstream capture")

    monkeypatch.setattr(da3_upstream, "capture_da3_upstream_reference", missing_capture)

    with pytest.raises(pytest.fail.Exception, match="missing upstream capture"):
        _reference_capture_or_skip(checkpoint, required=True)


def test_required_missing_local_load_fails_instead_of_skipping(tmp_path):
    missing = tmp_path / "missing-da3-small.npz"

    with pytest.raises(pytest.fail.Exception, match="not a file"):
        _converted_weights_or_skip(
            environ={
                da3_checkpoint.DA3_REQUIRED_GATE_ENV: "1",
                da3_convert_tool.DA3_CONVERTED_ENV: str(missing),
            },
            cache_root=tmp_path,
        )


def test_da3_upstream_vs_mlx_real_checkpoint_parity_writes_demo_artifacts(capsys):
    checkpoint, required = _checkpoint_or_skip(environ=dict(os.environ), cache_root=None)
    converted = _converted_weights_or_skip(environ=dict(os.environ), cache_root=None)

    reference = _reference_capture_or_skip(checkpoint, required=required)
    local = _local_capture_or_skip(converted, reference, required=required)
    report = da3_demo.compare_da3_captures(reference, local)
    artifacts = da3_demo.write_da3_demo_artifacts(
        reference,
        local,
        report,
        output_dir=da3_demo.DEFAULT_OUTPUT_DIR,
    )

    da3_checkpoint.print_checkpoint_evidence(checkpoint)
    print(json.dumps({"passed": report["passed"], "fields": report["fields"]}, indent=2, sort_keys=True))
    output = capsys.readouterr().out
    assert str(checkpoint.checkpoint_path) in output
    assert checkpoint.checkpoint_sha256 in output
    assert "weights_sha256" in output

    da3_demo.raise_for_failed_parity(report)
    _assert_demo_artifacts(artifacts, view_count=3)

    parity = json.loads(Path(artifacts["parity_summary"]).read_text())
    cameras = json.loads(Path(artifacts["camera_summary"]).read_text())
    assert parity["passed"] is True
    assert cameras["reference"]["selected_reference_index"] == 1


def test_da3_upstream_vs_mlx_real_image_and_video_parity_writes_demo_artifacts():
    checkpoint, required = _checkpoint_or_skip(environ=dict(os.environ), cache_root=None)
    converted = _converted_weights_or_skip(environ=dict(os.environ), cache_root=None)
    cases = [
        (
            "soh",
            da3_demo.load_da3_demo_images(
                [
                    REPO / "references/Depth-Anything-3/assets/examples/SOH/000.png",
                    REPO / "references/Depth-Anything-3/assets/examples/SOH/010.png",
                ],
                image_size=112,
            ),
            Path("/tmp/mlx-cv-da3-real-demo"),
        ),
        (
            "robot",
            da3_demo.load_da3_demo_video_frames(
                REPO / "references/Depth-Anything-3/assets/examples/robot_unitree.mp4",
                image_size=112,
            ),
            Path("/tmp/mlx-cv-da3-real-video-demo"),
        ),
    ]

    for case_name, images, output_dir in cases:
        reference = _reference_capture_or_skip(checkpoint, required=required, images=images)
        local = _local_capture_or_skip(converted, reference, required=required)
        report = da3_demo.compare_da3_captures(reference, local)
        artifacts = da3_demo.write_da3_demo_artifacts(
            reference,
            local,
            report,
            output_dir=output_dir,
        )

        da3_demo.raise_for_failed_parity(report)
        _assert_demo_artifacts(artifacts, view_count=int(images.shape[0]))
        parity = json.loads(Path(artifacts["parity_summary"]).read_text())
        cameras = json.loads(Path(artifacts["camera_summary"]).read_text())
        assert parity["passed"] is True
        assert parity["reference_summary"]["input_shape"][0] == int(images.shape[0])
        assert cameras["local"]["view_order"] == list(range(int(images.shape[0])))
