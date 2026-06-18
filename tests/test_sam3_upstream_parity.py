import json
import os
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
STATUS_PATH = Path(".agent/work/2026-06-16-release-parity-hardening/parity-status.json")
REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_SAM3_IMAGE_GATE"


SPEC = importlib.util.spec_from_file_location("sam3_image_upstream", REPO / "tools" / "sam3_image_upstream.py")
assert SPEC is not None and SPEC.loader is not None
sam3_image_upstream = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sam3_image_upstream
SPEC.loader.exec_module(sam3_image_upstream)


def _status():
    return json.loads(STATUS_PATH.read_text())["models"]["sam3_image"]


def _checkpoint_is_usable(path: Path) -> bool:
    return sam3_image_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(path)}
    ).admitted


def _patch_reference_path(tmp_path: Path, monkeypatch) -> Path:
    reference_path = tmp_path / "sam3-reference"
    reference_path.mkdir()
    monkeypatch.setattr(sam3_image_upstream, "SAM3_IMAGE_REFERENCE_PATH", reference_path)
    return reference_path


def _write_local_checkpoint(path: Path) -> None:
    np.savez(path, **{"decoder.query_embed": np.zeros((1,), dtype=np.float32)})


def _write_comparison_checkpoints(tmp_path: Path) -> tuple[Path, Path, Path]:
    admission = tmp_path / "admission.npz"
    local = tmp_path / "local.npz"
    upstream = tmp_path / "upstream.pt"
    _write_local_checkpoint(admission)
    _write_local_checkpoint(local)
    upstream.write_bytes(b"fake-upstream-torch-checkpoint")
    return admission, upstream, local


def _capture(*, drift: float = 0.0):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    prompt = "cat"
    masks = np.array([[[True, False], [False, True]]], dtype=np.bool_)
    boxes = np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float64)
    scores = np.array([0.75], dtype=np.float32)
    class_ids = np.array([0], dtype=np.int64)
    taps = {
        "text.token_ids": np.array([[1, 42, 2, 0]], dtype=np.int64),
        "text.language_features": np.ones((4, 1, 3), dtype=np.float32),
        "text.language_embeds": np.ones((4, 1, 5), dtype=np.float32) * 2.0,
    }
    local_taps = {key: value.copy() for key, value in taps.items()}
    local_taps["text.language_features"] = local_taps["text.language_features"] + np.float32(drift)
    reference = sam3_image_upstream.SAM3ImageCapture(
        source="reference",
        prompt_kind="text",
        image=image,
        prompt=prompt,
        masks=masks,
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        taps=taps,
    )
    local = sam3_image_upstream.SAM3ImageCapture(
        source="local",
        prompt_kind="text",
        image=image,
        prompt=prompt,
        masks=masks.copy(),
        boxes=boxes.copy(),
        scores=scores.copy(),
        class_ids=class_ids.copy(),
        taps=local_taps,
    )
    return reference, local


def test_sam3_image_gate_classifies_checkpoint_admission_failures(tmp_path):
    result = sam3_image_upstream.evaluate_sam3_image_gate(environ={})
    assert result.status == "BLOCKED:MLX_CV_SAM3_IMAGE_CHECKPOINT is unset"

    result = sam3_image_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(tmp_path / "missing.npz")}
    )
    assert "does not point to an existing path" in result.blocked_reason

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    result = sam3_image_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(checkpoint_dir)}
    )
    assert "does not point to a file" in result.blocked_reason

    tiny = tmp_path / "tiny.npz"
    _write_local_checkpoint(tiny)
    result = sam3_image_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(tiny)}
    )
    assert "not a usable SAM3 image checkpoint" in result.blocked_reason

    unsupported = tmp_path / "checkpoint.pt"
    unsupported.write_bytes(b"x" * 8)
    result = sam3_image_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(unsupported)},
        min_checkpoint_bytes=4,
    )
    assert "not loadable by the local image converter" in result.blocked_reason

    video = tmp_path / "video.npz"
    np.savez(video, **{"video_memory_encoder.weight": np.zeros((1,), dtype=np.float32)})
    result = sam3_image_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(video)},
        min_checkpoint_bytes=4,
    )
    assert "video/tracker checkpoint" in result.blocked_reason

    image = tmp_path / "image.npz"
    _write_local_checkpoint(image)
    result = sam3_image_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(image)},
        min_checkpoint_bytes=4,
    )
    assert result.admitted is True


def test_sam3_image_compare_captures_passes_with_documented_tolerances():
    reference, local = _capture()

    report = sam3_image_upstream.compare_sam3_image_captures(reference, local)

    assert report["passed"] is True
    assert report["tolerances"]["masks"] == {"atol": 0.0, "rtol": 0.0}
    assert report["tolerances"]["boxes"] == {"atol": 1e-4, "rtol": 1e-4}
    assert report["tolerances"]["tap.text.token_ids"] == {"atol": 0.0, "rtol": 0.0}
    assert report["selected_tap_pairs"] == [
        {"reference": "text.token_ids", "local": "text.token_ids"},
        {"reference": "text.language_features", "local": "text.language_features"},
        {"reference": "text.language_embeds", "local": "text.language_embeds"},
    ]
    assert report["detection_selection"] == {
        "score_threshold": 0.0,
        "score_threshold_op": ">",
        "order": "score_desc_stable",
    }
    assert [field["name"] for field in report["fields"]] == [
        "masks",
        "boxes",
        "scores",
        "class_ids",
        "tap.text.token_ids",
        "tap.text.language_features",
        "tap.text.language_embeds",
    ]


def test_sam3_image_compare_normalizes_upstream_singleton_mask_channel():
    reference, local = _capture()
    reference = sam3_image_upstream.SAM3ImageCapture(
        source=reference.source,
        prompt_kind=reference.prompt_kind,
        image=reference.image,
        prompt=reference.prompt,
        masks=reference.masks[:, None, :, :],
        boxes=reference.boxes,
        scores=reference.scores,
        class_ids=reference.class_ids,
        taps=reference.taps,
    )

    report = sam3_image_upstream.compare_sam3_image_captures(reference, local)

    assert report["passed"] is True
    assert report["reference_summary"]["masks_shape"] == [1, 1, 2, 2]
    mask_field = next(field for field in report["fields"] if field["name"] == "masks")
    assert mask_field["reference_shape"] == [1, 2, 2]
    assert mask_field["local_shape"] == [1, 2, 2]


def test_sam3_image_compare_canonicalizes_detection_threshold_and_order():
    reference, _ = _capture()
    zero_mask = np.zeros((2, 2), dtype=np.bool_)
    medium_mask = np.array([[True, False], [False, False]], dtype=np.bool_)
    high_mask = np.array([[True, True], [False, False]], dtype=np.bool_)
    reference = sam3_image_upstream.SAM3ImageCapture(
        source="reference",
        prompt_kind="text",
        image=reference.image,
        prompt=reference.prompt,
        masks=np.stack([zero_mask, medium_mask, high_mask], axis=0),
        boxes=np.array(
            [
                [9.0, 9.0, 10.0, 10.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
            ],
            dtype=np.float64,
        ),
        scores=np.array([0.0, 0.2, 0.8], dtype=np.float32),
        class_ids=np.array([9, 2, 1], dtype=np.int64),
        taps=reference.taps,
    )
    local = sam3_image_upstream.SAM3ImageCapture(
        source="local",
        prompt_kind="text",
        image=reference.image,
        prompt=reference.prompt,
        masks=np.stack([high_mask, medium_mask], axis=0),
        boxes=np.array([[3.0, 3.0, 4.0, 4.0], [1.0, 1.0, 2.0, 2.0]], dtype=np.float64),
        scores=np.array([0.8, 0.2], dtype=np.float32),
        class_ids=np.array([1, 2], dtype=np.int64),
        taps=reference.taps,
    )

    report = sam3_image_upstream.compare_sam3_image_captures(reference, local)

    assert report["passed"] is True
    assert report["detection_selection"]["score_threshold_op"] == ">"
    for name in ("masks", "boxes", "scores", "class_ids"):
        field = next(field for field in report["fields"] if field["name"] == name)
        assert field["passed"] is True


def test_sam3_image_comparison_gate_passes_with_injected_captures(tmp_path, monkeypatch):
    admission, upstream_checkpoint, local_checkpoint = _write_comparison_checkpoints(tmp_path)
    _patch_reference_path(tmp_path, monkeypatch)
    reference, local = _capture()
    seen = {}

    def reference_capture(path, *, reference_path):
        seen["reference_path"] = Path(path)
        return reference

    def local_capture(path, *, inputs=None):
        seen["local_path"] = Path(path)
        return local

    result = sam3_image_upstream.evaluate_sam3_image_comparison_gate(
        environ={
            "MLX_CV_SAM3_IMAGE_CHECKPOINT": str(admission),
            "MLX_CV_SAM3_IMAGE_UPSTREAM_CHECKPOINT": str(upstream_checkpoint),
            "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT": str(local_checkpoint),
        },
        min_checkpoint_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=reference_capture,
        local_capture_func=local_capture,
    )

    assert result.status == "UPSTREAM_PASSED"
    assert result.comparison_report["passed"] is True
    assert seen == {"reference_path": upstream_checkpoint, "local_path": local_checkpoint}
    assert result.upstream_checkpoint_path == str(upstream_checkpoint)
    assert result.local_checkpoint_path == str(local_checkpoint)
    assert sam3_image_upstream.status_dict(result)["claim_level"] == "upstream_passed"


def test_sam3_image_comparison_gate_blocks_on_numeric_drift(tmp_path, monkeypatch):
    admission, upstream_checkpoint, local_checkpoint = _write_comparison_checkpoints(tmp_path)
    _patch_reference_path(tmp_path, monkeypatch)
    reference, local = _capture(drift=1e-2)

    result = sam3_image_upstream.evaluate_sam3_image_comparison_gate(
        environ={
            "MLX_CV_SAM3_IMAGE_CHECKPOINT": str(admission),
            "MLX_CV_SAM3_IMAGE_UPSTREAM_CHECKPOINT": str(upstream_checkpoint),
            "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT": str(local_checkpoint),
        },
        min_checkpoint_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda _path, *, reference_path: reference,
        local_capture_func=lambda _path, *, inputs=None: local,
    )

    assert result.status.startswith("BLOCKED:")
    assert "parity drift" in result.blocked_reason
    assert "tap.text.language_features" in result.blocked_reason


def test_sam3_image_comparison_gate_blocks_when_upstream_checkpoint_is_missing(tmp_path, monkeypatch):
    admission, _, local_checkpoint = _write_comparison_checkpoints(tmp_path)
    _patch_reference_path(tmp_path, monkeypatch)

    result = sam3_image_upstream.evaluate_sam3_image_comparison_gate(
        environ={
            "MLX_CV_SAM3_IMAGE_CHECKPOINT": str(admission),
            "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT": str(local_checkpoint),
        },
        min_checkpoint_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda *args, **kwargs: pytest.fail("reference capture should not run"),
        local_capture_func=lambda *args, **kwargs: pytest.fail("local capture should not run"),
    )

    assert result.status.startswith("BLOCKED:")
    assert "MLX_CV_SAM3_IMAGE_UPSTREAM_CHECKPOINT is unset" in result.blocked_reason


def test_sam3_image_comparison_gate_blocks_when_local_checkpoint_is_missing(tmp_path, monkeypatch):
    admission, upstream_checkpoint, _ = _write_comparison_checkpoints(tmp_path)
    _patch_reference_path(tmp_path, monkeypatch)

    result = sam3_image_upstream.evaluate_sam3_image_comparison_gate(
        environ={
            "MLX_CV_SAM3_IMAGE_CHECKPOINT": str(admission),
            "MLX_CV_SAM3_IMAGE_UPSTREAM_CHECKPOINT": str(upstream_checkpoint),
        },
        min_checkpoint_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda *args, **kwargs: pytest.fail("reference capture should not run"),
        local_capture_func=lambda *args, **kwargs: pytest.fail("local capture should not run"),
    )

    assert result.status.startswith("BLOCKED:")
    assert "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT is unset" in result.blocked_reason


def test_sam3_image_comparison_gate_blocks_on_missing_upstream_or_local_path(tmp_path, monkeypatch):
    admission, upstream_checkpoint, local_checkpoint = _write_comparison_checkpoints(tmp_path)
    _patch_reference_path(tmp_path, monkeypatch)

    result = sam3_image_upstream.evaluate_sam3_image_comparison_gate(
        environ={
            "MLX_CV_SAM3_IMAGE_CHECKPOINT": str(admission),
            "MLX_CV_SAM3_IMAGE_UPSTREAM_CHECKPOINT": str(tmp_path / "missing.pt"),
            "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT": str(local_checkpoint),
        },
        min_checkpoint_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda *args, **kwargs: pytest.fail("reference capture should not run"),
        local_capture_func=lambda *args, **kwargs: pytest.fail("local capture should not run"),
    )
    assert result.status.startswith("BLOCKED:")
    assert "MLX_CV_SAM3_IMAGE_UPSTREAM_CHECKPOINT does not point to an existing path" in result.blocked_reason

    result = sam3_image_upstream.evaluate_sam3_image_comparison_gate(
        environ={
            "MLX_CV_SAM3_IMAGE_CHECKPOINT": str(admission),
            "MLX_CV_SAM3_IMAGE_UPSTREAM_CHECKPOINT": str(upstream_checkpoint),
            "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT": str(tmp_path / "missing.npz"),
        },
        min_checkpoint_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda *args, **kwargs: pytest.fail("reference capture should not run"),
        local_capture_func=lambda *args, **kwargs: pytest.fail("local capture should not run"),
    )
    assert result.status.startswith("BLOCKED:")
    assert "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT does not point to an existing path" in result.blocked_reason


def test_sam3_image_upstream_parity_gate_records_missing_checkpoint_or_taps_blocker():
    model_status = _status()
    required = os.environ.get(REQUIRED_GATE_ENV) == "1"
    reference_path = Path(model_status["reference_path"])
    if not reference_path.exists():
        assert model_status["status"].startswith("BLOCKED:")
        assert model_status["blocked_reason"]
        if required:
            return
        pytest.skip(f"{reference_path} is missing; blocker recorded in parity-status.json")

    checkpoint = os.environ.get(model_status["checkpoint_env"])
    if not checkpoint:
        assert model_status["status"].startswith("BLOCKED:")
        assert model_status["blocked_reason"]
        assert model_status["checkpoint_env"] in model_status["status"]
        if required:
            return
        pytest.skip(f"{model_status['checkpoint_env']} is unset; blocker recorded in parity-status.json")

    checkpoint_path = Path(checkpoint)
    if not _checkpoint_is_usable(checkpoint_path):
        assert model_status["status"].startswith("BLOCKED:")
        assert model_status["blocked_reason"]
        if required:
            return
        pytest.skip(f"{checkpoint_path} is not a usable SAM3 image checkpoint")

    comparison_env = {model_status["checkpoint_env"]: str(checkpoint_path)}
    for key in (
        sam3_image_upstream.SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV,
        sam3_image_upstream.SAM3_IMAGE_LOCAL_CHECKPOINT_ENV,
    ):
        if os.environ.get(key):
            comparison_env[key] = os.environ[key]
    result = sam3_image_upstream.evaluate_sam3_image_comparison_gate(environ=comparison_env)
    assert result.blocked is True
    assert any(
        text in result.blocked_reason
        for text in (
            "MLX_CV_SAM3_IMAGE_UPSTREAM_CHECKPOINT",
            "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT",
            "reference capture",
            "requires torch",
            "reference modules could not be imported",
            "local MLX capture",
            "parity drift",
        )
    )
