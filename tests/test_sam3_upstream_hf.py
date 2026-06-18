"""Slice 1: transformers-based SAM3 reference-capture harness (image + video).

Mirrors tests/test_la_upstream_parity.py: admission classification, the documented
tap comparison, an injected-capture PASS, numeric-drift BLOCK, the honest
not-yet-ported local blocker, and the parity-status honest-blocker contract.
No torch/transformers/weights required — captures are injected.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
STATUS_PATH = Path(".agent/work/2026-06-16-release-parity-hardening/parity-status.json")

SPEC = importlib.util.spec_from_file_location("sam3_upstream", REPO / "tools" / "sam3_upstream.py")
assert SPEC is not None and SPEC.loader is not None
sam3_upstream = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sam3_upstream
SPEC.loader.exec_module(sam3_upstream)


def _status(model: str) -> dict:
    return json.loads(STATUS_PATH.read_text())["models"][model]


def _write_config(path: Path) -> None:
    (path / "config.json").write_text(json.dumps({"model_type": "sam3"}))


def _full_checkpoint_dir(tmp_path: Path, *, name: str = "full", weight: str = "model.safetensors") -> Path:
    full_dir = tmp_path / name
    full_dir.mkdir()
    _write_config(full_dir)
    (full_dir / weight).write_bytes(b"x" * 8)
    return full_dir


def _capture(*, drift: float = 0.0):
    pixel_values = np.zeros((1, 3, 4, 4), dtype=np.float32)
    tap = np.ones((1, 4, 8), dtype=np.float32)
    reference = sam3_upstream.Sam3Capture(
        source="upstream_reference",
        pixel_values=pixel_values,
        taps={"vision.last_hidden_state": tap},
    )
    local = sam3_upstream.Sam3Capture(
        source="mlx_local",
        pixel_values=pixel_values.copy(),
        taps={"vision.last_hidden_state": tap.copy() + np.float32(drift)},
    )
    return reference, local


# --- admission ----------------------------------------------------------------


def test_sam3_image_gate_classifies_checkpoint_admission_failures(tmp_path):
    result = sam3_upstream.evaluate_sam3_image_gate(environ={})
    assert result.status == "BLOCKED:MLX_CV_SAM3_IMAGE_CHECKPOINT is unset"

    result = sam3_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(tmp_path / "missing")}
    )
    assert "does not point to an existing path" in result.blocked_reason

    no_config = tmp_path / "no_config"
    no_config.mkdir()
    result = sam3_upstream.evaluate_sam3_image_gate(environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(no_config)})
    assert "missing config.json" in result.blocked_reason

    no_weights = tmp_path / "no_weights"
    no_weights.mkdir()
    _write_config(no_weights)
    result = sam3_upstream.evaluate_sam3_image_gate(environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(no_weights)})
    assert "missing model.safetensors" in result.blocked_reason

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _write_config(stub_dir)
    (stub_dir / "model.safetensors").write_bytes(b"stub")
    result = sam3_upstream.evaluate_sam3_image_gate(environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(stub_dir)})
    assert "LFS stub or incomplete" in result.blocked_reason

    # Image admission does not accept torch pickles.
    pickle_file = tmp_path / "weights.pt"
    pickle_file.write_bytes(b"x" * 8)
    result = sam3_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(pickle_file)}, min_shard_bytes=4
    )
    assert "unsupported SAM3 checkpoint format" in result.blocked_reason

    full_dir = _full_checkpoint_dir(tmp_path)
    result = sam3_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(full_dir)}, min_shard_bytes=4
    )
    assert result.admitted is True
    assert result.status == "ADMITTED"
    assert result.comparison_report is None
    assert sam3_upstream.status_dict(result)["claim_level"] == "checkpoint_admitted"


def test_sam3_video_gate_admits_torch_pickle_checkpoint(tmp_path):
    # The multiplex video checkpoint ships as a .pt; video admission accepts it.
    pt_dir = _full_checkpoint_dir(tmp_path, name="video", weight="sam3.1_multiplex.pt")
    result = sam3_upstream.evaluate_sam3_video_gate(
        environ={"MLX_CV_SAM3_VIDEO_CHECKPOINT": str(pt_dir)}, min_shard_bytes=4
    )
    assert result.admitted is True
    assert sam3_upstream.status_dict(result)["claim_level"] == "checkpoint_admitted"

    bare_pt = tmp_path / "sam3.1_multiplex.pt"
    bare_pt.write_bytes(b"x" * 8)
    result = sam3_upstream.evaluate_sam3_video_gate(
        environ={"MLX_CV_SAM3_VIDEO_CHECKPOINT": str(bare_pt)}, min_shard_bytes=4
    )
    assert result.admitted is True


# --- comparison ---------------------------------------------------------------


def test_sam3_image_compare_captures_passes_with_documented_tolerances():
    reference, local = _capture()
    report = sam3_upstream.compare_sam3_image_captures(reference, local)
    assert report["passed"] is True
    assert report["tolerances"]["tap.vision.last_hidden_state"] == {"atol": 1e-4, "rtol": 1e-4}
    assert report["selected_tap_pairs"] == [
        {"reference": "vision.last_hidden_state", "local": "vision.last_hidden_state"}
    ]
    assert [field["name"] for field in report["fields"]] == ["tap.vision.last_hidden_state"]


def test_sam3_image_comparison_gate_passes_with_injected_captures(tmp_path):
    full_dir = _full_checkpoint_dir(tmp_path)
    local_weights = tmp_path / "local.npz"
    local_weights.write_bytes(b"fake-local-capture-placeholder")
    reference, local = _capture()

    result = sam3_upstream.evaluate_sam3_image_comparison_gate(
        environ={
            "MLX_CV_SAM3_IMAGE_CHECKPOINT": str(full_dir),
            "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT": str(local_weights),
        },
        min_shard_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda _path, *, inputs=None: reference,
        local_capture_func=lambda _path, *, inputs=None: local,
    )

    assert result.status == "UPSTREAM_PASSED"
    assert result.comparison_report["passed"] is True
    assert sam3_upstream.status_dict(result)["claim_level"] == "upstream_passed"


def test_sam3_image_comparison_gate_blocks_on_numeric_drift(tmp_path):
    full_dir = _full_checkpoint_dir(tmp_path)
    local_weights = tmp_path / "local.npz"
    local_weights.write_bytes(b"fake-local-capture-placeholder")
    reference, local = _capture(drift=1e-2)

    result = sam3_upstream.evaluate_sam3_image_comparison_gate(
        environ={
            "MLX_CV_SAM3_IMAGE_CHECKPOINT": str(full_dir),
            "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT": str(local_weights),
        },
        min_shard_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda _path, *, inputs=None: reference,
        local_capture_func=lambda _path, *, inputs=None: local,
    )

    assert result.status.startswith("BLOCKED:")
    assert "parity drift" in result.blocked_reason
    assert "tap.vision.last_hidden_state" in result.blocked_reason


def test_sam3_image_admitted_checkpoint_reports_missing_local_capture_blocker(tmp_path):
    full_dir = _full_checkpoint_dir(tmp_path)
    reference, _ = _capture()

    result = sam3_upstream.evaluate_sam3_image_comparison_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(full_dir)},
        min_shard_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda _path, *, inputs=None: reference,
    )

    assert result.status.startswith("BLOCKED:")
    assert "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT is unset" in result.blocked_reason
    assert "not a local MLX .npz" in result.blocked_reason


def test_sam3_video_comparison_gate_passes_with_injected_captures(tmp_path):
    pt_dir = _full_checkpoint_dir(tmp_path, name="video", weight="sam3.1_multiplex.pt")
    local_weights = tmp_path / "local_video.npz"
    local_weights.write_bytes(b"fake-local-capture-placeholder")
    reference, local = _capture()

    result = sam3_upstream.evaluate_sam3_video_comparison_gate(
        environ={
            "MLX_CV_SAM3_VIDEO_CHECKPOINT": str(pt_dir),
            "MLX_CV_SAM3_VIDEO_LOCAL_CHECKPOINT": str(local_weights),
        },
        min_shard_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda _path, *, inputs=None: reference,
        local_capture_func=lambda _path, *, inputs=None: local,
    )

    assert result.status == "UPSTREAM_PASSED"
    assert result.comparison_report["passed"] is True


# --- honest not-yet-ported local blockers (no synthetic pass) -----------------


def test_sam3_image_local_capture_is_honest_not_yet_ported(tmp_path):
    npz = tmp_path / "local.npz"
    npz.write_bytes(b"placeholder")
    with pytest.raises(sam3_upstream.Sam3LocalCaptureError) as excinfo:
        sam3_upstream.capture_sam3_image_local(npz)
    message = str(excinfo.value)
    assert "not yet ported" in message
    assert "slice 2" in message


def test_sam3_video_local_capture_is_honest_not_yet_ported(tmp_path):
    npz = tmp_path / "local.npz"
    npz.write_bytes(b"placeholder")
    with pytest.raises(sam3_upstream.Sam3LocalCaptureError) as excinfo:
        sam3_upstream.capture_sam3_video_local(npz)
    message = str(excinfo.value)
    assert "not yet ported" in message
    assert "slices 8-11" in message


def test_sam3_local_capture_rejects_non_npz(tmp_path):
    not_npz = tmp_path / "weights.bin"
    not_npz.write_bytes(b"x")
    with pytest.raises(sam3_upstream.Sam3LocalCaptureError, match="must point to a converted local MLX .npz"):
        sam3_upstream.capture_sam3_image_local(not_npz)


# --- transformers version guard -----------------------------------------------


def test_transformers_version_guard_rejects_unsupported():
    with pytest.raises(sam3_upstream.Sam3ReferenceDependencyError, match="requires transformers"):
        sam3_upstream._check_transformers_version(SimpleNamespace(__version__="4.55.0"))
    with pytest.raises(sam3_upstream.Sam3ReferenceDependencyError):
        sam3_upstream._check_transformers_version(SimpleNamespace(__version__="6.0.0"))
    # In-range versions do not raise.
    sam3_upstream._check_transformers_version(SimpleNamespace(__version__="5.12.1"))
    sam3_upstream._check_transformers_version(SimpleNamespace(__version__="5.10.0"))


# --- parity-status honest-blocker contract ------------------------------------


@pytest.mark.parametrize(
    ("model", "required_env"),
    [("sam3_image", "MLX_CV_REQUIRE_SAM3_IMAGE_GATE"), ("sam3_video", "MLX_CV_REQUIRE_SAM3_VIDEO_GATE")],
)
def test_sam3_parity_status_records_missing_checkpoint_blocker(model, required_env):
    model_status = _status(model)
    required = os.environ.get(required_env) == "1"
    checkpoint = os.environ.get(model_status["checkpoint_env"])
    if not checkpoint:
        status = model_status["status"]
        if status == "UPSTREAM_PASSED":
            assert model_status["passed_gate"]["command"]
        else:
            assert status.startswith("BLOCKED:")
            assert model_status["blocked_reason"]
        if required:
            return
        pytest.skip(f"{model_status['checkpoint_env']} is unset")
