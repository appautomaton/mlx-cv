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


def _image_taps(drift: float = 0.0) -> dict:
    # End-to-end detector taps (slice 7). Shapes are illustrative; the comparison is
    # shape/tolerance-driven, so small fixed arrays exercise all selected tap pairs.
    base = {
        "vision.last_hidden_state": np.ones((1, 4, 8), dtype=np.float32),
        "pred_logits": np.ones((1, 5), dtype=np.float32),
        "pred_boxes": np.full((1, 5, 4), 0.5, dtype=np.float32),
        "presence_logits": np.ones((1, 1), dtype=np.float32),
        "pred_masks": np.ones((1, 5, 8, 8), dtype=np.float32),
        "semantic_seg": np.ones((1, 1, 8, 8), dtype=np.float32),
    }
    return {name: value + np.float32(drift) for name, value in base.items()}


def _capture(*, drift: float = 0.0):
    pixel_values = np.zeros((1, 3, 4, 4), dtype=np.float32)
    input_ids = np.array([[1, 2, 3, 49407]], dtype=np.int64)
    attention_mask = np.ones((1, 4), dtype=np.int64)
    reference = sam3_upstream.Sam3Capture(
        source="upstream_reference",
        pixel_values=pixel_values,
        taps=_image_taps(),
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    local = sam3_upstream.Sam3Capture(
        source="mlx_local",
        pixel_values=pixel_values.copy(),
        taps=_image_taps(drift=drift),
        input_ids=input_ids.copy(),
        attention_mask=attention_mask.copy(),
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
    assert report["selected_tap_pairs"][0] == {
        "reference": "vision.last_hidden_state",
        "local": "vision.last_hidden_state",
    }
    assert [field["name"] for field in report["fields"]] == [
        "tap.vision.last_hidden_state",
        "tap.pred_logits",
        "tap.pred_boxes",
        "tap.presence_logits",
        "tap.pred_masks",
        "tap.semantic_seg",
    ]


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


# --- local capture blockers (no synthetic pass) -------------------------------


def test_sam3_image_local_capture_rejects_invalid_npz(tmp_path):
    # The image detector is ported (slice 7): a malformed .npz fails to load with a
    # precise weight-load blocker rather than a synthetic pass.
    npz = tmp_path / "local.npz"
    npz.write_bytes(b"placeholder")
    with pytest.raises(sam3_upstream.Sam3LocalCaptureError) as excinfo:
        sam3_upstream.capture_sam3_image_local(npz)
    assert "weight load failed" in str(excinfo.value)


def test_sam3_video_local_capture_loads_faithful_model_and_fails_on_invalid_weights(tmp_path):
    npz = tmp_path / "local.npz"
    npz.write_bytes(b"placeholder")
    with pytest.raises(sam3_upstream.Sam3LocalCaptureError) as excinfo:
        sam3_upstream.capture_sam3_video_local(npz)
    # The streaming/association loop is ported now: the capture is wired to the faithful
    # video model and fails honestly at weight load on an unusable checkpoint (no "not yet
    # ported" stub, no synthetic pass).
    assert "weight load failed" in str(excinfo.value)


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

# --- end-to-end gate runner (out-of-sandbox with the gated checkpoint) ---------


def test_sam3_image_comparison_gate_runs_when_required():
    """When the image gate is required, the real transformers-vs-MLX gate must pass.

    In-sandbox (no gated weights) this skips; out-of-sandbox the user sets
    MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1 with the HF checkpoint + converted local .npz
    and the end-to-end parity gate must report UPSTREAM_PASSED (no synthetic pass).
    """

    if os.environ.get("MLX_CV_REQUIRE_SAM3_IMAGE_GATE") != "1":
        pytest.skip("MLX_CV_REQUIRE_SAM3_IMAGE_GATE != 1; image gate runs out-of-sandbox with the gated checkpoint")
    result = sam3_upstream.evaluate_sam3_image_comparison_gate()
    assert result.status == "UPSTREAM_PASSED", result.blocked_reason
