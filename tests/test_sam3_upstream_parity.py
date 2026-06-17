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
    np.savez(tiny, **{"decoder.query_embed": np.zeros((1,), dtype=np.float32)})
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
    np.savez(image, **{"decoder.query_embed": np.zeros((1,), dtype=np.float32)})
    result = sam3_image_upstream.evaluate_sam3_image_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(image)},
        min_checkpoint_bytes=4,
    )
    assert result.admitted is True


def test_sam3_image_admitted_checkpoint_reports_missing_tap_comparison(tmp_path):
    image = tmp_path / "image.npz"
    np.savez(image, **{"decoder.query_embed": np.zeros((1,), dtype=np.float32)})

    result = sam3_image_upstream.evaluate_sam3_image_comparison_gate(
        environ={"MLX_CV_SAM3_IMAGE_CHECKPOINT": str(image)},
        min_checkpoint_bytes=4,
        check_reference_dependencies=False,
    )

    assert result.status.startswith("BLOCKED:")
    assert "stable image tap capture" in result.blocked_reason
    assert "paired detections" in result.blocked_reason


def test_sam3_image_upstream_parity_gate_records_missing_checkpoint_or_taps_blocker():
    model_status = _status()
    required = os.environ.get(REQUIRED_GATE_ENV) == "1"
    reference_path = Path(model_status["reference_path"])
    assert reference_path.exists(), "SAM3 source checkout should be present for image-mode hardening"

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

    result = sam3_image_upstream.evaluate_sam3_image_comparison_gate(
        environ={model_status["checkpoint_env"]: str(checkpoint_path)}
    )
    assert result.blocked is True
    assert "comparison" in result.blocked_reason or "requires torch" in result.blocked_reason
