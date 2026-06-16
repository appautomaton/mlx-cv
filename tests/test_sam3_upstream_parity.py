import json
import os
from pathlib import Path

import pytest


STATUS_PATH = Path(".agent/work/2026-06-16-release-parity-hardening/parity-status.json")


def _status():
    return json.loads(STATUS_PATH.read_text())["models"]["sam3_image"]


def _checkpoint_is_usable(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1_000_000


def test_sam3_image_upstream_parity_gate_records_missing_checkpoint_or_taps_blocker():
    model_status = _status()
    reference_path = Path(model_status["reference_path"])
    assert reference_path.exists(), "SAM3 source checkout should be present for image-mode hardening"

    checkpoint = os.environ.get(model_status["checkpoint_env"])
    if not checkpoint:
        assert model_status["status"].startswith("BLOCKED:")
        assert model_status["blocked_reason"]
        pytest.skip(f"{model_status['checkpoint_env']} is unset; blocker recorded in parity-status.json")

    checkpoint_path = Path(checkpoint)
    if not _checkpoint_is_usable(checkpoint_path):
        assert model_status["status"].startswith("BLOCKED:")
        assert model_status["blocked_reason"]
        pytest.skip(f"{checkpoint_path} is not a usable SAM3 image checkpoint")

    pytest.importorskip("torch")
    pytest.fail(
        "SAM3 image checkpoint prerequisites are present, but stable upstream "
        "image-mode tap comparison is not implemented in this workspace yet."
    )
