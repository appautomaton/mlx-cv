import hashlib
import json
import os
from pathlib import Path

import pytest


STATUS_PATH = Path(".agent/work/2026-06-16-release-parity-hardening/parity-status.json")


def _status():
    return json.loads(STATUS_PATH.read_text())["models"]["rfdetr"]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_rfdetr_upstream_parity_gate_records_missing_checkpoint_blocker():
    model_status = _status()
    checkpoint = os.environ.get(model_status["checkpoint_env"])
    if not checkpoint:
        assert model_status["status"].startswith("BLOCKED:")
        assert model_status["blocked_reason"]
        pytest.skip(f"{model_status['checkpoint_env']} is unset; blocker recorded in parity-status.json")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        assert model_status["status"].startswith("BLOCKED:")
        assert model_status["blocked_reason"]
        pytest.skip(f"{checkpoint_path} is not available")

    if _md5(checkpoint_path) != model_status["expected_md5"]:
        assert model_status["status"].startswith("BLOCKED:")
        assert model_status["blocked_reason"]
        pytest.skip(f"{checkpoint_path} does not match expected RF-DETR Nano MD5")

    pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    pytest.fail(
        "RF-DETR Nano upstream checkpoint prerequisites are present, but full "
        "reference comparison is not implemented in this workspace yet."
    )
