import json
import os
from pathlib import Path

import pytest


STATUS_PATH = Path(".agent/work/2026-06-16-release-parity-hardening/parity-status.json")
REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_LOCATEANYTHING_GATE"


def _status():
    return json.loads(STATUS_PATH.read_text())["models"]["locateanything"]


def _checkpoint_is_usable(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 1_000_000
    if path.is_dir():
        shards = list(path.glob("*.safetensors"))
        return bool(shards) and all(p.stat().st_size > 1_000_000 for p in shards)
    return False


def test_locateanything_upstream_parity_gate_records_missing_checkpoint_blocker():
    model_status = _status()
    required = os.environ.get(REQUIRED_GATE_ENV) == "1"
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
        pytest.skip(f"{checkpoint_path} is not a usable full LocateAnything checkpoint")

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.fail(
        "LocateAnything upstream checkpoint prerequisites are present, but full reference "
        "comparison is not implemented in this workspace yet."
    )
