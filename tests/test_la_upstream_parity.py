import json
import os
import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
STATUS_PATH = Path(".agent/work/2026-06-16-release-parity-hardening/parity-status.json")
REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_LOCATEANYTHING_GATE"


SPEC = importlib.util.spec_from_file_location("locateanything_upstream", REPO / "tools" / "locateanything_upstream.py")
assert SPEC is not None and SPEC.loader is not None
locateanything_upstream = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = locateanything_upstream
SPEC.loader.exec_module(locateanything_upstream)


def _status():
    return json.loads(STATUS_PATH.read_text())["models"]["locateanything"]


def _checkpoint_is_usable(path: Path) -> bool:
    return locateanything_upstream.evaluate_locateanything_gate(
        environ={"MLX_CV_LOCATEANYTHING_CHECKPOINT": str(path)}
    ).admitted


def _write_index(path: Path, shards: list[str]) -> None:
    path.write_text(json.dumps({"weight_map": {f"tensor_{i}": shard for i, shard in enumerate(shards)}}))


def test_locateanything_gate_classifies_checkpoint_admission_failures(tmp_path):
    result = locateanything_upstream.evaluate_locateanything_gate(environ={})
    assert result.status == "BLOCKED:MLX_CV_LOCATEANYTHING_CHECKPOINT is unset"

    result = locateanything_upstream.evaluate_locateanything_gate(
        environ={"MLX_CV_LOCATEANYTHING_CHECKPOINT": str(tmp_path / "missing")}
    )
    assert "does not point to an existing path" in result.blocked_reason

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    result = locateanything_upstream.evaluate_locateanything_gate(
        environ={"MLX_CV_LOCATEANYTHING_CHECKPOINT": str(checkpoint_dir)}
    )
    assert "missing model.safetensors.index.json" in result.blocked_reason

    _write_index(checkpoint_dir / "model.safetensors.index.json", ["model-00001-of-00002.safetensors"])
    result = locateanything_upstream.evaluate_locateanything_gate(
        environ={"MLX_CV_LOCATEANYTHING_CHECKPOINT": str(checkpoint_dir)}
    )
    assert "missing shard" in result.blocked_reason

    shard = checkpoint_dir / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"stub")
    result = locateanything_upstream.evaluate_locateanything_gate(
        environ={"MLX_CV_LOCATEANYTHING_CHECKPOINT": str(checkpoint_dir)}
    )
    assert "LFS stubs or incomplete" in result.blocked_reason

    unsupported = tmp_path / "weights.bin"
    unsupported.write_bytes(b"not-safetensors")
    result = locateanything_upstream.evaluate_locateanything_gate(
        environ={"MLX_CV_LOCATEANYTHING_CHECKPOINT": str(unsupported)}
    )
    assert "unsupported LocateAnything checkpoint format" in result.blocked_reason

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    _write_index(full_dir / "model.safetensors.index.json", ["model-00001-of-00001.safetensors"])
    (full_dir / "model-00001-of-00001.safetensors").write_bytes(b"x" * 8)
    result = locateanything_upstream.evaluate_locateanything_gate(
        environ={"MLX_CV_LOCATEANYTHING_CHECKPOINT": str(full_dir)},
        min_shard_bytes=4,
    )
    assert result.admitted is True


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
        gate = locateanything_upstream.evaluate_locateanything_gate(
            environ={model_status["checkpoint_env"]: str(checkpoint_path)}
        )
        assert model_status["status"].startswith("BLOCKED:")
        assert model_status["blocked_reason"]
        assert gate.blocked is True
        if required:
            return
        pytest.skip(f"{checkpoint_path} is not a usable full LocateAnything checkpoint")

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.fail(
        "LocateAnything upstream checkpoint prerequisites are present, but full reference "
        "comparison is not implemented in this workspace yet."
    )
