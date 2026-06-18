import json
import os
import importlib.util
import sys
from pathlib import Path

import numpy as np
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


def _full_checkpoint_dir(tmp_path: Path) -> Path:
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    _write_index(full_dir / "model.safetensors.index.json", ["model-00001-of-00001.safetensors"])
    (full_dir / "model-00001-of-00001.safetensors").write_bytes(b"x" * 8)
    return full_dir


def _patch_reference_path(tmp_path: Path, monkeypatch) -> Path:
    reference_path = tmp_path / "LocateAnything-3B-reference"
    reference_path.mkdir()
    monkeypatch.setattr(locateanything_upstream, "LOCATEANYTHING_REFERENCE_PATH", reference_path)
    return reference_path


def _capture(*, drift: float = 0.0):
    input_ids = np.array([[10, 50, 11]], dtype=np.int64)
    cached = np.arange(8, dtype=np.float32).reshape(1, 8)
    logits = np.zeros((6, 16), dtype=np.float32)
    generated = np.array([70, 100, 200, 300, 400, 71, 70, 500, 250, 71], dtype=np.int32)
    boxes = np.array([[2.0, 2.0, 6.0, 4.0]], dtype=np.float64)
    points = np.array([[10.0, 2.5]], dtype=np.float64)
    taps = {
        "projector": np.ones((1, 4), dtype=np.float32),
        "inputs_embeds": np.ones((1, 3, 4), dtype=np.float32) * 2.0,
        "sampled_tokens": np.array([70, 100, 200, 300, 400, 71], dtype=np.int64),
    }
    local_taps = {key: value.copy() for key, value in taps.items()}
    local_taps["projector"] = local_taps["projector"] + np.float32(drift)
    reference = locateanything_upstream.LocateAnythingCapture(
        source="reference",
        input_ids=input_ids,
        cached_image_features=cached,
        pbd_block_logits=logits,
        generated_ids=generated,
        boxes=boxes,
        points=points,
        taps=taps,
    )
    local = locateanything_upstream.LocateAnythingCapture(
        source="local",
        input_ids=input_ids,
        cached_image_features=cached,
        pbd_block_logits=logits,
        generated_ids=generated,
        boxes=boxes.copy(),
        points=points.copy(),
        taps=local_taps,
    )
    return reference, local


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

    full_dir = _full_checkpoint_dir(tmp_path)
    result = locateanything_upstream.evaluate_locateanything_gate(
        environ={"MLX_CV_LOCATEANYTHING_CHECKPOINT": str(full_dir)},
        min_shard_bytes=4,
    )
    assert result.admitted is True
    assert result.status == "ADMITTED"
    assert result.comparison_report is None
    assert locateanything_upstream.status_dict(result)["claim_level"] == "checkpoint_admitted"


def test_locateanything_compare_captures_passes_with_documented_tolerances():
    reference, local = _capture()

    report = locateanything_upstream.compare_locateanything_captures(reference, local)

    assert report["passed"] is True
    assert report["tolerances"]["boxes"] == {"atol": 1e-6, "rtol": 0.0}
    assert report["tolerances"]["points"] == {"atol": 1e-6, "rtol": 0.0}
    assert report["selected_tap_pairs"] == [
        {"reference": "projector", "local": "projector"},
        {"reference": "inputs_embeds", "local": "inputs_embeds"},
        {"reference": "sampled_tokens", "local": "sampled_tokens"},
    ]
    assert [field["name"] for field in report["fields"]] == [
        "boxes",
        "points",
        "tap.projector",
        "tap.inputs_embeds",
        "tap.sampled_tokens",
    ]


def test_locateanything_comparison_gate_passes_with_injected_captures(tmp_path, monkeypatch):
    full_dir = _full_checkpoint_dir(tmp_path)
    _patch_reference_path(tmp_path, monkeypatch)
    local_weights = tmp_path / "local.npz"
    local_weights.write_bytes(b"fake-local-capture-placeholder")
    reference, local = _capture()

    result = locateanything_upstream.evaluate_locateanything_comparison_gate(
        environ={
            "MLX_CV_LOCATEANYTHING_CHECKPOINT": str(full_dir),
            "MLX_CV_LOCATEANYTHING_LOCAL_CHECKPOINT": str(local_weights),
        },
        min_shard_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda _path, *, reference_path: reference,
        local_capture_func=lambda _path, *, inputs=None: local,
    )

    assert result.status == "UPSTREAM_PASSED"
    assert result.comparison_report["passed"] is True
    assert locateanything_upstream.status_dict(result)["claim_level"] == "upstream_passed"


def test_locateanything_comparison_gate_blocks_on_numeric_drift(tmp_path, monkeypatch):
    full_dir = _full_checkpoint_dir(tmp_path)
    _patch_reference_path(tmp_path, monkeypatch)
    local_weights = tmp_path / "local.npz"
    local_weights.write_bytes(b"fake-local-capture-placeholder")
    reference, local = _capture(drift=1e-2)

    result = locateanything_upstream.evaluate_locateanything_comparison_gate(
        environ={
            "MLX_CV_LOCATEANYTHING_CHECKPOINT": str(full_dir),
            "MLX_CV_LOCATEANYTHING_LOCAL_CHECKPOINT": str(local_weights),
        },
        min_shard_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda _path, *, reference_path: reference,
        local_capture_func=lambda _path, *, inputs=None: local,
    )

    assert result.status.startswith("BLOCKED:")
    assert "parity drift" in result.blocked_reason
    assert "tap.projector" in result.blocked_reason


def test_locateanything_admitted_checkpoint_reports_missing_local_capture_blocker(tmp_path, monkeypatch):
    full_dir = _full_checkpoint_dir(tmp_path)
    _patch_reference_path(tmp_path, monkeypatch)
    reference, _ = _capture()

    result = locateanything_upstream.evaluate_locateanything_comparison_gate(
        environ={"MLX_CV_LOCATEANYTHING_CHECKPOINT": str(full_dir)},
        min_shard_bytes=4,
        check_reference_dependencies=False,
        reference_capture_func=lambda _path, *, reference_path: reference,
    )

    assert result.status.startswith("BLOCKED:")
    assert "MLX_CV_LOCATEANYTHING_LOCAL_CHECKPOINT is unset" in result.blocked_reason
    assert "not a local MLX .npz" in result.blocked_reason


def test_locateanything_upstream_parity_gate_records_missing_checkpoint_blocker():
    model_status = _status()
    required = os.environ.get(REQUIRED_GATE_ENV) == "1"
    checkpoint = os.environ.get(model_status["checkpoint_env"])
    if not checkpoint:
        status = model_status["status"]
        if status == "UPSTREAM_PASSED":
            # Real parity achieved out-of-sandbox (see passed_gate); no checkpoint configured here.
            assert model_status["passed_gate"]["command"]
        else:
            assert status.startswith("BLOCKED:")
            assert model_status["blocked_reason"]
            assert model_status["checkpoint_env"] in status
        if required:
            return
        pytest.skip(f"{model_status['checkpoint_env']} is unset")

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

    result = locateanything_upstream.evaluate_locateanything_comparison_gate(
        environ={model_status["checkpoint_env"]: str(checkpoint_path)}
    )
    assert result.blocked is True
    assert any(
        text in result.blocked_reason
        for text in (
            "reference capture",
            "requires torch",
            "requires torch, transformers, peft",
            "local MLX capture",
            "MLX_CV_LOCATEANYTHING_LOCAL_CHECKPOINT",
            "parity drift",
        )
    )
