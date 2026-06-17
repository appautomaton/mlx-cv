from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sam3_video_upstream", REPO / "tools" / "sam3_video_upstream.py")
assert SPEC is not None and SPEC.loader is not None
sam3_video_upstream = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sam3_video_upstream
SPEC.loader.exec_module(sam3_video_upstream)

SAM3_VIDEO_CHECKPOINT_ENV = sam3_video_upstream.SAM3_VIDEO_CHECKPOINT_ENV
SAM3_VIDEO_CONFIG_ENV = sam3_video_upstream.SAM3_VIDEO_CONFIG_ENV
SAM3_VIDEO_MODEL_ID_ENV = sam3_video_upstream.SAM3_VIDEO_MODEL_ID_ENV
SAM3_VIDEO_CACHE_DIR_ENV = sam3_video_upstream.SAM3_VIDEO_CACHE_DIR_ENV
SAM3_VIDEO_REQUIRED_GATE_ENV = sam3_video_upstream.SAM3_VIDEO_REQUIRED_GATE_ENV
SAM3_VIDEO_OFFICIAL_MODEL_ID = sam3_video_upstream.SAM3_VIDEO_OFFICIAL_MODEL_ID
SAM3_VIDEO_CHECKPOINT_NAME = sam3_video_upstream.SAM3_VIDEO_CHECKPOINT_NAME
SAM3_VIDEO_CONFIG_NAME = sam3_video_upstream.SAM3_VIDEO_CONFIG_NAME
evaluate_sam3_video_gate = sam3_video_upstream.evaluate_sam3_video_gate
evaluate_sam3_video_reference_gate = sam3_video_upstream.evaluate_sam3_video_reference_gate
evaluate_sam3_video_comparison_gate = sam3_video_upstream.evaluate_sam3_video_comparison_gate
status_dict = sam3_video_upstream.status_dict


def _write_admitted_checkpoint_pair(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    config = tmp_path / "config.json"
    config.write_text("{}")
    return checkpoint, config


def _admitted_env(checkpoint: Path, config: Path) -> dict[str, str]:
    return {
        SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
        SAM3_VIDEO_CHECKPOINT_ENV: str(checkpoint),
        SAM3_VIDEO_CONFIG_ENV: str(config),
    }


def test_sam3_video_required_gate_reports_missing_checkpoint_blocker():
    result = evaluate_sam3_video_gate(environ={SAM3_VIDEO_REQUIRED_GATE_ENV: "1"})

    assert result.status == f"BLOCKED:{SAM3_VIDEO_CHECKPOINT_ENV} is unset"
    assert result.blocked is True
    assert result.blocked_reason == f"{SAM3_VIDEO_CHECKPOINT_ENV} is unset"
    assert status_dict(result)["claim_level"] == "external_blocker"
    assert status_dict(result)["official_model_id"] == SAM3_VIDEO_OFFICIAL_MODEL_ID


def test_sam3_video_gate_reports_missing_config_blocker(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-real-but-present")
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CHECKPOINT_ENV: str(checkpoint),
        },
        min_checkpoint_bytes=1,
    )

    assert result.status.startswith("BLOCKED:")
    assert result.blocked_reason == f"{SAM3_VIDEO_CONFIG_ENV} is unset for SAM3 video checkpoint admission"
    assert result.checkpoint_path == str(checkpoint)


def test_sam3_video_gate_reports_unusable_checkpoint_blocker(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    checkpoint.write_bytes(b"stub")
    config = tmp_path / "config.json"
    config.write_text("{}")
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CHECKPOINT_ENV: str(checkpoint),
            SAM3_VIDEO_CONFIG_ENV: str(config),
        }
    )

    assert result.status.startswith("BLOCKED:")
    assert "not a usable SAM3 video checkpoint" in result.blocked_reason
    assert result.checkpoint_path == str(checkpoint)
    assert result.config_path == str(config)


def test_sam3_video_gate_reports_missing_checkpoint_path_blocker(tmp_path):
    checkpoint = tmp_path / "missing.pt"
    config = tmp_path / "config.json"
    config.write_text("{}")
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CHECKPOINT_ENV: str(checkpoint),
            SAM3_VIDEO_CONFIG_ENV: str(config),
        },
        min_checkpoint_bytes=1,
    )

    assert result.status.startswith("BLOCKED:")
    assert "does not point to an existing path" in result.blocked_reason
    assert result.checkpoint_path == str(checkpoint)
    assert result.config_path == str(config)


def test_sam3_video_gate_reports_unsupported_model_id_blocker():
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_MODEL_ID_ENV: "facebook/sam3",
        }
    )

    assert result.status.startswith("BLOCKED:")
    assert result.blocked_reason == (
        f"unsupported SAM3 video model id: facebook/sam3; expected {SAM3_VIDEO_OFFICIAL_MODEL_ID}"
    )
    assert status_dict(result)["blocker_kind"] == "source"


def test_sam3_video_gate_reports_uncached_hf_blocker(tmp_path):
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CACHE_DIR_ENV: str(tmp_path),
        }
    )

    assert result.status.startswith("BLOCKED:")
    assert "not cached" in result.blocked_reason
    assert "Hugging Face auth" in result.blocked_reason
    assert result.checkpoint_path.endswith(SAM3_VIDEO_CHECKPOINT_NAME)
    assert result.config_path.endswith(SAM3_VIDEO_CONFIG_NAME)
    assert status_dict(result)["blocker_kind"] == "download_auth"


def test_sam3_video_gate_admits_explicit_checkpoint_and_config(tmp_path):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    result = evaluate_sam3_video_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
    )

    assert result.status == "ADMITTED"
    assert result.blocked is False
    assert result.admitted is True
    assert result.checkpoint_sha256 == hashlib.sha256(b"checkpoint-bytes").hexdigest()
    assert result.config_sha256 == hashlib.sha256(b"{}").hexdigest()
    assert status_dict(result)["claim_level"] == "checkpoint_admitted"
    assert status_dict(result)["provenance_status"] == "cached"


def test_sam3_video_gate_admits_cache_checkpoint_and_config(tmp_path):
    model_dir = tmp_path / SAM3_VIDEO_OFFICIAL_MODEL_ID.replace("/", "--")
    model_dir.mkdir()
    checkpoint = model_dir / SAM3_VIDEO_CHECKPOINT_NAME
    checkpoint.write_bytes(b"cache-checkpoint")
    config = model_dir / SAM3_VIDEO_CONFIG_NAME
    config.write_text("{}")

    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CACHE_DIR_ENV: str(tmp_path),
        },
        min_checkpoint_bytes=1,
    )

    assert result.status == "ADMITTED"
    assert result.checkpoint_path == str(checkpoint)
    assert result.config_path == str(config)
    assert result.cache_dir == str(tmp_path)


def test_sam3_video_reference_gate_reports_missing_reference_path(tmp_path, monkeypatch):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    monkeypatch.setattr(sam3_video_upstream, "SAM3_VIDEO_REFERENCE_PATH", tmp_path / "missing-reference")

    result = evaluate_sam3_video_reference_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
    )

    assert result.status.startswith("BLOCKED:")
    assert "reference path is missing" in result.blocked_reason
    assert result.checkpoint_sha256 == hashlib.sha256(b"checkpoint-bytes").hexdigest()
    assert status_dict(result)["blocker_kind"] == "reference_path"


def test_sam3_video_reference_gate_reports_missing_reference_surface(tmp_path, monkeypatch):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    reference = tmp_path / "reference"
    reference.mkdir()
    monkeypatch.setattr(sam3_video_upstream, "SAM3_VIDEO_REFERENCE_PATH", reference)

    result = evaluate_sam3_video_reference_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
    )

    assert result.status.startswith("BLOCKED:")
    assert "missing expected Object Multiplex surface" in result.blocked_reason
    assert status_dict(result)["blocker_kind"] == "reference_surface"


def test_sam3_video_reference_gate_reports_missing_torch_dependency(tmp_path, monkeypatch):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)

    def _missing_torch(name):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return __import__(name)

    monkeypatch.setattr(sam3_video_upstream.importlib, "import_module", _missing_torch)
    result = evaluate_sam3_video_reference_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
    )

    assert result.status.startswith("BLOCKED:")
    assert result.blocked_reason == "SAM3 video upstream reference execution requires torch: No module named 'torch'"
    assert status_dict(result)["blocker_kind"] == "reference_runtime"


def test_sam3_video_reference_gate_reports_reference_capture_blocker(tmp_path):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    result = evaluate_sam3_video_reference_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
    )

    assert result.status.startswith("BLOCKED:")
    assert "upstream video/Object Multiplex output capture has not completed" in result.blocked_reason
    assert status_dict(result)["blocker_kind"] == "reference_capture"


def test_sam3_video_comparison_gate_reports_missing_local_components(tmp_path):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    result = evaluate_sam3_video_comparison_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
    )

    assert result.status.startswith("BLOCKED:")
    assert "local MLX checkpoint conversion" in result.blocked_reason
    assert "stable video tap capture" in result.blocked_reason
    assert "output mapper/comparator" in result.blocked_reason
    assert "not implemented" not in result.blocked_reason
    assert status_dict(result)["blocker_kind"] == "local_comparison"
