from __future__ import annotations

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
SAM3_VIDEO_REQUIRED_GATE_ENV = sam3_video_upstream.SAM3_VIDEO_REQUIRED_GATE_ENV
evaluate_sam3_video_gate = sam3_video_upstream.evaluate_sam3_video_gate
status_dict = sam3_video_upstream.status_dict


def test_sam3_video_required_gate_reports_missing_checkpoint_blocker():
    result = evaluate_sam3_video_gate(environ={SAM3_VIDEO_REQUIRED_GATE_ENV: "1"})

    assert result.status == f"BLOCKED:{SAM3_VIDEO_CHECKPOINT_ENV} is unset"
    assert result.blocked is True
    assert result.blocked_reason == f"{SAM3_VIDEO_CHECKPOINT_ENV} is unset"
    assert status_dict(result)["claim_level"] == "external_blocker"


def test_sam3_video_gate_reports_unusable_checkpoint_blocker(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    checkpoint.write_bytes(b"stub")
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CHECKPOINT_ENV: str(checkpoint),
        }
    )

    assert result.status.startswith("BLOCKED:")
    assert "not a usable SAM3 video checkpoint" in result.blocked_reason
    assert result.checkpoint_path == str(checkpoint)
