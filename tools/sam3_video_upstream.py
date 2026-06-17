"""SAM3 video checkpoint gate helpers.

This tool owns external checkpoint admission for the Phase 3 SAM3 video path.
It deliberately does not make video checkpoints loadable through the SAM3
image-mode converter.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


SAM3_VIDEO_CHECKPOINT_ENV = "MLX_CV_SAM3_VIDEO_CHECKPOINT"
SAM3_VIDEO_CONFIG_ENV = "MLX_CV_SAM3_VIDEO_CONFIG"
SAM3_VIDEO_MODEL_ID_ENV = "MLX_CV_SAM3_VIDEO_MODEL_ID"
SAM3_VIDEO_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_SAM3_VIDEO_GATE"
SAM3_VIDEO_STATUS_PATH = Path(".agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json")
SAM3_VIDEO_REFERENCE_PATH = Path("references/sam3")


@dataclass(frozen=True)
class SAM3VideoGateResult:
    status: str
    checkpoint_env: str
    config_env: str
    model_id_env: str
    reference_path: str
    checkpoint_path: str | None = None
    config_path: str | None = None
    model_id: str | None = None
    checkpoint_sha256: str | None = None
    blocked_reason: str | None = None

    @property
    def blocked(self) -> bool:
        return self.status.startswith("BLOCKED:")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _block(reason: str, *, environ: Mapping[str, str]) -> SAM3VideoGateResult:
    return SAM3VideoGateResult(
        status=f"BLOCKED:{reason}",
        checkpoint_env=SAM3_VIDEO_CHECKPOINT_ENV,
        config_env=SAM3_VIDEO_CONFIG_ENV,
        model_id_env=SAM3_VIDEO_MODEL_ID_ENV,
        reference_path=str(SAM3_VIDEO_REFERENCE_PATH),
        checkpoint_path=environ.get(SAM3_VIDEO_CHECKPOINT_ENV),
        config_path=environ.get(SAM3_VIDEO_CONFIG_ENV),
        model_id=environ.get(SAM3_VIDEO_MODEL_ID_ENV),
        blocked_reason=reason,
    )


def evaluate_sam3_video_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_checkpoint_bytes: int = 1_000_000,
) -> SAM3VideoGateResult:
    env = os.environ if environ is None else environ
    checkpoint = env.get(SAM3_VIDEO_CHECKPOINT_ENV)
    if not checkpoint:
        return _block(f"{SAM3_VIDEO_CHECKPOINT_ENV} is unset", environ=env)

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        return _block(f"{SAM3_VIDEO_CHECKPOINT_ENV} does not point to a file: {checkpoint_path}", environ=env)
    if checkpoint_path.stat().st_size < min_checkpoint_bytes:
        return _block(f"{checkpoint_path} is not a usable SAM3 video checkpoint", environ=env)
    if not SAM3_VIDEO_REFERENCE_PATH.exists():
        return _block(f"SAM3 reference path is missing: {SAM3_VIDEO_REFERENCE_PATH}", environ=env)

    return _block(
        "SAM3 video checkpoint is present, but upstream-vs-local video comparison is not implemented in this workspace yet",
        environ=env,
    )


def status_dict(result: SAM3VideoGateResult) -> dict:
    out = asdict(result)
    out["phase"] = "sam3-video-object-multiplex"
    out["model"] = "sam3_video"
    out["display_name"] = "SAM 3.1 Video / Object Multiplex"
    out["claim_level"] = "external_blocker" if result.blocked else "upstream_passed"
    return out


def write_status(result: SAM3VideoGateResult, path: Path = SAM3_VIDEO_STATUS_PATH) -> None:
    path.write_text(json.dumps(status_dict(result), indent=2) + "\n")


def main() -> int:
    result = evaluate_sam3_video_gate()
    write_status(result)
    print(json.dumps(status_dict(result), indent=2))
    return 0 if not result.blocked else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
