"""SAM 3.1 image checkpoint admission and upstream parity gate helpers."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


SAM3_IMAGE_CHECKPOINT_ENV = "MLX_CV_SAM3_IMAGE_CHECKPOINT"
SAM3_IMAGE_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_SAM3_IMAGE_GATE"
SAM3_IMAGE_REFERENCE_PATH = Path("references/sam3")
_SUPPORTED_IMAGE_FORMATS = {".npz", ".safetensors"}
_VIDEO_KEY_PARTS = (
    "video",
    "tracker",
    "track",
    "memory_encoder",
    "memory_attention",
    "temporal",
    "maskmem",
    "multiplex",
    "sam2_predictor",
    "obj_ptr",
)


@dataclass(frozen=True)
class SAM3ImageGateResult:
    status: str
    checkpoint_env: str
    required_gate_env: str
    reference_path: str
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    blocked_reason: str | None = None
    admitted: bool = False

    @property
    def blocked(self) -> bool:
        return self.status.startswith("BLOCKED:")


def required_gate_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SAM3_IMAGE_REQUIRED_GATE_ENV) == "1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _block(reason: str, *, environ: Mapping[str, str]) -> SAM3ImageGateResult:
    return SAM3ImageGateResult(
        status=f"BLOCKED:{reason}",
        checkpoint_env=SAM3_IMAGE_CHECKPOINT_ENV,
        required_gate_env=SAM3_IMAGE_REQUIRED_GATE_ENV,
        reference_path=str(SAM3_IMAGE_REFERENCE_PATH),
        checkpoint_path=environ.get(SAM3_IMAGE_CHECKPOINT_ENV),
        blocked_reason=reason,
    )


def _admit(path: Path, *, environ: Mapping[str, str]) -> SAM3ImageGateResult:
    return SAM3ImageGateResult(
        status="ADMITTED",
        checkpoint_env=SAM3_IMAGE_CHECKPOINT_ENV,
        required_gate_env=SAM3_IMAGE_REQUIRED_GATE_ENV,
        reference_path=str(SAM3_IMAGE_REFERENCE_PATH),
        checkpoint_path=str(path),
        checkpoint_sha256=_sha256(path),
        admitted=True,
    )


def _contains_video_or_tracker_key(keys: list[str]) -> bool:
    return any(any(part in key.lower() for part in _VIDEO_KEY_PARTS) for key in keys)


def _npz_keys(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as npz:
        return list(npz.files)


def evaluate_sam3_image_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_checkpoint_bytes: int = 1_000_000,
) -> SAM3ImageGateResult:
    env = os.environ if environ is None else environ
    checkpoint = env.get(SAM3_IMAGE_CHECKPOINT_ENV)
    if not checkpoint:
        return _block(f"{SAM3_IMAGE_CHECKPOINT_ENV} is unset", environ=env)

    path = Path(checkpoint)
    if not path.exists():
        return _block(f"{SAM3_IMAGE_CHECKPOINT_ENV} does not point to an existing path: {path}", environ=env)
    if not path.is_file():
        return _block(f"{SAM3_IMAGE_CHECKPOINT_ENV} does not point to a file: {path}", environ=env)
    if path.stat().st_size < min_checkpoint_bytes:
        return _block(f"{path} is not a usable SAM3 image checkpoint", environ=env)
    if path.suffix not in _SUPPORTED_IMAGE_FORMATS:
        return _block(
            f"SAM3 image checkpoint format is not loadable by the local image converter: {path.suffix or path.name}",
            environ=env,
        )
    if path.suffix == ".npz" and _contains_video_or_tracker_key(_npz_keys(path)):
        return _block(f"{path} appears to be a video/tracker checkpoint, not a SAM3 image checkpoint", environ=env)

    return _admit(path, environ=env)


def evaluate_sam3_image_comparison_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_checkpoint_bytes: int = 1_000_000,
    check_reference_dependencies: bool = True,
) -> SAM3ImageGateResult:
    env = os.environ if environ is None else environ
    admission = evaluate_sam3_image_gate(environ=env, min_checkpoint_bytes=min_checkpoint_bytes)
    if admission.blocked:
        return admission

    if not SAM3_IMAGE_REFERENCE_PATH.exists():
        return _block(f"SAM3 reference path is missing: {SAM3_IMAGE_REFERENCE_PATH}", environ=env)

    if check_reference_dependencies:
        try:
            __import__("torch")
        except Exception as exc:
            return _block(f"SAM3 image upstream comparison requires torch: {exc}", environ=env)

    return _block(
        "SAM3 image checkpoint is admitted, but stable image tap capture and upstream-vs-MLX "
        "comparison are missing for masks, paired detections, and token/text evidence",
        environ=env,
    )


def status_dict(result: SAM3ImageGateResult) -> dict:
    out = asdict(result)
    out["model"] = "sam3_image"
    out["display_name"] = "SAM 3.1 image-mode"
    out["claim_level"] = "external_blocker" if result.blocked else "checkpoint_admitted"
    return out
