"""SAM3 video checkpoint gate helpers.

This tool owns external checkpoint admission for the SAM 3.1 video/Object
Multiplex path. It deliberately does not make video checkpoints loadable
through the SAM3 image-mode converter.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


SAM3_VIDEO_CHECKPOINT_ENV = "MLX_CV_SAM3_VIDEO_CHECKPOINT"
SAM3_VIDEO_CONFIG_ENV = "MLX_CV_SAM3_VIDEO_CONFIG"
SAM3_VIDEO_MODEL_ID_ENV = "MLX_CV_SAM3_VIDEO_MODEL_ID"
SAM3_VIDEO_CACHE_DIR_ENV = "MLX_CV_SAM3_VIDEO_CACHE_DIR"
SAM3_VIDEO_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_SAM3_VIDEO_GATE"
SAM3_VIDEO_STATUS_PATH = Path(
    ".agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json"
)
SAM3_VIDEO_LOCAL_CONTRACT_STATUS_PATH = Path(
    ".agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json"
)
SAM3_VIDEO_REFERENCE_PATH = Path("references/sam3")
SAM3_VIDEO_OFFICIAL_MODEL_ID = "facebook/sam3.1"
SAM3_VIDEO_CHECKPOINT_NAME = "sam3.1_multiplex.pt"
SAM3_VIDEO_CONFIG_NAME = "config.json"
SAM3_VIDEO_SOURCE_URL = "https://huggingface.co/facebook/sam3.1"
SAM3_VIDEO_LICENSE_OR_TERMS = "SAM license; Hugging Face gated access and accepted terms/auth required"
SAM3_VIDEO_SUPPORTED_MODEL_IDS = {SAM3_VIDEO_OFFICIAL_MODEL_ID}
SAM3_VIDEO_REFERENCE_SURFACES = (
    "build_sam3_video_predictor",
    "build_sam3_multiplex_video_predictor",
    "build_sam3_predictor(version=\"sam3.1\")",
    "Sam3TrackerPredictor",
    "SimpleMaskEncoder",
    "MultiplexController",
    "VideoTrackingDynamicMultiplex",
    "start_session",
    "add_prompt",
    "propagate_in_video",
)
_REFERENCE_SURFACE_PATTERNS = {
    "build_sam3_video_predictor": ("sam3/model_builder.py", "def build_sam3_video_predictor"),
    "build_sam3_multiplex_video_predictor": ("sam3/model_builder.py", "def build_sam3_multiplex_video_predictor"),
    "build_sam3_predictor(version=\"sam3.1\")": ("sam3/model_builder.py", "version == \"sam3.1\""),
    "Sam3TrackerPredictor": ("sam3/model_builder.py", "Sam3TrackerPredictor"),
    "SimpleMaskEncoder": ("sam3/model_builder.py", "SimpleMaskEncoder"),
    "MultiplexController": ("sam3/model_builder.py", "MultiplexController"),
    "VideoTrackingDynamicMultiplex": ("sam3/model_builder.py", "VideoTrackingDynamicMultiplex"),
    "start_session": ("sam3/model/sam3_base_predictor.py", "def start_session"),
    "add_prompt": ("sam3/model/sam3_base_predictor.py", "def add_prompt"),
    "propagate_in_video": ("sam3/model/sam3_base_predictor.py", "def propagate_in_video"),
}


@dataclass(frozen=True)
class SAM3VideoGateResult:
    status: str
    checkpoint_env: str
    config_env: str
    model_id_env: str
    cache_dir_env: str
    required_gate_env: str
    reference_path: str
    checkpoint_path: str | None = None
    config_path: str | None = None
    cache_dir: str | None = None
    model_id: str | None = None
    checkpoint_sha256: str | None = None
    config_sha256: str | None = None
    blocked_reason: str | None = None
    blocker_kind: str | None = None
    admitted: bool = False

    @property
    def blocked(self) -> bool:
        return self.status.startswith("BLOCKED:")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def required_gate_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SAM3_VIDEO_REQUIRED_GATE_ENV) == "1"


def _model_id(environ: Mapping[str, str]) -> str:
    return environ.get(SAM3_VIDEO_MODEL_ID_ENV, SAM3_VIDEO_OFFICIAL_MODEL_ID)


def _cache_model_dir(cache_dir: Path, model_id: str) -> Path:
    return cache_dir / model_id.replace("/", "--")


def _block(
    reason: str,
    *,
    environ: Mapping[str, str],
    checkpoint_path: Path | None = None,
    config_path: Path | None = None,
    blocker_kind: str = "external",
) -> SAM3VideoGateResult:
    return SAM3VideoGateResult(
        status=f"BLOCKED:{reason}",
        checkpoint_env=SAM3_VIDEO_CHECKPOINT_ENV,
        config_env=SAM3_VIDEO_CONFIG_ENV,
        model_id_env=SAM3_VIDEO_MODEL_ID_ENV,
        cache_dir_env=SAM3_VIDEO_CACHE_DIR_ENV,
        required_gate_env=SAM3_VIDEO_REQUIRED_GATE_ENV,
        reference_path=str(SAM3_VIDEO_REFERENCE_PATH),
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else environ.get(SAM3_VIDEO_CHECKPOINT_ENV),
        config_path=str(config_path) if config_path is not None else environ.get(SAM3_VIDEO_CONFIG_ENV),
        cache_dir=environ.get(SAM3_VIDEO_CACHE_DIR_ENV),
        model_id=_model_id(environ),
        blocked_reason=reason,
        blocker_kind=blocker_kind,
    )


def _admit(
    checkpoint_path: Path,
    config_path: Path,
    *,
    environ: Mapping[str, str],
) -> SAM3VideoGateResult:
    return SAM3VideoGateResult(
        status="ADMITTED",
        checkpoint_env=SAM3_VIDEO_CHECKPOINT_ENV,
        config_env=SAM3_VIDEO_CONFIG_ENV,
        model_id_env=SAM3_VIDEO_MODEL_ID_ENV,
        cache_dir_env=SAM3_VIDEO_CACHE_DIR_ENV,
        required_gate_env=SAM3_VIDEO_REQUIRED_GATE_ENV,
        reference_path=str(SAM3_VIDEO_REFERENCE_PATH),
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        cache_dir=environ.get(SAM3_VIDEO_CACHE_DIR_ENV),
        model_id=_model_id(environ),
        checkpoint_sha256=_sha256(checkpoint_path),
        config_sha256=_sha256(config_path),
        admitted=True,
    )


def _block_from_admission(
    admission: SAM3VideoGateResult,
    reason: str,
    *,
    blocker_kind: str,
) -> SAM3VideoGateResult:
    return SAM3VideoGateResult(
        status=f"BLOCKED:{reason}",
        checkpoint_env=SAM3_VIDEO_CHECKPOINT_ENV,
        config_env=SAM3_VIDEO_CONFIG_ENV,
        model_id_env=SAM3_VIDEO_MODEL_ID_ENV,
        cache_dir_env=SAM3_VIDEO_CACHE_DIR_ENV,
        required_gate_env=SAM3_VIDEO_REQUIRED_GATE_ENV,
        reference_path=str(SAM3_VIDEO_REFERENCE_PATH),
        checkpoint_path=admission.checkpoint_path,
        config_path=admission.config_path,
        cache_dir=admission.cache_dir,
        model_id=admission.model_id,
        checkpoint_sha256=admission.checkpoint_sha256,
        config_sha256=admission.config_sha256,
        blocked_reason=reason,
        blocker_kind=blocker_kind,
    )


def _resolve_checkpoint_and_config(environ: Mapping[str, str]) -> tuple[Path | None, Path | None, SAM3VideoGateResult | None]:
    model_id = _model_id(environ)
    if model_id not in SAM3_VIDEO_SUPPORTED_MODEL_IDS:
        return None, None, _block(
            f"unsupported SAM3 video model id: {model_id}; expected {SAM3_VIDEO_OFFICIAL_MODEL_ID}",
            environ=environ,
            blocker_kind="source",
        )

    checkpoint = environ.get(SAM3_VIDEO_CHECKPOINT_ENV)
    config = environ.get(SAM3_VIDEO_CONFIG_ENV)
    cache_dir = environ.get(SAM3_VIDEO_CACHE_DIR_ENV)

    if checkpoint or config:
        if not checkpoint:
            return None, None, _block(f"{SAM3_VIDEO_CHECKPOINT_ENV} is unset", environ=environ)
        if not config:
            return Path(checkpoint), None, _block(
                f"{SAM3_VIDEO_CONFIG_ENV} is unset for SAM3 video checkpoint admission",
                environ=environ,
                checkpoint_path=Path(checkpoint),
                blocker_kind="config",
            )
        return Path(checkpoint), Path(config), None

    if cache_dir:
        model_dir = _cache_model_dir(Path(cache_dir), model_id)
        checkpoint_path = model_dir / SAM3_VIDEO_CHECKPOINT_NAME
        config_path = model_dir / SAM3_VIDEO_CONFIG_NAME
        if not checkpoint_path.exists() or not config_path.exists():
            return checkpoint_path, config_path, _block(
                "SAM3 video checkpoint/config are not cached under "
                f"{model_dir}; download requires Hugging Face auth and accepted terms for {SAM3_VIDEO_SOURCE_URL}",
                environ=environ,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                blocker_kind="download_auth",
            )
        return checkpoint_path, config_path, None

    return None, None, _block(f"{SAM3_VIDEO_CHECKPOINT_ENV} is unset", environ=environ)


def evaluate_sam3_video_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_checkpoint_bytes: int = 1_000_000,
    min_config_bytes: int = 2,
) -> SAM3VideoGateResult:
    env = os.environ if environ is None else environ
    checkpoint_path, config_path, blocker = _resolve_checkpoint_and_config(env)
    if blocker is not None:
        return blocker
    assert checkpoint_path is not None
    assert config_path is not None

    if not checkpoint_path.exists():
        return _block(
            f"{SAM3_VIDEO_CHECKPOINT_ENV} does not point to an existing path: {checkpoint_path}",
            environ=env,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
    if not checkpoint_path.is_file():
        return _block(
            f"{SAM3_VIDEO_CHECKPOINT_ENV} does not point to a file: {checkpoint_path}",
            environ=env,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
    if checkpoint_path.stat().st_size < min_checkpoint_bytes:
        return _block(
            f"{checkpoint_path} is not a usable SAM3 video checkpoint",
            environ=env,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            blocker_kind="checkpoint",
        )

    if not config_path.exists():
        return _block(
            f"{SAM3_VIDEO_CONFIG_ENV} does not point to an existing path: {config_path}",
            environ=env,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            blocker_kind="config",
        )
    if not config_path.is_file():
        return _block(
            f"{SAM3_VIDEO_CONFIG_ENV} does not point to a file: {config_path}",
            environ=env,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            blocker_kind="config",
        )
    if config_path.stat().st_size < min_config_bytes:
        return _block(
            f"{config_path} is not a usable SAM3 video config",
            environ=env,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            blocker_kind="config",
        )

    return _admit(checkpoint_path, config_path, environ=env)


def _missing_reference_surfaces(reference_path: Path) -> list[str]:
    missing: list[str] = []
    for surface, (relative_path, pattern) in _REFERENCE_SURFACE_PATTERNS.items():
        path = reference_path / relative_path
        if not path.exists() or pattern not in path.read_text():
            missing.append(surface)
    return missing


def evaluate_sam3_video_reference_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_checkpoint_bytes: int = 1_000_000,
    min_config_bytes: int = 2,
    check_reference_dependencies: bool = True,
) -> SAM3VideoGateResult:
    env = os.environ if environ is None else environ
    admission = evaluate_sam3_video_gate(
        environ=env,
        min_checkpoint_bytes=min_checkpoint_bytes,
        min_config_bytes=min_config_bytes,
    )
    if admission.blocked:
        return admission

    if not SAM3_VIDEO_REFERENCE_PATH.exists():
        return _block_from_admission(
            admission,
            f"SAM3 reference path is missing: {SAM3_VIDEO_REFERENCE_PATH}",
            blocker_kind="reference_path",
        )

    missing_surfaces = _missing_reference_surfaces(SAM3_VIDEO_REFERENCE_PATH)
    if missing_surfaces:
        return _block_from_admission(
            admission,
            "SAM3 video reference tree is missing expected Object Multiplex surface(s): "
            + ", ".join(missing_surfaces[:3]),
            blocker_kind="reference_surface",
        )

    if check_reference_dependencies:
        try:
            importlib.import_module("torch")
        except Exception as exc:
            return _block_from_admission(
                admission,
                f"SAM3 video upstream reference execution requires torch: {exc}",
                blocker_kind="reference_runtime",
            )

    return _block_from_admission(
        admission,
        "SAM3 video checkpoint/config are admitted and reference surfaces are present, "
        "but upstream video/Object Multiplex output capture has not completed in this workspace",
        blocker_kind="reference_capture",
    )


def status_dict(result: SAM3VideoGateResult) -> dict:
    out = asdict(result)
    out["schema_version"] = 1
    out["phase"] = "sam3-video-real-checkpoint-admission"
    out["model"] = "sam3_video"
    out["display_name"] = "SAM 3.1 Video / Object Multiplex"
    out["claim_level"] = "external_blocker" if result.blocked else "checkpoint_admitted"
    out["blocker_kind"] = result.blocker_kind
    out["official_model_id"] = SAM3_VIDEO_OFFICIAL_MODEL_ID
    out["checkpoint_name"] = SAM3_VIDEO_CHECKPOINT_NAME
    out["config_name"] = SAM3_VIDEO_CONFIG_NAME
    out["source_url"] = SAM3_VIDEO_SOURCE_URL
    out["license_or_terms"] = SAM3_VIDEO_LICENSE_OR_TERMS
    out["provenance_status"] = "cached" if result.admitted else "not_cached"
    out["reference_surfaces"] = list(SAM3_VIDEO_REFERENCE_SURFACES)
    out["comparison_scope"] = "SAM 3.1 video/Object Multiplex checkpoint admission and smallest stable local comparison"
    out["local_contract_status"] = str(SAM3_VIDEO_LOCAL_CONTRACT_STATUS_PATH)
    out["release_parity_matrix"] = (
        ".agent/work/2026-06-16-release-parity-hardening/parity-status.json is intentionally "
        "not expanded for sam3_video"
    )
    return out


def write_status(result: SAM3VideoGateResult, path: Path = SAM3_VIDEO_STATUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status_dict(result), indent=2) + "\n")


def main() -> int:
    result = evaluate_sam3_video_gate()
    write_status(result)
    print(json.dumps(status_dict(result), indent=2))
    return 0 if not result.blocked else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
