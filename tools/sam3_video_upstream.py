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
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SAM3_VIDEO_CHECKPOINT_ENV = "MLX_CV_SAM3_VIDEO_CHECKPOINT"
SAM3_VIDEO_CONFIG_ENV = "MLX_CV_SAM3_VIDEO_CONFIG"
SAM3_VIDEO_MODEL_ID_ENV = "MLX_CV_SAM3_VIDEO_MODEL_ID"
SAM3_VIDEO_CACHE_DIR_ENV = "MLX_CV_SAM3_VIDEO_CACHE_DIR"
SAM3_VIDEO_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_SAM3_VIDEO_GATE"
SAM3_VIDEO_LOCAL_CHECKPOINT_ENV = "MLX_CV_SAM3_VIDEO_LOCAL_CHECKPOINT"
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
SAM3_VIDEO_FIELD_TOLERANCES: dict[str, dict[str, float]] = {
    "frame_indices": {"atol": 0.0, "rtol": 0.0},
    "track_ids": {"atol": 0.0, "rtol": 0.0},
    "masks": {"atol": 0.0, "rtol": 0.0},
    "boxes": {"atol": 1.0e-4, "rtol": 1.0e-4},
    "scores": {"atol": 1.0e-4, "rtol": 1.0e-4},
    "tap.score_probs": {"atol": 1.0e-4, "rtol": 1.0e-4},
}
SAM3_VIDEO_SELECTED_TAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("score_probs", "score_probs"),
)
_SUPPORTED_LOCAL_FORMATS = {".npz", ".safetensors"}
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
    local_checkpoint_env: str = SAM3_VIDEO_LOCAL_CHECKPOINT_ENV
    local_checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    config_sha256: str | None = None
    blocked_reason: str | None = None
    blocker_kind: str | None = None
    admitted: bool = False
    comparison_report: dict[str, Any] | None = None

    @property
    def blocked(self) -> bool:
        return self.status.startswith("BLOCKED:")


class SAM3VideoReferenceDependencyError(RuntimeError):
    """Raised when the upstream SAM3 video reference runtime is unavailable."""


class SAM3VideoReferenceCaptureError(RuntimeError):
    """Raised when upstream SAM3 video capture cannot run or is malformed."""


class SAM3VideoLocalCaptureError(RuntimeError):
    """Raised when the local MLX SAM3 video capture cannot run."""


class SAM3VideoParityError(AssertionError):
    """Raised when SAM3 video upstream-vs-MLX comparison cannot be evaluated."""


@dataclass(frozen=True)
class FieldComparison:
    name: str
    reference_shape: list[int]
    local_shape: list[int]
    atol: float
    rtol: float
    max_abs_error: float | None
    max_rel_error: float | None
    passed: bool


@dataclass(frozen=True)
class SAM3VideoCapture:
    source: str
    frames: np.ndarray
    prompt: Any
    prompt_kind: str
    frame_indices: np.ndarray
    track_ids: np.ndarray
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    multiplex: dict[str, Any]
    taps: dict[str, np.ndarray]

    def inputs_for_local(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "prompt": self.prompt,
            "prompt_kind": self.prompt_kind,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "prompt_kind": self.prompt_kind,
            "frames_shape": list(np.asarray(self.frames).shape),
            "frame_indices": np.asarray(self.frame_indices).tolist(),
            "track_ids_shape": list(np.asarray(self.track_ids).shape),
            "masks_shape": list(np.asarray(self.masks).shape),
            "boxes_shape": list(np.asarray(self.boxes).shape),
            "scores_shape": list(np.asarray(self.scores).shape),
            "multiplex": self.multiplex,
            "tap_order": list(self.taps),
            "tap_shapes": {name: list(np.asarray(value).shape) for name, value in self.taps.items()},
        }


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
        local_checkpoint_path=admission.local_checkpoint_path,
        checkpoint_sha256=admission.checkpoint_sha256,
        config_sha256=admission.config_sha256,
        blocked_reason=reason,
        blocker_kind=blocker_kind,
    )


def _pass_from_admission(
    admission: SAM3VideoGateResult,
    report: dict[str, Any],
    *,
    local_checkpoint_path: Path,
) -> SAM3VideoGateResult:
    return SAM3VideoGateResult(
        status="UPSTREAM_PASSED",
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
        local_checkpoint_path=str(local_checkpoint_path),
        checkpoint_sha256=admission.checkpoint_sha256,
        config_sha256=admission.config_sha256,
        admitted=True,
        comparison_report=report,
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


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _default_capture_inputs() -> dict[str, Any]:
    _ensure_src_on_path()
    from mlx_cv.prompts import BoxPrompt

    frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    frames[1, 8:24, 8:24, :] = 64
    frames[2, 12:28, 12:28, :] = 128
    return {
        "frames": frames,
        "prompt": BoxPrompt([[4, 4, 20, 20]]),
        "prompt_kind": "box",
        "object_id": 1,
        "frame_index": 0,
    }


def _capture_inputs(inputs: Mapping[str, Any] | None) -> dict[str, Any]:
    out = _default_capture_inputs() if inputs is None else dict(inputs)
    if "frames" not in out:
        raise SAM3VideoParityError("SAM3 video comparison inputs are missing 'frames'")
    if "prompt" not in out:
        raise SAM3VideoParityError("SAM3 video comparison inputs are missing 'prompt'")
    out.setdefault("prompt_kind", "box")
    out.setdefault("object_id", 1)
    out.setdefault("frame_index", 0)
    return out


def _import_reference_builder(reference_path: Path = SAM3_VIDEO_REFERENCE_PATH):
    if not reference_path.exists():
        raise SAM3VideoReferenceDependencyError(f"SAM3 reference path is missing: {reference_path}")
    try:
        package_name = "sam3"
        package_root = reference_path / "sam3"
        if not package_root.exists():
            raise FileNotFoundError(package_root)
        package = sys.modules.get(package_name)
        if package is None or str(package_root.resolve()) not in list(getattr(package, "__path__", [])):
            import types

            package = types.ModuleType(package_name)
            package.__path__ = [str(package_root.resolve())]  # type: ignore[attr-defined]
            sys.modules[package_name] = package
        return importlib.import_module("sam3.model_builder")
    except Exception as exc:  # pragma: no cover - depends on optional reference env.
        raise SAM3VideoReferenceDependencyError(
            f"SAM3 video upstream reference modules could not be imported from {reference_path}: {exc}"
        ) from exc


def _prompt_box_xyxy(prompt: Any) -> np.ndarray | None:
    if hasattr(prompt, "boxes"):
        return np.asarray(prompt.boxes, dtype=np.float64).reshape(-1, 4)[0]
    if isinstance(prompt, Mapping):
        boxes = prompt.get("boxes", prompt.get("box"))
        if boxes is not None:
            return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[0]
    return None


def _box_xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float64)
    return np.asarray([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)


def _box_xyxy_to_normalized_xywh(box: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    h, w = image_size
    if h <= 0 or w <= 0:
        raise SAM3VideoReferenceCaptureError(f"SAM3 video upstream frame size must be positive, got {image_size}")
    out = _box_xyxy_to_xywh(box)
    out[[0, 2]] /= float(w)
    out[[1, 3]] /= float(h)
    return out


def _boxes_xywh_to_xyxy(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    h, w = image_size
    if boxes.size and np.nanmax(np.abs(boxes)) <= 1.5:
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def _mask_boxes(masks: np.ndarray) -> np.ndarray:
    boxes = []
    for mask in np.asarray(masks, dtype=np.bool_):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            boxes.append([0.0, 0.0, 1.0, 1.0])
        else:
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def _stable_multiplex_metadata(track_ids: np.ndarray) -> dict[str, Any]:
    ids = sorted({int(v) for v in np.asarray(track_ids).reshape(-1)})
    return {"active_object_ids": ids}


def _metadata_config(path: Path) -> dict[str, Any] | None:
    if path.suffix != ".npz":
        return None
    try:
        with np.load(path, allow_pickle=False) as weights:
            if "__config_json__" not in weights.files:
                return None
            raw = np.asarray(weights["__config_json__"]).item()
    except Exception:
        return None
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return None


def _tuple_fields(data: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    out = dict(data)
    for field in fields:
        if field in out and isinstance(out[field], list):
            out[field] = tuple(out[field])
    return out


def _local_video_config(weights_path: Path):
    _ensure_src_on_path()
    from mlx_cv.models.sam3 import (
        SAM3MultiplexDecoderConfig,
        SAM3VideoConfig,
        SAM3VideoMemoryConfig,
        SAM3VideoTrackerConfig,
    )

    metadata = _metadata_config(weights_path) or {}
    video_data = dict(metadata.get("video", metadata))
    if not video_data:
        return SAM3VideoConfig.tiny_fixture()
    tracker_data = _tuple_fields(video_data.get("tracker", {}), ("image_size", "feature_grid"))
    memory_data = _tuple_fields(video_data.get("memory", {}), ("image_size", "feature_grid"))
    decoder_data = _tuple_fields(video_data.get("decoder", {}), ("low_res_mask_size", "high_res_mask_size"))
    tracker = SAM3VideoTrackerConfig(**tracker_data)
    memory = SAM3VideoMemoryConfig(**memory_data) if memory_data else None
    decoder = SAM3MultiplexDecoderConfig(**decoder_data) if decoder_data else None
    return SAM3VideoConfig(tracker=tracker, memory=memory, decoder=decoder)


def _video_result_capture(
    *,
    source: str,
    frames: np.ndarray,
    prompt: Any,
    prompt_kind: str,
    video: Any,
) -> SAM3VideoCapture:
    frame_indices = np.asarray(video.frame_indices, dtype=np.int64)
    track_ids = np.stack([np.asarray(frame.tracks.ids, dtype=np.int64) for frame in video.frames], axis=0)
    masks = np.stack([np.asarray(frame.masks.data, dtype=np.bool_) for frame in video.frames], axis=0)
    boxes = np.stack([np.asarray(frame.detections.boxes, dtype=np.float64) for frame in video.frames], axis=0)
    scores = np.stack([np.asarray(frame.detections.scores, dtype=np.float32) for frame in video.frames], axis=0)
    taps = {"score_probs": scores}
    return SAM3VideoCapture(
        source=source,
        frames=np.asarray(frames),
        prompt=prompt,
        prompt_kind=prompt_kind,
        frame_indices=frame_indices,
        track_ids=track_ids,
        masks=masks,
        boxes=boxes,
        scores=scores,
        multiplex=_stable_multiplex_metadata(track_ids),
        taps=taps,
    )


def _reference_outputs_capture(
    *,
    source: str,
    frames: np.ndarray,
    prompt: Any,
    prompt_kind: str,
    outputs: Sequence[Mapping[str, Any]],
    object_id: int,
) -> SAM3VideoCapture:
    frame_indices = []
    track_ids = []
    masks = []
    boxes = []
    scores = []
    image_size = (int(frames.shape[1]), int(frames.shape[2]))
    for item in outputs:
        frame_indices.append(int(item["frame_index"]))
        out = item["outputs"]
        ids = np.asarray(out.get("out_obj_ids", [object_id]), dtype=np.int64).reshape(-1)
        frame_masks = np.asarray(out.get("out_binary_masks"), dtype=np.bool_)
        if frame_masks.ndim == 2:
            frame_masks = frame_masks[None, :, :]
        if frame_masks.ndim != 3:
            raise SAM3VideoReferenceCaptureError(
                f"SAM3 video upstream out_binary_masks must have shape (N,H,W), got {frame_masks.shape}"
            )
        raw_boxes = out.get("out_boxes_xywh")
        frame_boxes = _mask_boxes(frame_masks) if raw_boxes is None else _boxes_xywh_to_xyxy(raw_boxes, image_size)
        raw_scores = out.get("out_probs", out.get("scores"))
        frame_scores = (
            np.ones((len(ids),), dtype=np.float32)
            if raw_scores is None
            else np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        )
        track_ids.append(ids)
        masks.append(frame_masks)
        boxes.append(frame_boxes)
        scores.append(frame_scores)
    track_ids_arr = np.stack(track_ids, axis=0)
    scores_arr = np.stack(scores, axis=0)
    return SAM3VideoCapture(
        source=source,
        frames=np.asarray(frames),
        prompt=prompt,
        prompt_kind=prompt_kind,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        track_ids=track_ids_arr,
        masks=np.stack(masks, axis=0),
        boxes=np.stack(boxes, axis=0),
        scores=scores_arr,
        multiplex=_stable_multiplex_metadata(track_ids_arr),
        taps={"score_probs": scores_arr},
    )


def capture_sam3_video_upstream_reference(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    reference_path: Path = SAM3_VIDEO_REFERENCE_PATH,
    inputs: Mapping[str, Any] | None = None,
) -> SAM3VideoCapture:
    """Run the upstream Torch SAM3 video reference on deterministic box input."""

    if not reference_path.exists():
        raise SAM3VideoReferenceDependencyError(f"SAM3 reference path is missing: {reference_path}")
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on optional reference env.
        raise SAM3VideoReferenceDependencyError(f"SAM3 video upstream reference capture requires PIL: {exc}") from exc
    try:
        builder = _import_reference_builder(reference_path)
    except SAM3VideoReferenceDependencyError:
        raise
    except Exception as exc:  # pragma: no cover - depends on optional reference env.
        raise SAM3VideoReferenceDependencyError(
            f"SAM3 video upstream reference modules could not be imported from {reference_path}: {exc}"
        ) from exc

    capture_inputs = _capture_inputs(inputs)
    frames = np.asarray(capture_inputs["frames"], dtype=np.uint8)
    if frames.ndim != 4:
        raise SAM3VideoReferenceCaptureError(f"SAM3 video upstream frames must be THWC, got {frames.shape}")
    prompt = capture_inputs["prompt"]
    box = _prompt_box_xyxy(prompt)
    if box is None:
        raise SAM3VideoReferenceCaptureError(
            "SAM3 video upstream comparison currently supports box prompts for tracker propagation"
        )
    obj_id = int(capture_inputs["object_id"])
    frame_index = int(capture_inputs["frame_index"])
    image_size = (int(frames.shape[1]), int(frames.shape[2]))

    try:
        with tempfile.TemporaryDirectory(prefix="mlx-cv-sam3-video-") as tmp:
            frame_dir = Path(tmp)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(frame_dir / f"{i:05d}.png")
            predictor = builder.build_sam3_multiplex_video_predictor(
                checkpoint_path=str(checkpoint_path),
                async_loading_frames=False,
                compile=False,
                warm_up=False,
            )
            session = predictor.start_session(resource_path=str(frame_dir), session_id="mlx-cv-sam3-video")
            session_id = session.get("session_id", "mlx-cv-sam3-video") if isinstance(session, dict) else "mlx-cv-sam3-video"
            predictor.add_prompt(
                session_id=session_id,
                frame_idx=frame_index,
                bounding_boxes=_box_xyxy_to_normalized_xywh(box, image_size)[None, :],
                bounding_box_labels=np.ones((1,), dtype=np.int64),
                obj_id=obj_id,
                rel_coordinates=True,
            )
            outputs = list(
                predictor.propagate_in_video(
                    session_id=session_id,
                    propagation_direction="forward",
                    start_frame_idx=frame_index,
                    max_frame_num_to_track=len(frames),
                )
            )
    except Exception as exc:  # pragma: no cover - requires real upstream runtime/checkpoint.
        raise SAM3VideoReferenceCaptureError(f"SAM3 video upstream reference capture failed: {exc}") from exc

    return _reference_outputs_capture(
        source="upstream_reference",
        frames=frames,
        prompt=prompt,
        prompt_kind=str(capture_inputs["prompt_kind"]),
        outputs=outputs,
        object_id=obj_id,
    )


def capture_sam3_video_local(
    local_checkpoint_path: str | Path,
    *,
    inputs: Mapping[str, Any] | None = None,
) -> SAM3VideoCapture:
    """Run the local MLX SAM3 video path from a converted .npz or safetensors checkpoint."""

    path = Path(local_checkpoint_path)
    if not path.is_file() or path.suffix not in _SUPPORTED_LOCAL_FORMATS:
        raise SAM3VideoLocalCaptureError(
            "SAM3 video local MLX capture requires a converted local .npz or safetensors checkpoint; "
            f"got {path}."
        )
    _ensure_src_on_path()
    try:
        from mlx_cv.models.sam3 import SAM3VideoModel, SAM3VideoSessionManager, load_sam3_video_weights
    except Exception as exc:  # pragma: no cover - depends on local MLX runtime.
        raise SAM3VideoLocalCaptureError(f"SAM3 video local MLX capture requires mlx-cv runtime imports: {exc}") from exc

    capture_inputs = _capture_inputs(inputs)
    try:
        cfg = _local_video_config(path)
        model = load_sam3_video_weights(SAM3VideoModel(cfg), path)
        manager = SAM3VideoSessionManager(model=model, multiplex_bucket_capacity=cfg.tracker.multiplex_count)
        state = manager.start_session(frames=capture_inputs["frames"])
        manager.add_prompt(
            state.session_id,
            frame_index=int(capture_inputs["frame_index"]),
            prompt=capture_inputs["prompt"],
            object_id=int(capture_inputs["object_id"]),
        )
        video = manager.propagate_in_video(state.session_id)
    except Exception as exc:
        raise SAM3VideoLocalCaptureError(f"SAM3 video local MLX capture failed: {exc}") from exc

    return _video_result_capture(
        source="mlx_local",
        frames=np.asarray(capture_inputs["frames"]),
        prompt=capture_inputs["prompt"],
        prompt_kind=str(capture_inputs["prompt_kind"]),
        video=video,
    )


def _max_rel_error(got: np.ndarray, expected: np.ndarray) -> float:
    denom = np.maximum(np.abs(expected), 1.0e-8)
    if got.size == 0:
        return 0.0
    return float(np.max(np.abs(got - expected) / denom))


def _compare_array(name: str, reference: Any, local: Any, tolerances: Mapping[str, float]) -> FieldComparison:
    ref = _np(reference)
    got = _np(local)
    atol = float(tolerances["atol"])
    rtol = float(tolerances["rtol"])
    same_shape = got.shape == ref.shape
    finite = bool(np.all(np.isfinite(got)) and np.all(np.isfinite(ref))) if same_shape else False
    max_abs = None
    max_rel = None
    passed = False
    if same_shape and finite:
        ref64 = ref.astype(np.float64, copy=False)
        got64 = got.astype(np.float64, copy=False)
        diff = np.abs(got64 - ref64)
        max_abs = float(np.max(diff)) if diff.size else 0.0
        max_rel = _max_rel_error(got64, ref64)
        passed = bool(np.all(diff <= (atol + rtol * np.abs(ref64))))
    return FieldComparison(
        name=name,
        reference_shape=list(ref.shape),
        local_shape=list(got.shape),
        atol=atol,
        rtol=rtol,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        passed=passed,
    )


def _capture_summary(capture: Any) -> dict[str, Any]:
    summary = getattr(capture, "summary", None)
    if callable(summary):
        return dict(summary())
    return {
        "source": getattr(capture, "source", type(capture).__name__),
        "frame_indices": np.asarray(capture.frame_indices).tolist(),
        "track_ids_shape": list(np.asarray(capture.track_ids).shape),
        "masks_shape": list(np.asarray(capture.masks).shape),
        "boxes_shape": list(np.asarray(capture.boxes).shape),
        "scores_shape": list(np.asarray(capture.scores).shape),
        "tap_order": list(getattr(capture, "taps", {})),
    }


def compare_sam3_video_captures(
    reference: Any,
    local: Any,
    *,
    selected_tap_pairs: Sequence[tuple[str, str]] = SAM3_VIDEO_SELECTED_TAP_PAIRS,
    tolerances: Mapping[str, Mapping[str, float]] = SAM3_VIDEO_FIELD_TOLERANCES,
) -> dict[str, Any]:
    """Compare SAM3 video public outputs, object ids, scores, and stable taps."""

    comparisons = [
        _compare_array("frame_indices", reference.frame_indices, local.frame_indices, tolerances["frame_indices"]),
        _compare_array("track_ids", reference.track_ids, local.track_ids, tolerances["track_ids"]),
        _compare_array("masks", reference.masks, local.masks, tolerances["masks"]),
        _compare_array("boxes", reference.boxes, local.boxes, tolerances["boxes"]),
        _compare_array("scores", reference.scores, local.scores, tolerances["scores"]),
    ]
    for reference_key, local_key in selected_tap_pairs:
        if reference_key not in reference.taps:
            raise SAM3VideoParityError(f"SAM3 video upstream capture missing selected tap {reference_key!r}")
        if local_key not in local.taps:
            raise SAM3VideoParityError(f"SAM3 video local capture missing selected tap {local_key!r}")
        field_name = f"tap.{reference_key}"
        comparisons.append(
            _compare_array(field_name, reference.taps[reference_key], local.taps[local_key], tolerances[field_name])
        )

    fields = [asdict(item) for item in comparisons]
    multiplex_equal = reference.multiplex == local.multiplex
    return {
        "passed": all(item["passed"] for item in fields) and multiplex_equal,
        "tolerances": {key: dict(value) for key, value in tolerances.items()},
        "selected_tap_pairs": [
            {"reference": reference_key, "local": local_key}
            for reference_key, local_key in selected_tap_pairs
        ],
        "fields": fields,
        "metadata": {"multiplex_equal": multiplex_equal},
        "reference_summary": _capture_summary(reference),
        "local_summary": _capture_summary(local),
    }


def _failure_summary(report: Mapping[str, Any]) -> str:
    failed = [field for field in report["fields"] if not field["passed"]]
    if failed:
        first = failed[0]
        return (
            f"{first['name']} max_abs={first['max_abs_error']} max_rel={first['max_rel_error']} "
            f"tol=({first['atol']},{first['rtol']})"
        )
    if not report.get("metadata", {}).get("multiplex_equal", True):
        return "Object Multiplex metadata mismatch"
    return "unknown comparison failure"


def _reference_inputs(capture: Any) -> Mapping[str, Any] | None:
    method = getattr(capture, "inputs_for_local", None)
    if callable(method):
        return method()
    if hasattr(capture, "frames") and hasattr(capture, "prompt"):
        return {
            "frames": np.asarray(capture.frames),
            "prompt": capture.prompt,
            "prompt_kind": getattr(capture, "prompt_kind", "box"),
        }
    return None


def _resolve_local_checkpoint(env: Mapping[str, str]) -> Path:
    local = env.get(SAM3_VIDEO_LOCAL_CHECKPOINT_ENV)
    if not local:
        raise SAM3VideoLocalCaptureError(
            f"{SAM3_VIDEO_LOCAL_CHECKPOINT_ENV} is unset; set it to a converted local MLX .npz or "
            "safetensors SAM3 video checkpoint"
        )
    path = Path(local)
    if not path.exists():
        raise SAM3VideoLocalCaptureError(
            f"{SAM3_VIDEO_LOCAL_CHECKPOINT_ENV} does not point to an existing path: {path}"
        )
    if not path.is_file():
        raise SAM3VideoLocalCaptureError(f"{SAM3_VIDEO_LOCAL_CHECKPOINT_ENV} does not point to a file: {path}")
    if path.suffix not in _SUPPORTED_LOCAL_FORMATS:
        raise SAM3VideoLocalCaptureError(
            f"{SAM3_VIDEO_LOCAL_CHECKPOINT_ENV} must point to a converted local .npz or safetensors checkpoint: {path}"
        )
    return path


def evaluate_sam3_video_comparison_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_checkpoint_bytes: int = 1_000_000,
    min_config_bytes: int = 2,
    check_reference_dependencies: bool = True,
    reference_capture_func: Callable[..., Any] | None = None,
    local_capture_func: Callable[..., Any] | None = None,
    compare_func: Callable[..., dict[str, Any]] | None = None,
) -> SAM3VideoGateResult:
    env = os.environ if environ is None else environ
    reference_capture_func = reference_capture_func or capture_sam3_video_upstream_reference
    local_capture_func = local_capture_func or capture_sam3_video_local
    compare_func = compare_func or compare_sam3_video_captures
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

    try:
        local_checkpoint = _resolve_local_checkpoint(env)
    except SAM3VideoLocalCaptureError as exc:
        return _block_from_admission(admission, str(exc), blocker_kind="local_checkpoint")

    if check_reference_dependencies:
        try:
            importlib.import_module("torch")
        except Exception as exc:
            return _block_from_admission(
                admission,
                f"SAM3 video upstream reference execution requires torch: {exc}",
                blocker_kind="reference_runtime",
            )

    assert admission.checkpoint_path is not None
    assert admission.config_path is not None
    try:
        reference = reference_capture_func(
            Path(admission.checkpoint_path),
            config_path=Path(admission.config_path),
            reference_path=SAM3_VIDEO_REFERENCE_PATH,
        )
    except SAM3VideoReferenceDependencyError as exc:
        return _block_from_admission(admission, str(exc), blocker_kind="reference_runtime")
    except SAM3VideoReferenceCaptureError as exc:
        return _block_from_admission(
            admission,
            f"SAM3 video upstream reference capture failed: {exc}",
            blocker_kind="reference_capture",
        )
    except Exception as exc:
        return _block_from_admission(
            admission,
            f"SAM3 video upstream reference capture failed: {exc}",
            blocker_kind="reference_capture",
        )

    try:
        local = local_capture_func(local_checkpoint, inputs=_reference_inputs(reference))
    except SAM3VideoLocalCaptureError as exc:
        return _block_from_admission(admission, str(exc), blocker_kind="local_capture")
    except Exception as exc:
        return _block_from_admission(
            admission,
            f"SAM3 video local MLX capture failed: {exc}",
            blocker_kind="local_capture",
        )

    try:
        report = compare_func(reference, local)
    except SAM3VideoParityError as exc:
        return _block_from_admission(
            admission,
            f"SAM3 video comparison component unavailable: {exc}",
            blocker_kind="local_comparison",
        )
    except Exception as exc:
        return _block_from_admission(
            admission,
            f"SAM3 video comparison failed: {exc}",
            blocker_kind="local_comparison",
        )
    if not report["passed"]:
        return _block_from_admission(
            admission,
            f"SAM3 video upstream-vs-MLX parity drift: {_failure_summary(report)}",
            blocker_kind="parity_drift",
        )
    return _pass_from_admission(admission, report, local_checkpoint_path=local_checkpoint)


def status_dict(result: SAM3VideoGateResult) -> dict:
    out = asdict(result)
    out["schema_version"] = 1
    out["phase"] = "sam3-video-real-checkpoint-admission"
    out["model"] = "sam3_video"
    out["display_name"] = "SAM 3.1 Video / Object Multiplex"
    if result.blocked:
        claim_level = "external_blocker"
    elif result.status == "UPSTREAM_PASSED" and result.comparison_report is not None:
        claim_level = "upstream_passed"
    elif result.admitted:
        claim_level = "checkpoint_admitted"
    else:
        claim_level = "unknown"
    out["claim_level"] = claim_level
    out["blocker_kind"] = result.blocker_kind
    out["official_model_id"] = SAM3_VIDEO_OFFICIAL_MODEL_ID
    out["checkpoint_name"] = SAM3_VIDEO_CHECKPOINT_NAME
    out["config_name"] = SAM3_VIDEO_CONFIG_NAME
    out["source_url"] = SAM3_VIDEO_SOURCE_URL
    out["license_or_terms"] = SAM3_VIDEO_LICENSE_OR_TERMS
    out["provenance_status"] = "cached" if result.admitted else "not_cached"
    out["reference_surfaces"] = list(SAM3_VIDEO_REFERENCE_SURFACES)
    out["comparison_scope"] = (
        "SAM 3.1 video/Object Multiplex masks, boxes, track IDs, object scores, and multiplex metadata"
    )
    out["local_checkpoint_env"] = SAM3_VIDEO_LOCAL_CHECKPOINT_ENV
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
    result = evaluate_sam3_video_comparison_gate()
    write_status(result)
    print(json.dumps(status_dict(result), indent=2))
    return 0 if not result.blocked else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
