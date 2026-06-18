"""SAM 3 (transformers-native) upstream parity gate helpers — image + video.

Mirrors :mod:`tools.locateanything_upstream`: checkpoint admission, an upstream
reference capture, a local MLX capture, and a documented numeric comparison that
flips ``parity-status.json`` only on a *real* PASS (never a synthetic one).

The reference here is the pip-installable ``transformers`` ``Sam3Model`` /
``Sam3VideoModel`` (their state_dicts map 1:1 to the ``facebook/sam3``
``model.safetensors`` ``detector_model.*`` / ``tracker_model.*`` / ``tracker_neck.*``
namespaces), not the ``references/`` research repo. The research-repo harnesses
(``tools/sam3_image_upstream.py`` / ``tools/sam3_video_upstream.py``) stay as a
secondary path; this transformers gate is authoritative.

``torch`` / ``transformers`` are imported lazily inside reference capture only
(tools/tests, never ``src/mlx_cv``). Local MLX capture for the faithful detector
and tracker is wired incrementally as slices 2-11 of
``.agent/work/2026-06-18-sam3-real-architecture-port`` land; until then it returns
a precise "not yet ported" blocker rather than a fake pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent

REFERENCE_TRANSFORMERS_PIN = "transformers>=5.10,<6"
_MIN_TRANSFORMERS = (5, 10)
_MAX_TRANSFORMERS = (6, 0)

# Env names reuse the identifiers already recorded in parity-status.json so the
# transformers gate and the existing blockers describe the same checkpoints.
SAM3_IMAGE_CHECKPOINT_ENV = "MLX_CV_SAM3_IMAGE_CHECKPOINT"
SAM3_IMAGE_LOCAL_CHECKPOINT_ENV = "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT"
SAM3_IMAGE_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_SAM3_IMAGE_GATE"
SAM3_VIDEO_CHECKPOINT_ENV = "MLX_CV_SAM3_VIDEO_CHECKPOINT"
SAM3_VIDEO_LOCAL_CHECKPOINT_ENV = "MLX_CV_SAM3_VIDEO_LOCAL_CHECKPOINT"
SAM3_VIDEO_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_SAM3_VIDEO_GATE"

_ENV: dict[str, tuple[str, str, str]] = {
    "sam3_image": (SAM3_IMAGE_CHECKPOINT_ENV, SAM3_IMAGE_LOCAL_CHECKPOINT_ENV, SAM3_IMAGE_REQUIRED_GATE_ENV),
    "sam3_video": (SAM3_VIDEO_CHECKPOINT_ENV, SAM3_VIDEO_LOCAL_CHECKPOINT_ENV, SAM3_VIDEO_REQUIRED_GATE_ENV),
}
_DISPLAY = {
    "sam3_image": "SAM 3 image detector (transformers reference)",
    "sam3_video": "SAM 3 video tracker (transformers reference)",
}
_SCOPE = {
    "sam3_image": "vision-encoder feature-map tap (slice-1 reference spine); full Sam3Model parity lands at slice 7",
    "sam3_video": "detector vision feature-map tap (slice-1 reference spine); full Sam3VideoModel parity lands at slice 11",
}

# The vision feature-map tap allows small fp32 implementation noise across
# Torch and MLX, mirroring the LocateAnything embedding-tap tolerances.
SAM3_IMAGE_FIELD_TOLERANCES: dict[str, dict[str, float]] = {
    "tap.vision.last_hidden_state": {"atol": 1.0e-4, "rtol": 1.0e-4},
}
SAM3_IMAGE_SELECTED_TAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("vision.last_hidden_state", "vision.last_hidden_state"),
)
SAM3_VIDEO_FIELD_TOLERANCES: dict[str, dict[str, float]] = {
    "tap.vision.last_hidden_state": {"atol": 1.0e-4, "rtol": 1.0e-4},
}
SAM3_VIDEO_SELECTED_TAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("vision.last_hidden_state", "vision.last_hidden_state"),
)

__all__ = [
    "Sam3GateResult",
    "Sam3Capture",
    "FieldComparison",
    "Sam3ReferenceDependencyError",
    "Sam3ReferenceCaptureError",
    "Sam3LocalCaptureError",
    "Sam3ParityError",
    "required_image_gate_enabled",
    "required_video_gate_enabled",
    "evaluate_sam3_image_gate",
    "evaluate_sam3_video_gate",
    "status_dict",
    "capture_sam3_image_upstream_reference",
    "capture_sam3_video_upstream_reference",
    "capture_sam3_image_local",
    "capture_sam3_video_local",
    "compare_sam3_image_captures",
    "compare_sam3_video_captures",
    "evaluate_sam3_image_comparison_gate",
    "evaluate_sam3_video_comparison_gate",
]


@dataclass(frozen=True)
class Sam3GateResult:
    model: str
    status: str
    checkpoint_env: str
    local_checkpoint_env: str
    required_gate_env: str
    reference_pin: str
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    blocked_reason: str | None = None
    admitted: bool = False
    comparison_report: dict[str, Any] | None = None

    @property
    def blocked(self) -> bool:
        return self.status.startswith("BLOCKED:")


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
class Sam3Capture:
    source: str
    pixel_values: np.ndarray
    taps: dict[str, np.ndarray]

    def inputs_for_local(self) -> dict[str, np.ndarray]:
        return {"pixel_values": self.pixel_values}

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pixel_values_shape": list(np.asarray(self.pixel_values).shape),
            "tap_order": list(self.taps),
            "tap_shapes": {name: list(np.asarray(value).shape) for name, value in self.taps.items()},
        }


class Sam3ReferenceDependencyError(RuntimeError):
    """Raised when the transformers SAM3 reference runtime is unavailable."""


class Sam3ReferenceCaptureError(RuntimeError):
    """Raised when the transformers SAM3 reference capture cannot run."""


class Sam3LocalCaptureError(RuntimeError):
    """Raised when the local MLX SAM3 capture cannot run (incl. not-yet-ported)."""


class Sam3ParityError(AssertionError):
    """Raised when a SAM3 upstream-vs-MLX comparison cannot be evaluated."""


# --- environment / admission --------------------------------------------------


def required_image_gate_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SAM3_IMAGE_REQUIRED_GATE_ENV) == "1"


def required_video_gate_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SAM3_VIDEO_REQUIRED_GATE_ENV) == "1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _block(model: str, reason: str, *, environ: Mapping[str, str]) -> Sam3GateResult:
    checkpoint_env, local_env, required_env = _ENV[model]
    return Sam3GateResult(
        model=model,
        status=f"BLOCKED:{reason}",
        checkpoint_env=checkpoint_env,
        local_checkpoint_env=local_env,
        required_gate_env=required_env,
        reference_pin=REFERENCE_TRANSFORMERS_PIN,
        checkpoint_path=environ.get(checkpoint_env),
        blocked_reason=reason,
    )


def _admit(model: str, path: Path, *, environ: Mapping[str, str], sha256: str | None = None) -> Sam3GateResult:
    checkpoint_env, local_env, required_env = _ENV[model]
    return Sam3GateResult(
        model=model,
        status="ADMITTED",
        checkpoint_env=checkpoint_env,
        local_checkpoint_env=local_env,
        required_gate_env=required_env,
        reference_pin=REFERENCE_TRANSFORMERS_PIN,
        checkpoint_path=str(path),
        checkpoint_sha256=sha256,
        admitted=True,
    )


def _block_from_admission(admission: Sam3GateResult, reason: str) -> Sam3GateResult:
    return Sam3GateResult(
        model=admission.model,
        status=f"BLOCKED:{reason}",
        checkpoint_env=admission.checkpoint_env,
        local_checkpoint_env=admission.local_checkpoint_env,
        required_gate_env=admission.required_gate_env,
        reference_pin=admission.reference_pin,
        checkpoint_path=admission.checkpoint_path,
        checkpoint_sha256=admission.checkpoint_sha256,
        blocked_reason=reason,
        admitted=admission.admitted,
    )


def _pass_from_admission(admission: Sam3GateResult, report: dict[str, Any]) -> Sam3GateResult:
    return Sam3GateResult(
        model=admission.model,
        status="UPSTREAM_PASSED",
        checkpoint_env=admission.checkpoint_env,
        local_checkpoint_env=admission.local_checkpoint_env,
        required_gate_env=admission.required_gate_env,
        reference_pin=admission.reference_pin,
        checkpoint_path=admission.checkpoint_path,
        checkpoint_sha256=admission.checkpoint_sha256,
        admitted=True,
        comparison_report=report,
    )


def _index_shards(index_path: Path) -> list[str] | None:
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return None
    shards = sorted(set(index.get("weight_map", {}).values()))
    return shards or None


def _admit_checkpoint(
    model: str,
    *,
    environ: Mapping[str, str],
    min_shard_bytes: int,
    allow_torch_pickle: bool,
) -> Sam3GateResult:
    checkpoint_env = _ENV[model][0]
    checkpoint = environ.get(checkpoint_env)
    if not checkpoint:
        return _block(model, f"{checkpoint_env} is unset", environ=environ)

    path = Path(checkpoint)
    if not path.exists():
        return _block(model, f"{checkpoint_env} does not point to an existing path: {path}", environ=environ)

    if path.is_file():
        suffixes = {".safetensors"} | ({".pt", ".bin"} if allow_torch_pickle else set())
        if path.suffix not in suffixes:
            return _block(model, f"unsupported SAM3 checkpoint format: {path.suffix or path.name}", environ=environ)
        if path.stat().st_size < min_shard_bytes:
            return _block(model, f"{path} is not a usable SAM3 checkpoint file", environ=environ)
        return _admit(model, path, environ=environ)

    if not path.is_dir():
        return _block(model, f"{checkpoint_env} is neither a file nor a directory: {path}", environ=environ)

    if not (path / "config.json").exists():
        return _block(model, f"SAM3 checkpoint directory is missing config.json: {path}", environ=environ)

    index_path = path / "model.safetensors.index.json"
    single = path / "model.safetensors"
    if index_path.exists():
        shards = _index_shards(index_path)
        if not shards:
            return _block(model, f"SAM3 safetensors index has no weight_map entries: {index_path}", environ=environ)
        missing = [shard for shard in shards if not (path / shard).exists()]
        if missing:
            return _block(model, f"SAM3 checkpoint directory is missing shard(s): {', '.join(missing[:3])}", environ=environ)
        stubs = [shard for shard in shards if (path / shard).stat().st_size < min_shard_bytes]
        if stubs:
            return _block(model, f"SAM3 checkpoint shard(s) are LFS stubs or incomplete: {', '.join(stubs[:3])}", environ=environ)
        return _admit(model, path, environ=environ)

    if single.exists():
        if single.stat().st_size < min_shard_bytes:
            return _block(model, f"SAM3 checkpoint model.safetensors is an LFS stub or incomplete: {single}", environ=environ)
        return _admit(model, path, environ=environ)

    if allow_torch_pickle:
        pts = sorted(path.glob("*.pt"))
        if pts:
            if pts[0].stat().st_size < min_shard_bytes:
                return _block(model, f"SAM3 checkpoint {pts[0].name} is an incomplete stub: {pts[0]}", environ=environ)
            return _admit(model, path, environ=environ)

    return _block(model, f"SAM3 checkpoint directory is missing model.safetensors[.index.json]: {path}", environ=environ)


def evaluate_sam3_image_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_shard_bytes: int = 1_000_000,
) -> Sam3GateResult:
    env = os.environ if environ is None else environ
    return _admit_checkpoint("sam3_image", environ=env, min_shard_bytes=min_shard_bytes, allow_torch_pickle=False)


def evaluate_sam3_video_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_shard_bytes: int = 1_000_000,
) -> Sam3GateResult:
    # The multiplex video checkpoint ships as a torch pickle (sam3.1_multiplex.pt),
    # so video admission accepts .pt in addition to safetensors.
    env = os.environ if environ is None else environ
    return _admit_checkpoint("sam3_video", environ=env, min_shard_bytes=min_shard_bytes, allow_torch_pickle=True)


def status_dict(result: Sam3GateResult) -> dict[str, Any]:
    out = asdict(result)
    out["display_name"] = _DISPLAY[result.model]
    if result.blocked:
        claim_level = "external_blocker"
    elif result.status == "UPSTREAM_PASSED" and result.comparison_report is not None:
        claim_level = "upstream_passed"
    elif result.admitted:
        claim_level = "checkpoint_admitted"
    else:
        claim_level = "unknown"
    out["claim_level"] = claim_level
    out["comparison_scope"] = _SCOPE[result.model]
    return out


# --- transformers reference capture -------------------------------------------


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def _ensure_src_on_path() -> None:
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _check_transformers_version(transformers: Any) -> None:
    version = str(getattr(transformers, "__version__", "0"))
    parts: list[int] = []
    for piece in version.split(".")[:2]:
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    minor_major = tuple(parts) if len(parts) == 2 else (0, 0)
    if not (_MIN_TRANSFORMERS <= minor_major < _MAX_TRANSFORMERS):
        raise Sam3ReferenceDependencyError(
            f"SAM3 upstream reference requires {REFERENCE_TRANSFORMERS_PIN}; found transformers {version}"
        )


def _import_transformers() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover - depends on optional reference env.
        raise Sam3ReferenceDependencyError(
            "SAM3 upstream reference capture requires torch and "
            f'{REFERENCE_TRANSFORMERS_PIN} (tools-only; run via `uv run --with "{REFERENCE_TRANSFORMERS_PIN}" --with torch`).'
        ) from exc
    _check_transformers_version(transformers)
    return torch, transformers


def _image_size_from_config(config: Any) -> int:
    vision = getattr(config, "vision_config", None)
    for obj in (getattr(vision, "backbone_config", None), vision, config):
        size = getattr(obj, "image_size", None)
        if isinstance(size, int) and size > 0:
            return size
    return 1008


def _fixed_pixel_values(image_size: int, *, num_channels: int = 3) -> np.ndarray:
    """Deterministic, RNG-free pixel_values so both sides compare on the same input."""

    count = num_channels * image_size * image_size
    return np.linspace(-1.0, 1.0, num=count, dtype=np.float32).reshape(1, num_channels, image_size, image_size)


def _capture_vision_tap(torch: Any, detector: Any, capture_inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    pixel_values = torch.as_tensor(np.asarray(capture_inputs["pixel_values"]), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        vision = detector.get_vision_features(pixel_values=pixel_values)
    return {"vision.last_hidden_state": _np(vision.last_hidden_state)}


def _resolve_capture_inputs(config: Any, inputs: Mapping[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    if inputs is None:
        return {"pixel_values": _fixed_pixel_values(_image_size_from_config(config))}
    return {key: np.asarray(value) for key, value in inputs.items()}


def capture_sam3_image_upstream_reference(
    checkpoint_path: str | Path,
    *,
    inputs: Mapping[str, np.ndarray] | None = None,
) -> Sam3Capture:
    """Load the transformers ``Sam3Model`` and tap its vision encoder deterministically."""

    path = Path(checkpoint_path)
    torch, transformers = _import_transformers()
    if not path.is_dir():
        raise Sam3ReferenceCaptureError(
            "SAM3 image reference capture needs a HF checkpoint directory (config.json + "
            f"model.safetensors); got {path}"
        )
    try:
        model = transformers.Sam3Model.from_pretrained(str(path), torch_dtype=torch.float32)
        model = model.to("cpu").eval()
    except Exception as exc:  # pragma: no cover - requires the real checkpoint.
        raise Sam3ReferenceCaptureError(f"SAM3 image upstream model load failed: {exc}") from exc

    capture_inputs = _resolve_capture_inputs(model.config, inputs)
    try:
        taps = _capture_vision_tap(torch, model, capture_inputs)
    except Exception as exc:  # pragma: no cover - requires the real checkpoint.
        raise Sam3ReferenceCaptureError(f"SAM3 image vision-feature capture failed: {exc}") from exc
    return Sam3Capture(
        source="upstream_reference",
        pixel_values=np.asarray(capture_inputs["pixel_values"], dtype=np.float32),
        taps=taps,
    )


def capture_sam3_video_upstream_reference(
    checkpoint_path: str | Path,
    *,
    inputs: Mapping[str, np.ndarray] | None = None,
) -> Sam3Capture:
    """Load the transformers ``Sam3VideoModel`` and tap its detector vision encoder."""

    path = Path(checkpoint_path)
    torch, transformers = _import_transformers()
    if not path.is_dir():
        raise Sam3ReferenceCaptureError(
            "SAM3 video reference capture needs a HF checkpoint directory (config.json + weights); "
            f"got {path}"
        )
    try:
        model = transformers.Sam3VideoModel.from_pretrained(str(path), torch_dtype=torch.float32)
        model = model.to("cpu").eval()
    except Exception as exc:  # pragma: no cover - requires the real checkpoint.
        raise Sam3ReferenceCaptureError(f"SAM3 video upstream model load failed: {exc}") from exc

    detector = model.detector_model
    detector_config = getattr(model.config, "detector_config", model.config)
    capture_inputs = _resolve_capture_inputs(detector_config, inputs)
    try:
        taps = _capture_vision_tap(torch, detector, capture_inputs)
    except Exception as exc:  # pragma: no cover - requires the real checkpoint.
        raise Sam3ReferenceCaptureError(f"SAM3 video vision-feature capture failed: {exc}") from exc
    return Sam3Capture(
        source="upstream_reference",
        pixel_values=np.asarray(capture_inputs["pixel_values"], dtype=np.float32),
        taps=taps,
    )


# --- local MLX capture (wired incrementally as slices land) -------------------


def _validate_local_npz(path: Path, env_name: str) -> None:
    if not path.is_file() or path.suffix != ".npz":
        raise Sam3LocalCaptureError(
            f"{env_name} must point to a converted local MLX .npz weights file; got {path}"
        )


def capture_sam3_image_local(
    local_checkpoint_path: str | Path,
    *,
    inputs: Mapping[str, np.ndarray] | None = None,
) -> Sam3Capture:
    """Local MLX SAM3 image capture — honest not-yet-ported blocker until slice 2."""

    _validate_local_npz(Path(local_checkpoint_path), SAM3_IMAGE_LOCAL_CHECKPOINT_ENV)
    _ensure_src_on_path()
    raise Sam3LocalCaptureError(
        "faithful MLX SAM3 image detector is not yet ported; the windowed-RoPE ViT vision "
        "encoder lands in slice 2 of 2026-06-18-sam3-real-architecture-port. Local capture "
        "runs once the MLX Sam3 detector exists (no synthetic pass before then)."
    )


def capture_sam3_video_local(
    local_checkpoint_path: str | Path,
    *,
    inputs: Mapping[str, np.ndarray] | None = None,
) -> Sam3Capture:
    """Local MLX SAM3 video capture — honest not-yet-ported blocker until slices 8-11."""

    _validate_local_npz(Path(local_checkpoint_path), SAM3_VIDEO_LOCAL_CHECKPOINT_ENV)
    _ensure_src_on_path()
    raise Sam3LocalCaptureError(
        "faithful MLX SAM3 video tracker is not yet ported; the tracker neck/memory/mask "
        "modules land in slices 8-11 of 2026-06-18-sam3-real-architecture-port. Local capture "
        "runs once the MLX Sam3 tracker exists (no synthetic pass before then)."
    )


# --- comparison ---------------------------------------------------------------


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
        diff = np.abs(got.astype(np.float64, copy=False) - ref.astype(np.float64, copy=False))
        max_abs = float(np.max(diff)) if diff.size else 0.0
        max_rel = _max_rel_error(got.astype(np.float64, copy=False), ref.astype(np.float64, copy=False))
        passed = bool(np.all(diff <= (atol + rtol * np.abs(ref))))
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
    return {"source": getattr(capture, "source", type(capture).__name__), "tap_order": list(getattr(capture, "taps", {}))}


def _compare_captures(
    reference: Any,
    local: Any,
    *,
    selected_tap_pairs: Sequence[tuple[str, str]],
    tolerances: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    comparisons: list[FieldComparison] = []
    for reference_key, local_key in selected_tap_pairs:
        if reference_key not in reference.taps:
            raise Sam3ParityError(f"SAM3 upstream capture missing selected tap {reference_key!r}")
        if local_key not in local.taps:
            raise Sam3ParityError(f"SAM3 local capture missing selected tap {local_key!r}")
        field_name = f"tap.{reference_key}"
        comparisons.append(
            _compare_array(field_name, reference.taps[reference_key], local.taps[local_key], tolerances[field_name])
        )

    fields = [asdict(item) for item in comparisons]
    return {
        "passed": all(item["passed"] for item in fields),
        "tolerances": {key: dict(value) for key, value in tolerances.items()},
        "selected_tap_pairs": [
            {"reference": reference_key, "local": local_key} for reference_key, local_key in selected_tap_pairs
        ],
        "fields": fields,
        "reference_summary": _capture_summary(reference),
        "local_summary": _capture_summary(local),
    }


def compare_sam3_image_captures(
    reference: Any,
    local: Any,
    *,
    selected_tap_pairs: Sequence[tuple[str, str]] = SAM3_IMAGE_SELECTED_TAP_PAIRS,
    tolerances: Mapping[str, Mapping[str, float]] = SAM3_IMAGE_FIELD_TOLERANCES,
) -> dict[str, Any]:
    return _compare_captures(reference, local, selected_tap_pairs=selected_tap_pairs, tolerances=tolerances)


def compare_sam3_video_captures(
    reference: Any,
    local: Any,
    *,
    selected_tap_pairs: Sequence[tuple[str, str]] = SAM3_VIDEO_SELECTED_TAP_PAIRS,
    tolerances: Mapping[str, Mapping[str, float]] = SAM3_VIDEO_FIELD_TOLERANCES,
) -> dict[str, Any]:
    return _compare_captures(reference, local, selected_tap_pairs=selected_tap_pairs, tolerances=tolerances)


def _failure_summary(report: Mapping[str, Any]) -> str:
    failed = [field for field in report["fields"] if not field["passed"]]
    if not failed:
        return "unknown comparison failure"
    first = failed[0]
    return (
        f"{first['name']} max_abs={first['max_abs_error']} max_rel={first['max_rel_error']} "
        f"tol=({first['atol']},{first['rtol']})"
    )


def _reference_inputs(capture: Any) -> Mapping[str, np.ndarray] | None:
    method = getattr(capture, "inputs_for_local", None)
    if callable(method):
        return method()
    if hasattr(capture, "pixel_values"):
        return {"pixel_values": np.asarray(capture.pixel_values)}
    return None


def _resolve_local_checkpoint(model: str, env: Mapping[str, str]) -> Path:
    local_env = _ENV[model][1]
    local = env.get(local_env)
    if not local:
        raise Sam3LocalCaptureError(
            f"{local_env} is unset and the admitted checkpoint is not a local MLX .npz file; "
            "upstream weights are admitted for reference capture but are not a local MLX capture"
        )
    path = Path(local)
    if not path.exists():
        raise Sam3LocalCaptureError(f"{local_env} does not point to an existing path: {path}")
    if not path.is_file() or path.suffix != ".npz":
        raise Sam3LocalCaptureError(f"{local_env} must point to a converted local MLX .npz weights file: {path}")
    return path


def _evaluate_comparison_gate(
    model: str,
    *,
    environ: Mapping[str, str] | None,
    min_shard_bytes: int,
    check_reference_dependencies: bool,
    gate_func: Callable[..., Sam3GateResult],
    reference_capture_func: Callable[..., Any],
    local_capture_func: Callable[..., Any],
    compare_func: Callable[..., dict[str, Any]],
) -> Sam3GateResult:
    env = os.environ if environ is None else environ
    admission = gate_func(environ=env, min_shard_bytes=min_shard_bytes)
    if admission.blocked:
        return admission

    if check_reference_dependencies:
        try:
            _import_transformers()
        except Sam3ReferenceDependencyError as exc:
            return _block_from_admission(admission, str(exc))

    assert admission.checkpoint_path is not None
    try:
        local_checkpoint = _resolve_local_checkpoint(model, env)
    except Sam3LocalCaptureError as exc:
        return _block_from_admission(admission, str(exc))

    try:
        reference = reference_capture_func(Path(admission.checkpoint_path))
    except Sam3ReferenceDependencyError as exc:
        return _block_from_admission(admission, str(exc))
    except Sam3ReferenceCaptureError as exc:
        return _block_from_admission(admission, f"SAM3 upstream reference capture failed: {exc}")
    except Exception as exc:
        return _block_from_admission(admission, f"SAM3 upstream reference capture failed: {exc}")

    try:
        local = local_capture_func(local_checkpoint, inputs=_reference_inputs(reference))
    except Sam3LocalCaptureError as exc:
        return _block_from_admission(admission, str(exc))
    except Exception as exc:
        return _block_from_admission(admission, f"SAM3 local MLX capture failed: {exc}")

    try:
        report = compare_func(reference, local)
    except Sam3ParityError as exc:
        return _block_from_admission(admission, f"SAM3 comparison component unavailable: {exc}")
    except Exception as exc:
        return _block_from_admission(admission, f"SAM3 comparison failed: {exc}")

    if not report.get("passed", False):
        return _block_from_admission(
            admission, "SAM3 upstream-vs-MLX parity drift for selected taps: " + _failure_summary(report)
        )

    return _pass_from_admission(admission, report)


def evaluate_sam3_image_comparison_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_shard_bytes: int = 1_000_000,
    check_reference_dependencies: bool = True,
    reference_capture_func: Callable[..., Any] = capture_sam3_image_upstream_reference,
    local_capture_func: Callable[..., Any] = capture_sam3_image_local,
    compare_func: Callable[..., dict[str, Any]] = compare_sam3_image_captures,
) -> Sam3GateResult:
    """Full upstream-vs-MLX comparison gate for the SAM3 image detector.

    Admission is checked first; with a real HF checkpoint, the transformers
    reference, and a converted local MLX ``.npz`` available, this captures the
    vision feature-map tap on both sides and compares it within documented
    tolerances. Missing prerequisites return precise blockers, never a fake pass.
    """

    return _evaluate_comparison_gate(
        "sam3_image",
        environ=environ,
        min_shard_bytes=min_shard_bytes,
        check_reference_dependencies=check_reference_dependencies,
        gate_func=evaluate_sam3_image_gate,
        reference_capture_func=reference_capture_func,
        local_capture_func=local_capture_func,
        compare_func=compare_func,
    )


def evaluate_sam3_video_comparison_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_shard_bytes: int = 1_000_000,
    check_reference_dependencies: bool = True,
    reference_capture_func: Callable[..., Any] = capture_sam3_video_upstream_reference,
    local_capture_func: Callable[..., Any] = capture_sam3_video_local,
    compare_func: Callable[..., dict[str, Any]] = compare_sam3_video_captures,
) -> Sam3GateResult:
    """Full upstream-vs-MLX comparison gate for the SAM3 video tracker."""

    return _evaluate_comparison_gate(
        "sam3_video",
        environ=environ,
        min_shard_bytes=min_shard_bytes,
        check_reference_dependencies=check_reference_dependencies,
        gate_func=evaluate_sam3_video_gate,
        reference_capture_func=reference_capture_func,
        local_capture_func=local_capture_func,
        compare_func=compare_func,
    )
