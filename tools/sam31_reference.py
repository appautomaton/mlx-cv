"""Official SAM 3.1 checkpoint admission and reference-capture helpers.

The official repository under ``references/sam3`` and the merged
``sam3.1_multiplex.pt`` checkpoint are the only upstream sources used here.
This module deliberately has no import-time PyTorch dependency so structural
tests and stored-capture comparisons remain available in the normal test
environment.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


SAM31_REFERENCE_ROOT = Path("references/sam3")
SAM31_CHECKPOINT_PATH = Path("models/sam3-video/upstream/sam3.1_multiplex.pt")
SAM31_CONFIG_PATH = Path("models/sam3-video/upstream/config.json")
SAM31_TENSOR_COUNT = 1623
SAM31_DETECTOR_TENSOR_COUNT = 1166
SAM31_TRACKER_TENSOR_COUNT = 457
SAM31_FLOAT32_TENSOR_COUNT = 1591
SAM31_COMPLEX64_TENSOR_COUNT = 32

SAM31_REQUIRED_SURFACES: dict[str, tuple[str, str]] = {
    "build_sam3_image_model": ("sam3/model_builder.py", "def build_sam3_image_model"),
    "build_sam3_multiplex_video_model": (
        "sam3/model_builder.py",
        "def build_sam3_multiplex_video_model",
    ),
    "build_sam3_multiplex_video_predictor": (
        "sam3/model_builder.py",
        "def build_sam3_multiplex_video_predictor",
    ),
    "MultiplexController": ("sam3/model_builder.py", "MultiplexController"),
    "SimpleMaskEncoder": ("sam3/model_builder.py", "SimpleMaskEncoder"),
    "start_session": ("sam3/model/sam3_base_predictor.py", "def start_session"),
    "add_prompt": ("sam3/model/sam3_base_predictor.py", "def add_prompt"),
    "propagate_in_video": (
        "sam3/model/sam3_base_predictor.py",
        "def propagate_in_video",
    ),
}

_ROPE_KEY = re.compile(
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn\.freqs_cis$"
)


class SAM31ContractError(ValueError):
    """Raised when a checkpoint or official-source contract is not SAM 3.1."""


class SAM31ReferenceDependencyError(RuntimeError):
    """Raised when the optional official PyTorch reference cannot be imported."""


@dataclass(frozen=True)
class SAM31CheckpointInventory:
    tensor_count: int
    detector_tensor_count: int
    tracker_tensor_count: int
    float32_tensor_count: int
    complex64_tensor_count: int
    complex_rope_blocks: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["complex_rope_blocks"] = list(self.complex_rope_blocks)
        return value


@dataclass(frozen=True)
class SAM31Admission:
    status: str
    checkpoint_path: str
    config_path: str
    reference_root: str
    checkpoint_sha256: str | None = None
    config_sha256: str | None = None
    inventory: SAM31CheckpointInventory | None = None
    blocked_reason: str | None = None

    @property
    def admitted(self) -> bool:
        return self.status == "ADMITTED"


@dataclass(frozen=True)
class SAM31ReferenceCapture:
    kind: str
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    taps: dict[str, np.ndarray]
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_state_dict(state: Mapping[str, Any]) -> Mapping[str, Any]:
    wrapped = state.get("model")
    if isinstance(wrapped, Mapping):
        return wrapped
    return state


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise SAM31ContractError("checkpoint contains a non-tensor value")
    return str(dtype).rsplit(".", 1)[-1]


def inspect_sam31_state_dict(
    state: Mapping[str, Any], *, require_exact: bool = True
) -> SAM31CheckpointInventory:
    """Inspect a loaded checkpoint without converting or copying tensor data."""

    tensors = _unwrap_state_dict(state)
    if not tensors:
        raise SAM31ContractError("SAM 3.1 checkpoint is empty")
    if not all(isinstance(key, str) for key in tensors):
        raise SAM31ContractError("SAM 3.1 checkpoint keys must be strings")

    detector = [key for key in tensors if key.startswith("detector.")]
    tracker = [key for key in tensors if key.startswith("tracker.")]
    unexpected = sorted(set(tensors) - set(detector) - set(tracker))
    if unexpected:
        raise SAM31ContractError(
            f"unsupported SAM 3.1 top-level keys: {unexpected[:5]}"
        )

    dtype_counts: dict[str, int] = {}
    rope_blocks: list[int] = []
    for key, value in tensors.items():
        dtype_name = _dtype_name(value)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
        if dtype_name == "complex64":
            match = _ROPE_KEY.match(key)
            if match is None:
                raise SAM31ContractError(
                    f"complex64 tensor is not an official RoPE table: {key}"
                )
            rope_blocks.append(int(match.group(1)))

    inventory = SAM31CheckpointInventory(
        tensor_count=len(tensors),
        detector_tensor_count=len(detector),
        tracker_tensor_count=len(tracker),
        float32_tensor_count=dtype_counts.get("float32", 0),
        complex64_tensor_count=dtype_counts.get("complex64", 0),
        complex_rope_blocks=tuple(sorted(rope_blocks)),
    )
    if require_exact:
        expected = SAM31CheckpointInventory(
            tensor_count=SAM31_TENSOR_COUNT,
            detector_tensor_count=SAM31_DETECTOR_TENSOR_COUNT,
            tracker_tensor_count=SAM31_TRACKER_TENSOR_COUNT,
            float32_tensor_count=SAM31_FLOAT32_TENSOR_COUNT,
            complex64_tensor_count=SAM31_COMPLEX64_TENSOR_COUNT,
            complex_rope_blocks=tuple(range(32)),
        )
        if inventory != expected:
            raise SAM31ContractError(
                "SAM 3.1 checkpoint inventory mismatch: "
                f"expected={expected.to_dict()} actual={inventory.to_dict()}"
            )
    return inventory


def load_sam31_state_dict(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise SAM31ReferenceDependencyError(
            "official SAM 3.1 checkpoint inspection requires PyTorch"
        ) from exc

    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:  # older PyTorch without mmap support
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise SAM31ContractError("SAM 3.1 checkpoint root must be a mapping")
    return _unwrap_state_dict(state)


def verify_official_reference_surfaces(root: Path = SAM31_REFERENCE_ROOT) -> None:
    if not root.is_dir():
        raise SAM31ContractError(f"official SAM 3.1 source is missing: {root}")
    for name, (relative_path, marker) in SAM31_REQUIRED_SURFACES.items():
        path = root / relative_path
        if not path.is_file():
            raise SAM31ContractError(
                f"official SAM 3.1 reference surface {name} is missing file {path}"
            )
        if marker not in path.read_text():
            raise SAM31ContractError(
                f"official SAM 3.1 reference surface {name} is missing marker {marker!r}"
            )


def admit_sam31_reference(
    checkpoint_path: Path = SAM31_CHECKPOINT_PATH,
    config_path: Path = SAM31_CONFIG_PATH,
    reference_root: Path = SAM31_REFERENCE_ROOT,
    *,
    include_sha256: bool = True,
    state_loader: Callable[[Path], Mapping[str, Any]] = load_sam31_state_dict,
) -> SAM31Admission:
    """Admit the official source/checkpoint/config or return one precise blocker."""

    for label, path in (
        ("checkpoint", checkpoint_path),
        ("config", config_path),
        ("official source", reference_root),
    ):
        if not path.exists():
            return SAM31Admission(
                status=f"BLOCKED:missing {label}",
                checkpoint_path=str(checkpoint_path),
                config_path=str(config_path),
                reference_root=str(reference_root),
                blocked_reason=f"missing {label}: {path}",
            )

    try:
        verify_official_reference_surfaces(reference_root)
        config = json.loads(config_path.read_text())
        if config.get("architectures") != ["Sam3VideoModel"]:
            raise SAM31ContractError(
                "SAM 3.1 config must declare architectures=['Sam3VideoModel']"
            )
        inventory = inspect_sam31_state_dict(state_loader(checkpoint_path))
    except (OSError, json.JSONDecodeError, SAM31ContractError, SAM31ReferenceDependencyError) as exc:
        return SAM31Admission(
            status=f"BLOCKED:{exc}",
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            reference_root=str(reference_root),
            blocked_reason=str(exc),
        )

    return SAM31Admission(
        status="ADMITTED",
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        reference_root=str(reference_root),
        checkpoint_sha256=sha256_file(checkpoint_path) if include_sha256 else None,
        config_sha256=sha256_file(config_path) if include_sha256 else None,
        inventory=inventory,
    )


def write_reference_capture(path: Path, capture: SAM31ReferenceCapture) -> None:
    """Persist a small, pickle-free deterministic parity capture."""

    if capture.kind not in {"image", "video"}:
        raise SAM31ContractError(f"unsupported SAM 3.1 capture kind: {capture.kind!r}")
    arrays: dict[str, np.ndarray] = {}
    for group_name, group in (
        ("input", capture.inputs),
        ("output", capture.outputs),
        ("tap", capture.taps),
    ):
        for name, value in group.items():
            arrays[f"{group_name}.{name}"] = np.asarray(value)
    metadata = {"kind": capture.kind, **capture.metadata}
    arrays["__metadata_json__"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_reference_capture(path: Path) -> SAM31ReferenceCapture:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["__metadata_json__"]).decode("utf-8"))
        groups: dict[str, dict[str, np.ndarray]] = {
            "input": {},
            "output": {},
            "tap": {},
        }
        for key in archive.files:
            if key == "__metadata_json__":
                continue
            prefix, separator, name = key.partition(".")
            if not separator or prefix not in groups:
                raise SAM31ContractError(f"malformed capture key: {key}")
            groups[prefix][name] = archive[key]
    kind = metadata.pop("kind")
    return SAM31ReferenceCapture(
        kind=kind,
        inputs=groups["input"],
        outputs=groups["output"],
        taps=groups["tap"],
        metadata=metadata,
    )

