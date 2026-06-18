"""Convert a verified DA3 safetensors checkpoint to local MLX ``.npz`` weights."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from da3_checkpoint import (
        DA3CheckpointError,
        DA3CheckpointInfo,
        default_cache_root,
        print_checkpoint_evidence,
        required_gate_enabled,
        resolve_da3_checkpoint,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported outside tools/.
    _CHECKPOINT_PATH = Path(__file__).with_name("da3_checkpoint.py")
    _SPEC = importlib.util.spec_from_file_location("da3_checkpoint", _CHECKPOINT_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _MODULE
    _SPEC.loader.exec_module(_MODULE)
    DA3CheckpointError = _MODULE.DA3CheckpointError
    DA3CheckpointInfo = _MODULE.DA3CheckpointInfo
    default_cache_root = _MODULE.default_cache_root
    print_checkpoint_evidence = _MODULE.print_checkpoint_evidence
    required_gate_enabled = _MODULE.required_gate_enabled
    resolve_da3_checkpoint = _MODULE.resolve_da3_checkpoint


DA3_CONVERTED_ENV = "MLX_CV_DA3_CONVERTED"
DA3_CONVERTED_FORMAT_VERSION = "v2"


class DA3ConversionError(RuntimeError):
    """Raised when a real DA3 checkpoint cannot be converted or resolved."""


class DA3ConversionDependencyError(DA3ConversionError):
    """Raised when optional conversion dependencies are unavailable."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_outside_repo(path: Path) -> None:
    resolved = path.expanduser().resolve(strict=False)
    root = _repo_root().resolve()
    if resolved == root or root in resolved.parents:
        raise DA3ConversionError(f"converted DA3 weights must stay outside git; {resolved} is under {root}")


def _load_safetensors(path: Path) -> dict[str, np.ndarray]:
    try:
        from safetensors.numpy import load_file
    except Exception as exc:  # pragma: no cover - depends on optional tool env.
        raise DA3ConversionDependencyError("DA3 checkpoint conversion requires safetensors.") from exc
    return {str(key): np.asarray(value) for key, value in load_file(str(path)).items()}


def _runtime_conversion_api():
    try:
        from mlx_cv.models.depth_anything_v3 import DA3MultiViewConfig, convert_da3_multiview_state_dict
    except Exception as exc:  # pragma: no cover - depends on optional MLX runtime availability.
        raise DA3ConversionDependencyError("DA3 checkpoint conversion requires the MLX runtime.") from exc
    return DA3MultiViewConfig, convert_da3_multiview_state_dict


def _model_slug(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower().replace("_", "-")


def default_converted_cache_path(checkpoint: DA3CheckpointInfo, *, cache_root: Path | None = None) -> Path:
    root = default_cache_root() if cache_root is None else cache_root.expanduser()
    return (
        root / "da3"
        / f"{_model_slug(checkpoint.model_id)}-{checkpoint.checkpoint_sha256[:12]}-mlx-{DA3_CONVERTED_FORMAT_VERSION}.npz"
    )


def config_from_checkpoint(checkpoint: DA3CheckpointInfo):
    """Build the local DA3 multiview config from the verified upstream config JSON."""

    import json

    DA3MultiViewConfig, _ = _runtime_conversion_api()
    raw = json.loads(checkpoint.config_path.read_text())
    config = raw.get("config", raw)
    return DA3MultiViewConfig.from_dict(config)


def convert_da3_checkpoint(
    checkpoint: DA3CheckpointInfo,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert a verified DA3 safetensors checkpoint to local MLX ``.npz`` weights."""

    output_path = output_path.expanduser()
    _ensure_outside_repo(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    _, convert_da3_multiview_state_dict = _runtime_conversion_api()
    converted = convert_da3_multiview_state_dict(_load_safetensors(checkpoint.checkpoint_path))
    arrays: dict[str, np.ndarray] = {}
    duplicates: list[str] = []
    for key, value in converted:
        if key in arrays:
            duplicates.append(key)
        arrays[key] = np.asarray(value)
    if duplicates:
        sample = ", ".join(repr(key) for key in duplicates[:5])
        more = "" if len(duplicates) <= 5 else f", and {len(duplicates) - 5} more"
        raise DA3ConversionError(f"duplicate converted DA3 keys: {sample}{more}")
    if not arrays:
        raise DA3ConversionError("DA3 conversion produced no tensors")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez(handle, **arrays)
        tmp.replace(output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output_path


def resolve_da3_converted_weights(
    *,
    environ: dict[str, str] | None = None,
    cache_root: Path | None = None,
    required: bool | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Resolve or create out-of-git converted DA3 multiview weights."""

    environ = os.environ if environ is None else environ
    required = required_gate_enabled(environ) if required is None else required

    env_path = environ.get(DA3_CONVERTED_ENV)
    if env_path:
        converted = Path(env_path).expanduser()
        if not converted.is_file():
            raise DA3ConversionError(f"{DA3_CONVERTED_ENV}={converted} is not a file")
        _ensure_outside_repo(converted)
        return converted

    checkpoint = resolve_da3_checkpoint(
        environ=environ,
        cache_root=cache_root,
        required=required,
    )
    if checkpoint is None:
        return None

    output = default_converted_cache_path(checkpoint, cache_root=cache_root)
    if output.is_file() and not overwrite:
        _ensure_outside_repo(output)
        return output
    return convert_da3_checkpoint(checkpoint, output, overwrite=overwrite)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert DA3 safetensors to local MLX .npz weights.")
    parser.add_argument("--output", type=Path, default=None, help="Out-of-git converted .npz output path.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Out-of-git checkpoint/cache root.")
    parser.add_argument("--required", action="store_true", help="Fail if checkpoint or conversion is unavailable.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite an existing converted output.")
    args = parser.parse_args(argv)

    try:
        checkpoint = resolve_da3_checkpoint(
            cache_root=args.cache_root,
            required=args.required or None,
        )
        if checkpoint is None:
            raise DA3ConversionError("DA3 checkpoint not configured")
        print_checkpoint_evidence(checkpoint)
        output = args.output or default_converted_cache_path(checkpoint, cache_root=args.cache_root)
        converted = convert_da3_checkpoint(checkpoint, output, overwrite=args.overwrite)
    except (DA3CheckpointError, DA3ConversionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Converted DA3 weights: path={converted}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through direct CLI use.
    raise SystemExit(main())
