"""Convert the verified RF-DETR Nano ``.pth`` checkpoint to local MLX weights.

This module is intentionally under ``tools/`` because it imports PyTorch to
extract the upstream checkpoint. Runtime loading in ``src/mlx_cv`` consumes only
the converted ``.npz``/safetensors arrays.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from rfdetr_checkpoint import (
        CheckpointError,
        CheckpointInfo,
        default_cache_root,
        print_checkpoint_evidence,
        required_gate_enabled,
        resolve_rfdetr_nano_checkpoint,
        verify_md5,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported outside tools/.
    _CHECKPOINT_PATH = Path(__file__).with_name("rfdetr_checkpoint.py")
    _SPEC = importlib.util.spec_from_file_location("rfdetr_checkpoint", _CHECKPOINT_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _MODULE
    _SPEC.loader.exec_module(_MODULE)
    CheckpointError = _MODULE.CheckpointError
    CheckpointInfo = _MODULE.CheckpointInfo
    default_cache_root = _MODULE.default_cache_root
    print_checkpoint_evidence = _MODULE.print_checkpoint_evidence
    required_gate_enabled = _MODULE.required_gate_enabled
    resolve_rfdetr_nano_checkpoint = _MODULE.resolve_rfdetr_nano_checkpoint
    verify_md5 = _MODULE.verify_md5

RFDETR_NANO_CONVERTED_ENV = "MLX_CV_RFDETR_NANO_CONVERTED"
RFDETR_NANO_CONVERTED_FILENAME = "rf-detr-nano-mlx.npz"


class RFDETRConversionError(RuntimeError):
    """Raised when a real RF-DETR checkpoint cannot be converted or resolved."""


class RFDETRConversionDependencyError(RFDETRConversionError):
    """Raised when optional tool-side conversion dependencies are unavailable."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_outside_repo(path: Path) -> None:
    resolved = path.expanduser().resolve(strict=False)
    root = _repo_root().resolve()
    if resolved == root or root in resolved.parents:
        raise RFDETRConversionError(
            f"converted RF-DETR weights must stay outside git; {resolved} is under {root}"
        )


def _args_to_dict(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return {str(key): value for key, value in args.items()}
    if hasattr(args, "__dict__"):
        return {str(key): value for key, value in vars(args).items()}
    return {}


def _load_torch_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch env.
        raise RFDETRConversionDependencyError("RF-DETR Nano checkpoint conversion requires torch.") from exc

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise RFDETRConversionError(f"{path} has unsupported checkpoint root type {type(raw)!r}")
    state = raw.get("model", raw)
    if not isinstance(state, dict):
        raise RFDETRConversionError(f"{path} does not contain a model state dict")
    return state, _args_to_dict(raw.get("args"))


def _runtime_conversion_api():
    try:
        from mlx_cv.models.rfdetr import RFDETRConfig, convert_rfdetr_state_dict
    except Exception as exc:  # pragma: no cover - depends on optional MLX runtime availability.
        raise RFDETRConversionDependencyError("RF-DETR Nano conversion requires the MLX runtime.") from exc
    return RFDETRConfig, convert_rfdetr_state_dict


def _to_numpy_state(state: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in state.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        out[str(key)] = np.asarray(value)
    return out


def default_converted_cache_path(checkpoint: CheckpointInfo, *, cache_root: Path | None = None) -> Path:
    root = default_cache_root() if cache_root is None else cache_root.expanduser()
    return root / "rfdetr" / f"rf-detr-nano-{checkpoint.md5[:12]}-mlx.npz"


def convert_rfdetr_nano_checkpoint(
    checkpoint: CheckpointInfo,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert a verified RF-DETR Nano checkpoint to local MLX ``.npz`` weights."""

    output_path = output_path.expanduser()
    _ensure_outside_repo(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    state, args = _load_torch_checkpoint(checkpoint.path)
    RFDETRConfig, convert_rfdetr_state_dict = _runtime_conversion_api()
    cfg = RFDETRConfig.rfdetr_nano()
    converted = convert_rfdetr_state_dict(
        _to_numpy_state(state),
        target_num_queries=cfg.decoder.num_queries,
        target_group_detr=cfg.decoder.group_detr,
        target_query_dim=cfg.decoder.query_dim,
        ckpt_num_queries=int(args.get("num_queries", cfg.decoder.num_queries)),
        ckpt_group_detr=int(args.get("group_detr", cfg.decoder.group_detr)),
    )
    arrays: dict[str, np.ndarray] = {}
    duplicates: list[str] = []
    for key, value in converted:
        if key in arrays:
            duplicates.append(key)
        arrays[key] = value
    if duplicates:
        sample = ", ".join(repr(key) for key in duplicates[:5])
        more = "" if len(duplicates) <= 5 else f", and {len(duplicates) - 5} more"
        raise RFDETRConversionError(f"duplicate converted RF-DETR keys: {sample}{more}")

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


def resolve_rfdetr_nano_converted_weights(
    *,
    environ: dict[str, str] | None = None,
    cache_root: Path | None = None,
    required: bool | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Resolve a converted RF-DETR Nano weight file, converting to cache if needed."""

    environ = os.environ if environ is None else environ
    required = required_gate_enabled(environ) if required is None else required

    env_path = environ.get(RFDETR_NANO_CONVERTED_ENV)
    if env_path:
        converted = Path(env_path).expanduser()
        if not converted.is_file():
            raise RFDETRConversionError(f"{RFDETR_NANO_CONVERTED_ENV}={converted} is not a file")
        _ensure_outside_repo(converted)
        return converted

    checkpoint = resolve_rfdetr_nano_checkpoint(
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
    return convert_rfdetr_nano_checkpoint(checkpoint, output, overwrite=overwrite)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert RF-DETR Nano .pth to local MLX .npz weights.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Verified RF-DETR Nano .pth path.")
    parser.add_argument("--output", type=Path, default=None, help="Out-of-git converted .npz output path.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Out-of-git checkpoint/cache root.")
    parser.add_argument("--required", action="store_true", help="Fail if checkpoint or conversion is unavailable.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite an existing converted output.")
    args = parser.parse_args(argv)

    try:
        if args.checkpoint is not None:
            path = args.checkpoint.expanduser()
            if not path.is_file():
                raise RFDETRConversionError(f"--checkpoint {path} is not a file")
            md5 = verify_md5(path)
            checkpoint = CheckpointInfo(path=path, md5=md5, source="--checkpoint")
        else:
            checkpoint = resolve_rfdetr_nano_checkpoint(
                cache_root=args.cache_root,
                required=args.required or None,
            )
            if checkpoint is None:
                raise RFDETRConversionError("RF-DETR Nano checkpoint not configured")
        print_checkpoint_evidence(checkpoint)
        output = args.output or default_converted_cache_path(checkpoint, cache_root=args.cache_root)
        converted = convert_rfdetr_nano_checkpoint(checkpoint, output, overwrite=args.overwrite)
    except (CheckpointError, RFDETRConversionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Converted RF-DETR Nano weights: path={converted}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through direct CLI use.
    raise SystemExit(main())
