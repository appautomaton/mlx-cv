"""Convert official SAM 3.1 weights once into final-layout MLX Safetensors."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mlx.core as mx

from mlx_cv.models.sam3.sam31_checkpoint import SAM31_CHECKPOINT_METADATA
from mlx_cv.models.sam3.sam31_convert import (
    convert_sam31_detector_state_dict,
    convert_sam31_tracker_state_dict,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_detector(source: Path, output: Path, *, overwrite: bool = False) -> Path:
    """Create an atomic BF16 detector checkpoint in final MLX name/layout form."""

    if output.suffix != ".safetensors":
        raise ValueError("SAM 3.1 converted output must end in .safetensors")
    if output.exists() and not overwrite:
        return output
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - conversion-only dependency
        raise RuntimeError("SAM 3.1 conversion requires PyTorch") from exc

    state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    converted = convert_sam31_detector_state_dict(state)
    weights = {
        key: mx.array(value).astype(mx.bfloat16)
        for key, value in converted.items()
    }
    mx.eval(weights)
    metadata = {
        **SAM31_CHECKPOINT_METADATA,
        "scope": "detector",
        "source_sha256": _sha256(source),
        "source_tensor_count": "1166",
        "tensor_count": str(len(weights)),
        "qkv": "split",
        "rope": "regenerated-from-config",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # MLX appends ``.safetensors`` when the supplied path has another suffix,
    # so keep that as the final suffix while retaining an atomic temp name.
    temporary = output.with_name(output.stem + ".tmp.safetensors")
    try:
        mx.save_safetensors(str(temporary), weights, metadata=metadata)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def convert_checkpoint(source: Path, output: Path, *, overwrite: bool = False) -> Path:
    """Atomically create the complete final-layout SAM 3.1 BF16 checkpoint."""

    if output.suffix != ".safetensors":
        raise ValueError("SAM 3.1 converted output must end in .safetensors")
    if output.exists() and not overwrite:
        return output
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - conversion-only dependency
        raise RuntimeError("SAM 3.1 conversion requires PyTorch") from exc

    state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    detector = convert_sam31_detector_state_dict(state)
    tracker = convert_sam31_tracker_state_dict(state)
    converted = {
        **{f"detector.{key}": value for key, value in detector.items()},
        **{f"tracker.{key}": value for key, value in tracker.items()},
    }
    weights = {
        key: mx.array(value).astype(mx.bfloat16)
        for key, value in converted.items()
    }
    mx.eval(weights)
    metadata = {
        **SAM31_CHECKPOINT_METADATA,
        "scope": "multiplex",
        "source_sha256": _sha256(source),
        "source_tensor_count": "1623",
        "source_detector_tensor_count": "1166",
        "source_tracker_tensor_count": "457",
        "tensor_count": str(len(weights)),
        "detector_tensor_count": str(len(detector)),
        "tracker_tensor_count": str(len(tracker)),
        "qkv": "split",
        "rope": "regenerated-from-config",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.safetensors")
    try:
        mx.save_safetensors(str(temporary), weights, metadata=metadata)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = convert_checkpoint(args.source, args.output, overwrite=args.overwrite)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
