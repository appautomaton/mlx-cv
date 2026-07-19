"""Small, dependency-free Safetensors header and metadata helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Mapping
from uuid import uuid4

__all__ = [
    "read_safetensors_header",
    "read_safetensors_metadata",
    "rewrite_safetensors_metadata",
    "sha256_file",
]


def read_safetensors_header(path: str | Path) -> dict:
    path = Path(path)
    try:
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise ValueError("missing Safetensors header length")
            header_size = int.from_bytes(raw_size, "little")
            if header_size <= 0 or header_size > 100_000_000:
                raise ValueError(f"invalid Safetensors header size: {header_size}")
            header = json.loads(handle.read(header_size))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header must be an object: {path}")
    return header


def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    metadata = read_safetensors_header(path).get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError("Safetensors metadata must contain only string pairs")
    return metadata


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_safetensors_metadata(
    source: str | Path,
    destination: str | Path,
    metadata: Mapping[str, str],
) -> Path:
    """Atomically copy a Safetensors file while replacing only its metadata.

    Tensor bytes stream directly from source to destination, so multi-gigabyte
    checkpoints do not materialize in RAM. Tensor offsets are relative to the
    data section and therefore remain unchanged when the JSON header changes.
    """

    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ for atomic metadata rewrite")
    header = read_safetensors_header(source)
    header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            old_size = int.from_bytes(reader.read(8), "little")
            reader.seek(8 + old_size)
            writer.write(len(encoded).to_bytes(8, "little"))
            writer.write(encoded)
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
