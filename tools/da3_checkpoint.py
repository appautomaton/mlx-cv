"""Depth Anything 3 checkpoint cache and provenance helpers.

This module is intentionally under ``tools/``: it may download files and is not
part of the ``mlx_cv`` runtime package.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DA3_MODEL_ID_ENV = "MLX_CV_DA3_MODEL_ID"
DA3_CHECKPOINT_ENV = "MLX_CV_DA3_CHECKPOINT"
DA3_CONFIG_ENV = "MLX_CV_DA3_CONFIG"
DA3_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_DA3_GATE"
DA3_DOWNLOAD_ENV = "MLX_CV_DA3_ALLOW_DOWNLOAD"
MLX_CV_CACHE_ENV = "MLX_CV_CACHE"

DA3_DEFAULT_MODEL_ID = "depth-anything/DA3-SMALL"
DA3_FALLBACK_MODEL_ID = "depth-anything/DA3-BASE"
DA3_SUPPORTED_MODEL_IDS = (DA3_DEFAULT_MODEL_ID, DA3_FALLBACK_MODEL_ID)

DA3_CONFIG_FILENAME = "config.json"
DA3_CHECKPOINT_FILENAME = "model.safetensors"
DA3_REVISION = "main"

DA3_LICENSE_NOTES = {
    DA3_DEFAULT_MODEL_ID: "Apache-2.0",
    DA3_FALLBACK_MODEL_ID: "Apache-2.0",
}


class DA3CheckpointError(RuntimeError):
    """Raised when a required DA3 checkpoint gate cannot proceed."""


@dataclass(frozen=True)
class DA3CheckpointInfo:
    model_id: str
    checkpoint_path: Path
    config_path: Path
    checkpoint_sha256: str
    config_sha256: str
    checkpoint_url: str
    config_url: str
    revision: str
    license_note: str
    source: str

    def evidence(self) -> str:
        return (
            "DA3 checkpoint: "
            f"model_id={self.model_id} "
            f"revision={self.revision} "
            f"license={self.license_note} "
            f"config={self.config_path} "
            f"config_sha256={self.config_sha256} "
            f"weights={self.checkpoint_path} "
            f"weights_sha256={self.checkpoint_sha256} "
            f"source={self.source}"
        )


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def required_gate_enabled(environ: dict[str, str] | None = None) -> bool:
    environ = os.environ if environ is None else environ
    return _truthy(environ.get(DA3_REQUIRED_GATE_ENV))


def download_enabled(environ: dict[str, str] | None = None) -> bool:
    environ = os.environ if environ is None else environ
    return _truthy(environ.get(DA3_DOWNLOAD_ENV))


def default_cache_root(environ: dict[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    if environ.get(MLX_CV_CACHE_ENV):
        return Path(environ[MLX_CV_CACHE_ENV]).expanduser()
    return Path.home() / ".cache" / "mlx-cv"


def normalize_model_id(model_id: str) -> str:
    model_id = model_id.strip()
    if model_id not in DA3_SUPPORTED_MODEL_IDS:
        raise DA3CheckpointError(
            f"unsupported DA3 model id {model_id!r}; expected one of {', '.join(DA3_SUPPORTED_MODEL_IDS)}"
        )
    return model_id


def model_id_from_env(environ: dict[str, str] | None = None) -> str:
    environ = os.environ if environ is None else environ
    return normalize_model_id(environ.get(DA3_MODEL_ID_ENV, DA3_DEFAULT_MODEL_ID))


def model_cache_dir(cache_root: Path, model_id: str) -> Path:
    safe = model_id.replace("/", "__")
    return cache_root.expanduser() / "da3" / safe


def hf_resolve_url(model_id: str, filename: str, revision: str = DA3_REVISION) -> str:
    return f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _explicit_paths(environ: dict[str, str]) -> tuple[Path | None, Path | None]:
    checkpoint = environ.get(DA3_CHECKPOINT_ENV)
    config = environ.get(DA3_CONFIG_ENV)
    return (
        None if checkpoint is None else Path(checkpoint).expanduser(),
        None if config is None else Path(config).expanduser(),
    )


def _resolve_existing_pair(
    *,
    model_id: str,
    checkpoint_path: Path,
    config_path: Path,
    source: str,
    revision: str,
) -> DA3CheckpointInfo:
    if not checkpoint_path.is_file():
        raise DA3CheckpointError(f"DA3 checkpoint file is missing: {checkpoint_path}")
    if not config_path.is_file():
        raise DA3CheckpointError(f"DA3 config file is missing: {config_path}")
    return DA3CheckpointInfo(
        model_id=model_id,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        config_sha256=sha256_file(config_path),
        checkpoint_url=hf_resolve_url(model_id, DA3_CHECKPOINT_FILENAME, revision),
        config_url=hf_resolve_url(model_id, DA3_CONFIG_FILENAME, revision),
        revision=revision,
        license_note=DA3_LICENSE_NOTES[model_id],
        source=source,
    )


def resolve_da3_checkpoint(
    *,
    environ: dict[str, str] | None = None,
    cache_root: Path | None = None,
    model_id: str | None = None,
    required: bool | None = None,
    allow_download: bool | None = None,
    revision: str = DA3_REVISION,
) -> DA3CheckpointInfo | None:
    """Resolve DA3 config and checkpoint files.

    Returns ``None`` only when the gate is optional and no checkpoint/config pair
    is configured. Required mode raises ``DA3CheckpointError`` for any missing
    prerequisite so phase-closing tests cannot silently skip.
    """

    environ = os.environ if environ is None else environ
    model_id = normalize_model_id(model_id) if model_id is not None else model_id_from_env(environ)
    required = required_gate_enabled(environ) if required is None else required
    allow_download = download_enabled(environ) if allow_download is None else allow_download

    explicit_checkpoint, explicit_config = _explicit_paths(environ)
    if explicit_checkpoint is not None or explicit_config is not None:
        if explicit_checkpoint is None or explicit_config is None:
            raise DA3CheckpointError(
                f"set both {DA3_CHECKPOINT_ENV} and {DA3_CONFIG_ENV} for explicit DA3 paths"
            )
        return _resolve_existing_pair(
            model_id=model_id,
            checkpoint_path=explicit_checkpoint,
            config_path=explicit_config,
            source=f"{DA3_CHECKPOINT_ENV}+{DA3_CONFIG_ENV}",
            revision=revision,
        )

    root = default_cache_root(environ) if cache_root is None else cache_root.expanduser()
    model_dir = model_cache_dir(root, model_id)
    checkpoint_path = model_dir / DA3_CHECKPOINT_FILENAME
    config_path = model_dir / DA3_CONFIG_FILENAME
    if checkpoint_path.is_file() or config_path.is_file():
        return _resolve_existing_pair(
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            source=str(model_dir),
            revision=revision,
        )

    if allow_download:
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_url = hf_resolve_url(model_id, DA3_CHECKPOINT_FILENAME, revision)
        config_url = hf_resolve_url(model_id, DA3_CONFIG_FILENAME, revision)
        _download_file(config_url, config_path)
        _download_file(checkpoint_url, checkpoint_path)
        return _resolve_existing_pair(
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            source=f"{model_id}@{revision}",
            revision=revision,
        )

    if required:
        raise DA3CheckpointError(
            f"DA3 checkpoint is required but missing. Set {DA3_CHECKPOINT_ENV} and {DA3_CONFIG_ENV}, "
            f"or place {DA3_CONFIG_FILENAME} and {DA3_CHECKPOINT_FILENAME} under {model_dir}. "
            f"To download explicitly, set {DA3_DOWNLOAD_ENV}=1 or pass --download."
        )
    return None


def print_checkpoint_evidence(info: DA3CheckpointInfo) -> None:
    print(info.evidence())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve and verify Depth Anything 3 checkpoint files.")
    parser.add_argument("--model-id", default=None, help=f"DA3 model id, default {DA3_DEFAULT_MODEL_ID}.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Out-of-git checkpoint cache root.")
    parser.add_argument("--download", action="store_true", help="Opt in to downloading config/checkpoint.")
    parser.add_argument("--required", action="store_true", help="Fail if config/checkpoint are unavailable.")
    args = parser.parse_args(argv)

    try:
        info = resolve_da3_checkpoint(
            cache_root=args.cache_root,
            model_id=args.model_id,
            required=args.required or None,
            allow_download=args.download or None,
        )
    except DA3CheckpointError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if info is None:
        model_id = args.model_id or os.environ.get(DA3_MODEL_ID_ENV, DA3_DEFAULT_MODEL_ID)
        print(
            f"DA3 checkpoint not configured for {model_id}; set {DA3_CHECKPOINT_ENV}+{DA3_CONFIG_ENV} "
            f"or {MLX_CV_CACHE_ENV}. Use --required for phase-closing gates.",
            file=sys.stderr,
        )
        return 1
    print_checkpoint_evidence(info)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through direct CLI use.
    raise SystemExit(main())
