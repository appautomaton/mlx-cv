"""RF-DETR Nano checkpoint cache and checksum helpers.

This module is intentionally under ``tools/``: it may download files and is not
part of the ``mlx_cv`` runtime package.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RFDETR_NANO_CHECKPOINT_ENV = "MLX_CV_RFDETR_NANO_CHECKPOINT"
RFDETR_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_RFDETR_GATE"
RFDETR_DOWNLOAD_ENV = "MLX_CV_RFDETR_ALLOW_DOWNLOAD"
MLX_CV_CACHE_ENV = "MLX_CV_CACHE"

RFDETR_NANO_CHECKPOINT_URL = "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
RFDETR_NANO_CHECKPOINT_FILENAME = "rf-detr-nano.pth"
RFDETR_NANO_CHECKPOINT_URL_BASENAME = "checkpoint_best_regular.pth"
RFDETR_NANO_EXPECTED_MD5 = "fb6504cce7fbdc783f7a46991f07639f"


class CheckpointError(RuntimeError):
    """Raised when a required RF-DETR checkpoint gate cannot proceed."""


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    md5: str
    source: str
    url: str = RFDETR_NANO_CHECKPOINT_URL
    filename: str = RFDETR_NANO_CHECKPOINT_FILENAME

    def evidence(self) -> str:
        return f"RF-DETR Nano checkpoint: path={self.path} md5={self.md5}"


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def required_gate_enabled(environ: dict[str, str] | None = None) -> bool:
    environ = os.environ if environ is None else environ
    return _truthy(environ.get(RFDETR_REQUIRED_GATE_ENV))


def download_enabled(environ: dict[str, str] | None = None) -> bool:
    environ = os.environ if environ is None else environ
    return _truthy(environ.get(RFDETR_DOWNLOAD_ENV))


def default_cache_root(environ: dict[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    if environ.get(MLX_CV_CACHE_ENV):
        return Path(environ[MLX_CV_CACHE_ENV]).expanduser()
    return Path.home() / ".cache" / "mlx-cv"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: Path, expected_md5: str = RFDETR_NANO_EXPECTED_MD5) -> str:
    actual = md5_file(path)
    if actual != expected_md5:
        raise CheckpointError(f"{path} has md5 {actual}; expected {expected_md5}")
    return actual


def _download_file(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _canonicalize_cache_file(cache_root: Path, expected_md5: str) -> Path | None:
    canonical = cache_root / RFDETR_NANO_CHECKPOINT_FILENAME
    if canonical.is_file():
        verify_md5(canonical, expected_md5)
        return canonical

    source_basename = cache_root / RFDETR_NANO_CHECKPOINT_URL_BASENAME
    if source_basename.is_file():
        verify_md5(source_basename, expected_md5)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_basename, canonical)
        verify_md5(canonical, expected_md5)
        return canonical

    return None


def resolve_rfdetr_nano_checkpoint(
    *,
    environ: dict[str, str] | None = None,
    cache_root: Path | None = None,
    required: bool | None = None,
    allow_download: bool | None = None,
    expected_md5: str = RFDETR_NANO_EXPECTED_MD5,
) -> CheckpointInfo | None:
    """Resolve and verify RF-DETR Nano checkpoint.

    Returns ``None`` only when the gate is optional and no checkpoint is
    configured. Required mode raises ``CheckpointError`` for any missing or bad
    prerequisite so phase-closing tests cannot silently skip.
    """

    environ = os.environ if environ is None else environ
    required = required_gate_enabled(environ) if required is None else required
    allow_download = download_enabled(environ) if allow_download is None else allow_download

    env_path = environ.get(RFDETR_NANO_CHECKPOINT_ENV)
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise CheckpointError(f"{RFDETR_NANO_CHECKPOINT_ENV}={path} is not a file")
        actual = verify_md5(path, expected_md5)
        return CheckpointInfo(path=path, md5=actual, source=RFDETR_NANO_CHECKPOINT_ENV)

    root = default_cache_root(environ) if cache_root is None else cache_root.expanduser()
    try:
        path = _canonicalize_cache_file(root, expected_md5)
    except CheckpointError:
        raise
    if path is not None:
        actual = verify_md5(path, expected_md5)
        return CheckpointInfo(path=path, md5=actual, source=str(root))

    if allow_download:
        path = root / RFDETR_NANO_CHECKPOINT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        _download_file(RFDETR_NANO_CHECKPOINT_URL, path)
        actual = verify_md5(path, expected_md5)
        return CheckpointInfo(path=path, md5=actual, source=RFDETR_NANO_CHECKPOINT_URL)

    if required:
        raise CheckpointError(
            f"RF-DETR Nano checkpoint is required but missing. Set {RFDETR_NANO_CHECKPOINT_ENV} "
            f"or place {RFDETR_NANO_CHECKPOINT_FILENAME} under {root}. To download explicitly, "
            f"set {RFDETR_DOWNLOAD_ENV}=1."
        )
    return None


def print_checkpoint_evidence(info: CheckpointInfo) -> None:
    print(info.evidence())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve and verify the RF-DETR Nano checkpoint.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Out-of-git checkpoint cache root.")
    parser.add_argument("--download", action="store_true", help="Opt in to downloading the checkpoint.")
    parser.add_argument("--required", action="store_true", help="Fail if the checkpoint is unavailable.")
    args = parser.parse_args(argv)

    try:
        info = resolve_rfdetr_nano_checkpoint(
            cache_root=args.cache_root,
            required=args.required or None,
            allow_download=args.download or None,
        )
    except CheckpointError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if info is None:
        print(
            f"RF-DETR Nano checkpoint not configured; set {RFDETR_NANO_CHECKPOINT_ENV} "
            f"or {MLX_CV_CACHE_ENV}. Use --required for phase-closing gates.",
            file=sys.stderr,
        )
        return 1
    print_checkpoint_evidence(info)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through direct CLI use.
    raise SystemExit(main())
