#!/usr/bin/env python3
"""List, stage, verify, upload, and remotely verify mlx-cv model releases."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx_cv.hub.release import MODEL_RELEASES, stage_release, verify_staged_release


def _targets(args) -> list[str]:
    if args.all:
        return list(MODEL_RELEASES)
    if not args.targets:
        raise SystemExit("select one or more targets, or pass --all")
    unknown = sorted(set(args.targets) - set(MODEL_RELEASES))
    if unknown:
        raise SystemExit(f"unknown release target(s): {', '.join(unknown)}")
    return args.targets


def _upload(release, package: Path, *, resume: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    exists = api.repo_exists(release.repo_id, repo_type="model")
    if exists and not resume:
        raise SystemExit(
            f"refusing existing repository {release.repo_id}; pass --resume only "
            "after confirming it is the exact configured release"
        )
    if not exists:
        api.create_repo(release.repo_id, repo_type="model", private=False, exist_ok=False)
    api.upload_large_folder(
        repo_id=release.repo_id,
        repo_type="model",
        folder_path=str(package),
        num_workers=1,
    )


def _verify_remote(release, *, fresh_cache: bool) -> None:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    remote = set(api.list_repo_files(release.repo_id, repo_type="model"))
    required = {"README.md", "LICENSE", "manifest.json", "model.safetensors"}
    missing = sorted(required - remote)
    if missing:
        raise SystemExit(f"remote {release.repo_id} is missing: {missing}")
    with tempfile.TemporaryDirectory(prefix="mlx-cv-hf-") if fresh_cache else _null_temp() as cache:
        snapshot = snapshot_download(
            repo_id=release.repo_id,
            repo_type="model",
            cache_dir=cache,
            force_download=fresh_cache,
        )
        verify_staged_release(release, snapshot, allow_cache_symlinks=True)


class _null_temp:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("list", "stage", "verify", "upload", "verify-remote"))
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--staging-root", type=Path, default=Path(".release/huggingface"))
    args = parser.parse_args()

    if args.command == "list":
        for name, release in MODEL_RELEASES.items():
            print(f"{name:<30} {release.repo_id}")
        return
    for name in _targets(args):
        release = MODEL_RELEASES[name]
        package = args.staging_root / name
        if args.command == "stage":
            package = stage_release(
                release, source_root=args.source_root, staging_root=args.staging_root
            )
            print(package)
        elif args.command == "verify":
            verify_staged_release(release, package)
            print(f"verified {package}")
        elif args.command == "upload":
            verify_staged_release(release, package)
            _upload(release, package, resume=args.resume)
        else:
            _verify_remote(release, fresh_cache=args.fresh_cache)


if __name__ == "__main__":
    main()
