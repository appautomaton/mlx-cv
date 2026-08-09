#!/usr/bin/env python3
"""List, stage, verify, upload, and remotely verify mlx-cv model releases."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx_cv.hub.safetensors import read_safetensors_metadata, sha256_file  # noqa: E402

__all__ = [
    "MODEL_RELEASES",
    "ModelRelease",
    "ReleaseVerificationError",
    "stage_release",
    "verify_staged_release",
]


@dataclass(frozen=True)
class ModelRelease:
    name: str
    repo_id: str
    checkpoint: str
    card: str
    license_file: str
    assets: tuple[tuple[str, str], ...]
    required_metadata: tuple[tuple[str, str], ...]


MODEL_RELEASES = {
    "locateanything-3b-bf16": ModelRelease(
        name="locateanything-3b-bf16",
        repo_id="appautomaton/locateanything-3b-bf16-mlx",
        checkpoint="models/locateanything_3b/mlx-bf16/model.safetensors",
        card="scripts/hugging_face/model_cards/appautomaton/locateanything-3b-bf16-mlx.md",
        license_file="models/locateanything_3b/mlx-bf16/LICENSE",
        assets=tuple(
            (f"models/locateanything_3b/mlx-bf16/{name}", name)
            for name in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template.json",
                "preprocessor_config.json",
                "processor_config.json",
                "vocab.json",
                "merges.txt",
            )
        ),
        required_metadata=(
            ("format", "mlx-cv-locateanything-v1"),
            ("architecture", "LocateAnything-3B"),
            ("layout", "mlx-final"),
            ("dtype", "bfloat16"),
            ("tensor_count", "769"),
        ),
    ),
    "sam3.1-multiplex-bf16": ModelRelease(
        name="sam3.1-multiplex-bf16",
        repo_id="appautomaton/sam3.1-multiplex-bf16-mlx",
        checkpoint="models/sam3_1_multiplex/mlx-bf16/model.safetensors",
        card="scripts/hugging_face/model_cards/appautomaton/sam3.1-multiplex-bf16-mlx.md",
        license_file="models/sam3_1_multiplex/mlx-bf16/LICENSE",
        assets=(
            ("models/sam3_1_multiplex/mlx-bf16/config.json", "config.json"),
            (
                "models/sam3_1_multiplex/mlx-bf16/bpe_simple_vocab_16e6.txt.gz",
                "bpe_simple_vocab_16e6.txt.gz",
            ),
        ),
        required_metadata=(
            ("format", "mlx-cv-sam3.1-v1"),
            ("architecture", "sam3.1-multiplex"),
            ("layout", "mlx-final"),
            ("dtype", "bfloat16"),
            ("scope", "multiplex"),
            ("tensor_count", "1963"),
        ),
    ),
}


class ReleaseVerificationError(ValueError):
    pass


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)


def _manifest(package: Path, release: ModelRelease) -> dict:
    files = []
    for path in sorted(package.iterdir(), key=lambda item: item.name):
        if path.name == "manifest.json":
            continue
        if not path.is_file() or path.is_symlink():
            raise ReleaseVerificationError(f"unsupported staged entry: {path}")
        files.append(
            {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {"schema_version": 1, "repo_id": release.repo_id, "files": files}


def stage_release(
    release: ModelRelease,
    *,
    source_root: str | Path,
    staging_root: str | Path,
) -> Path:
    source_root = Path(source_root).resolve()
    staging_root = Path(staging_root).resolve()
    destination = staging_root / release.name
    temporary = staging_root / f".{release.name}.{uuid4().hex}.partial"
    staging_root.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        _copy_file(source_root / release.checkpoint, temporary / "model.safetensors")
        _copy_file(source_root / release.card, temporary / "README.md")
        _copy_file(source_root / release.license_file, temporary / "LICENSE")
        for source, target in release.assets:
            _copy_file(source_root / source, temporary / target)
        manifest = _manifest(temporary, release)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_staged_release(release, temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def verify_staged_release(
    release: ModelRelease,
    package: str | Path,
    *,
    allow_cache_symlinks: bool = False,
) -> dict:
    package = Path(package)
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise ReleaseVerificationError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo_id") != release.repo_id:
        raise ReleaseVerificationError("manifest repo_id does not match release registry")
    declared = {entry["path"]: entry for entry in manifest.get("files", [])}
    actual = {path.name for path in package.iterdir() if path.name != "manifest.json"}
    if set(declared) != actual:
        raise ReleaseVerificationError(
            f"staged allowlist mismatch: declared={sorted(declared)}, actual={sorted(actual)}"
        )
    for name, entry in declared.items():
        path = package / name
        if not path.is_file() or (path.is_symlink() and not allow_cache_symlinks):
            raise ReleaseVerificationError(f"invalid staged file: {path}")
        if path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            raise ReleaseVerificationError(f"manifest mismatch: {path}")
    metadata = read_safetensors_metadata(package / "model.safetensors")
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in release.required_metadata
        if metadata.get(key) != expected
    }
    if not metadata.get("source_sha256"):
        mismatches["source_sha256"] = (metadata.get("source_sha256"), "non-empty")
    if mismatches:
        raise ReleaseVerificationError(f"checkpoint metadata mismatch: {mismatches}")
    if (package / "README.md").read_text().lstrip().startswith("---") is False:
        raise ReleaseVerificationError("model card must begin with YAML front matter")
    if not (package / "LICENSE").read_text().strip():
        raise ReleaseVerificationError("model license must not be empty")
    return manifest


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
