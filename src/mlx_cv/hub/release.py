"""Reproducible Hugging Face package staging and verification."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .safetensors import read_safetensors_metadata, sha256_file

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
        checkpoint="models/locateanything/mlx/locateanything-3b-bf16.safetensors",
        card="scripts/hugging_face/model_cards/appautomaton/locateanything-3b-bf16-mlx.md",
        license_file="models/locateanything/upstream/LICENSE",
        assets=tuple(
            (f"models/locateanything/upstream/{name}", name)
            for name in (
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
        checkpoint="models/sam3.1/mlx/sam3.1-multiplex-bf16.safetensors",
        card="scripts/hugging_face/model_cards/appautomaton/sam3.1-multiplex-bf16-mlx.md",
        license_file="models/sam3-video/upstream/LICENSE",
        assets=(
            ("models/sam3-video/upstream/config.json", "config.json"),
            (
                "references/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
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


def _locate_config(destination: Path) -> None:
    from ..models.locateanything import LocateAnythingConfig

    destination.write_text(
        json.dumps(LocateAnythingConfig().to_dict(), indent=2, sort_keys=True) + "\n"
    )


def _locate_tokenizer(source_root: Path, destination: Path) -> None:
    try:
        from tokenizers import AddedToken, Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
    except ImportError as exc:
        raise ImportError(
            "LocateAnything staging requires `pip install tokenizers`"
        ) from exc

    upstream = source_root / "models/locateanything/upstream"
    tokenizer = Tokenizer(
        BPE.from_file(str(upstream / "vocab.json"), str(upstream / "merges.txt"))
    )
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    decoder = json.loads((upstream / "tokenizer_config.json").read_text())[
        "added_tokens_decoder"
    ]
    for expected_id, config in sorted(decoder.items(), key=lambda item: int(item[0])):
        added = AddedToken(
            config["content"],
            single_word=bool(config.get("single_word", False)),
            lstrip=bool(config.get("lstrip", False)),
            rstrip=bool(config.get("rstrip", False)),
            normalized=bool(config.get("normalized", True)),
            special=bool(config.get("special", False)),
        )
        if added.special:
            tokenizer.add_special_tokens([added])
        else:
            tokenizer.add_tokens([added])
        actual_id = tokenizer.token_to_id(config["content"])
        if actual_id != int(expected_id):
            raise ReleaseVerificationError(
                f"tokenizer ID mismatch for {config['content']!r}: "
                f"{actual_id} != {expected_id}"
            )
    tokenizer.save(str(destination))


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
        if release.name == "locateanything-3b-bf16":
            _locate_config(temporary / "config.json")
            _locate_tokenizer(source_root, temporary / "tokenizer.json")
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
