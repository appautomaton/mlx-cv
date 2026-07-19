"""Resolve local model packages and Hugging Face snapshots lazily."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

__all__ = [
    "DEFAULT_MODEL_ALIASES",
    "HubDependencyError",
    "PretrainedResolutionError",
    "resolve_pretrained",
]


DEFAULT_MODEL_ALIASES = {
    "locateanything-3b": "appautomaton/locateanything-3b-bf16-mlx",
    "locateanything-3b-bf16": "appautomaton/locateanything-3b-bf16-mlx",
    "sam3.1": "appautomaton/sam3.1-multiplex-bf16-mlx",
    "sam3.1-multiplex-bf16": "appautomaton/sam3.1-multiplex-bf16-mlx",
}


class HubDependencyError(ImportError):
    """Raised when a remote/cache lookup needs the optional Hub extra."""


class PretrainedResolutionError(ValueError):
    """Raised when an identifier is neither a local path, alias, nor repo ID."""


def _offline_from_environment() -> bool:
    value = os.environ.get("HF_HUB_OFFLINE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def resolve_pretrained(
    identifier: str | os.PathLike[str],
    *,
    aliases: Mapping[str, str] | None = None,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool | None = None,
    token: str | bool | None = None,
) -> Path:
    """Resolve a package without importing Hub code for existing local paths.

    Remote identifiers are downloaded with ``snapshot_download``.  This helper
    never loads Python from a model repository and has no ``trust_remote_code``
    escape hatch.
    """

    candidate = Path(identifier).expanduser()
    if candidate.exists():
        return candidate.resolve()

    value = os.fspath(identifier)
    alias_table = dict(DEFAULT_MODEL_ALIASES)
    if aliases is not None:
        alias_table.update(aliases)
    repo_id = alias_table.get(value, value)
    if "/" not in repo_id or repo_id.startswith(("/", "./", "../")):
        known = ", ".join(sorted(alias_table))
        raise PretrainedResolutionError(
            f"unknown pretrained identifier {value!r}; use a local path, an exact "
            f"Hugging Face repo ID, or one of: {known}"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise HubDependencyError(
            "Hugging Face model resolution requires `pip install \"mlx-cv[hub]\"`"
        ) from exc

    offline = _offline_from_environment() if local_files_only is None else bool(local_files_only)
    try:
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=None if cache_dir is None else os.fspath(cache_dir),
            local_files_only=offline,
            token=token,
        )
    except Exception as exc:
        mode = "offline cache" if offline else "Hugging Face Hub"
        raise PretrainedResolutionError(
            f"could not resolve {repo_id!r} from the {mode}"
        ) from exc
    return Path(resolved).resolve()
