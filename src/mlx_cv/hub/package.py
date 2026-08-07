"""Shared local/remote package-file resolution for model families."""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

from .resolver import resolve_pretrained

__all__ = ["ResolvedModelPackage", "resolve_model_package"]


@dataclass(frozen=True)
class ResolvedModelPackage:
    """Resolved checkpoint plus optional JSON configuration."""

    source: Path
    weights: Path
    config: dict[str, Any] | None

    @property
    def root(self) -> Path:
        return self.source if self.source.is_dir() else self.source.parent


def resolve_model_package(
    identifier: str | PathLike[str],
    *,
    weight_names: tuple[str, ...] = ("model.safetensors", "model.npz"),
    require_config: bool = True,
    aliases: Mapping[str, str] | None = None,
    revision: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool | None = None,
    token: str | bool | None = None,
) -> ResolvedModelPackage:
    """Resolve standard model-package files without executing repository code."""

    source = resolve_pretrained(
        identifier,
        aliases=aliases,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    if source.is_file():
        return ResolvedModelPackage(source=source, weights=source, config=None)
    if not source.is_dir():
        raise ValueError(f"resolved model source is neither a file nor directory: {source}")

    weights = next((source / name for name in weight_names if (source / name).is_file()), None)
    if weights is None:
        expected = ", ".join(weight_names)
        raise FileNotFoundError(
            f"model package {source} is missing a checkpoint; expected one of: {expected}"
        )

    config_path = source / "config.json"
    if not config_path.is_file():
        if require_config:
            raise FileNotFoundError(f"model package is missing config.json: {source}")
        config = None
    else:
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"model package has invalid config.json: {config_path}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"model package config.json must contain a JSON object: {config_path}")

    return ResolvedModelPackage(source=source, weights=weights, config=config)
