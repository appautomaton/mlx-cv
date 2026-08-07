"""Checkpoint resolution, metadata, and weight-conversion utilities.

The resolver accepts local paths or revision-aware Hub snapshots. Declarative
``Rename``, ``Transpose``, and ``Drop`` rules convert external state dictionaries
into MLX parameter trees. Quantization is not part of this module's current API.
"""

from __future__ import annotations

from .convert import Drop, Rename, Transpose, convert_state_dict, load_into
from .package import ResolvedModelPackage, resolve_model_package
from .resolver import (
    DEFAULT_MODEL_ALIASES,
    HubDependencyError,
    PretrainedResolutionError,
    resolve_pretrained,
)
from .safetensors import (
    read_safetensors_header,
    read_safetensors_metadata,
    rewrite_safetensors_metadata,
    sha256_file,
)

__all__ = [
    "DEFAULT_MODEL_ALIASES",
    "Drop",
    "HubDependencyError",
    "PretrainedResolutionError",
    "ResolvedModelPackage",
    "Rename",
    "Transpose",
    "convert_state_dict",
    "load_into",
    "resolve_pretrained",
    "resolve_model_package",
    "read_safetensors_header",
    "read_safetensors_metadata",
    "rewrite_safetensors_metadata",
    "sha256_file",
]
