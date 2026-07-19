"""Hub: weight loading / conversion plumbing shared across models.

Seeded with the declarative weight-convert (`sanitize`) engine — the key-remap +
layout-fix machinery every model's load path reuses (`Rename`/`Transpose`/`Drop`
rules → mlx param tree). Download/cache and quantization land here in later phases.

mlx lives here, behind the ``[mlx]`` extra; ``core/`` stays mlx-free.
"""

from __future__ import annotations

from .convert import Drop, Rename, Transpose, convert_state_dict, load_into
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
    "Rename",
    "Transpose",
    "convert_state_dict",
    "load_into",
    "resolve_pretrained",
    "read_safetensors_header",
    "read_safetensors_metadata",
    "rewrite_safetensors_metadata",
    "sha256_file",
]
