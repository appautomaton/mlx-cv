"""Hub: weight loading / conversion plumbing shared across models.

Seeded with the declarative weight-convert (`sanitize`) engine — the key-remap +
layout-fix machinery every model's load path reuses (`Rename`/`Transpose`/`Drop`
rules → mlx param tree). Download/cache and quantization land here in later phases.

mlx lives here, behind the ``[mlx]`` extra; ``core/`` stays mlx-free.
"""

from __future__ import annotations

from .convert import Drop, Rename, Transpose, convert_state_dict, load_into

__all__ = ["Drop", "Rename", "Transpose", "convert_state_dict", "load_into"]
