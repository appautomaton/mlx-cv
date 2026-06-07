"""LocateAnything-3B — the anchor grounding model (NVIDIA). See ARCHITECTURE.md §16.

Build status:
  * Stage 1 (here): config, weight-convert rules, PBD output parser — mlx-free, tested.
  * Stages 2-5 (next): MLX MoonViT + Qwen2.5 backbones, assembly, processor, PBD
    generation loop, then weights + end-to-end + parity (need the ``mlx`` extra).

Only the mlx-free Stage-1 pieces are exported here so the package stays importable
without ``mlx``; the model + processor are imported lazily once they exist.
"""

from __future__ import annotations

from .config import LocateAnythingConfig, MoonViTConfig, Qwen2Config
from .convert import convert_state_dict, remap_key
from .decode import (
    GroundingItem,
    TokenScheme,
    parse_grounding_text,
    parse_grounding_tokens,
)

__all__ = [
    "LocateAnythingConfig", "MoonViTConfig", "Qwen2Config",
    "convert_state_dict", "remap_key",
    "GroundingItem", "TokenScheme", "parse_grounding_tokens", "parse_grounding_text",
]
