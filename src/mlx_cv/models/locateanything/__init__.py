"""LocateAnything-3B - the anchor grounding model (NVIDIA). See ARCHITECTURE.md section 16.

Build status:
  * Stage 1 (here): config, weight-convert rules, PBD output parser — mlx-free, tested.
  * Stages 2-5 (next): MLX MoonViT + Qwen2.5 backbones, assembly, processor, PBD
    generation loop, then weights + end-to-end + parity (need the ``mlx`` extra).

Config, conversion, and token parsing stay importable without ``mlx``. Concrete
model and processor classes are imported lazily so package-root imports remain
runtime-light.
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
    "LocateAnythingModel", "LocateAnythingProjector",
    "convert_state_dict", "remap_key",
    "GroundingItem", "TokenScheme", "parse_grounding_tokens", "parse_grounding_text",
]


def __getattr__(name: str):
    if name in {"LocateAnythingModel", "LocateAnythingProjector"}:
        from .modeling import LocateAnythingModel, LocateAnythingProjector

        return {
            "LocateAnythingModel": LocateAnythingModel,
            "LocateAnythingProjector": LocateAnythingProjector,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
