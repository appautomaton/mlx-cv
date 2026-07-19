"""LocateAnything-3B - the anchor grounding model (NVIDIA). See ARCHITECTURE.md section 16.

Build status:
  * mlx-free package root: config, conversion entry points, and token parser.
  * MLX lazy path: MoonViT + Qwen2.5 assembly, processor, PBD generation, local
    integration fixture, and ``predict`` wiring.

Config, conversion, and token parsing stay importable without ``mlx``. Concrete
model and processor classes are imported lazily so package-root imports remain
runtime-light.
"""

from __future__ import annotations

from .config import LocateAnythingConfig, MoonViTConfig, Qwen2Config
from .convert import (
    LOCATEANYTHING_CHECKPOINT_METADATA,
    LocateAnythingCheckpointError,
    convert_state_dict,
    load_locateanything_weights,
    remap_key,
)
from .decode import (
    GroundingItem,
    TokenScheme,
    parse_grounding_text,
    parse_grounding_tokens,
)

__all__ = [
    "LocateAnythingConfig", "MoonViTConfig", "Qwen2Config",
    "LocateAnythingModel", "LocateAnythingProjector",
    "LocateAnythingPipeline", "LocateAnythingTokenizer",
    "LocateAnythingProcessor", "LocateAnythingProcessorConfig", "LocateAnythingProcessorContext",
    "convert_state_dict", "load_locateanything_weights", "remap_key",
    "LOCATEANYTHING_CHECKPOINT_METADATA", "LocateAnythingCheckpointError",
    "PBDDecoder", "get_token_ids", "handle_pattern", "sample_block",
    "GroundingItem", "TokenScheme", "parse_grounding_tokens", "parse_grounding_text",
]


def __getattr__(name: str):
    if name == "LocateAnythingPipeline":
        from .pipeline import LocateAnythingPipeline

        return LocateAnythingPipeline
    if name == "LocateAnythingTokenizer":
        from .tokenizer import LocateAnythingTokenizer

        return LocateAnythingTokenizer
    if name in {"LocateAnythingModel", "LocateAnythingProjector"}:
        from .modeling import LocateAnythingModel, LocateAnythingProjector

        return {
            "LocateAnythingModel": LocateAnythingModel,
            "LocateAnythingProjector": LocateAnythingProjector,
        }[name]
    if name in {"PBDDecoder", "get_token_ids", "handle_pattern", "sample_block"}:
        from . import pbd

        return getattr(pbd, name)
    if name in {"LocateAnythingProcessor", "LocateAnythingProcessorConfig", "LocateAnythingProcessorContext"}:
        from . import processor

        return getattr(processor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
