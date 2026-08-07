"""LocateAnything-3B grounding with MoonViT, Qwen2.5, and PBD decoding.

Configuration, conversion, and token parsing stay importable without MLX.
Compute, processing, tokenization, and pipeline classes are resolved lazily so
the package root remains runtime-light.
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
