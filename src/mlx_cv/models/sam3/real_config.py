"""Faithful SAM 3 detector/tracker configs mirroring the ``transformers`` schema.

These dataclasses mirror ``transformers.models.sam3.configuration_sam3.Sam3Config``
(and its ``Sam3VideoConfig`` wrapper) field-for-field so the real ``facebook/sam3``
checkpoint can be ingested 1:1. They are intentionally separate from the reduced
clean-room :mod:`mlx_cv.models.sam3.config` so the existing image/video path keeps
working until the faithful modules reach upstream parity (see
``.agent/work/2026-06-18-sam3-real-architecture-port/PLAN.md``).

Pure data only — no ``mlx``/``torch`` imports — so config ingestion stays cheap and
import-light.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "Sam3ViTConfig",
    "Sam3VisionConfig",
    "Sam3TextConfig",
    "Sam3GeometryEncoderConfig",
    "Sam3DETREncoderConfig",
    "Sam3DETRDecoderConfig",
    "Sam3MaskDecoderConfig",
    "Sam3DetectorConfig",
    "from_hf_config",
    "from_hf_config_file",
]


@dataclass(frozen=True)
class Sam3ViTConfig:
    """Windowed-attention RoPE ViT backbone (``model_type='sam3_vit_model'``)."""

    hidden_size: int = 1024
    intermediate_size: int = 4736
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 1008
    patch_size: int = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0
    window_size: int = 24
    global_attn_indexes: tuple[int, ...] = (7, 15, 23, 31)
    layer_scale_init_value: float | None = None
    pretrain_image_size: int = 336
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02


@dataclass(frozen=True)
class Sam3VisionConfig:
    """FPN over the ViT backbone (``model_type='sam3_vision_model'``)."""

    backbone: Sam3ViTConfig = field(default_factory=Sam3ViTConfig)
    fpn_hidden_size: int = 256
    backbone_feature_sizes: tuple[tuple[int, int], ...] = ((288, 288), (144, 144), (72, 72))
    scale_factors: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    @property
    def image_size(self) -> int:
        return self.backbone.image_size


@dataclass(frozen=True)
class Sam3TextConfig:
    """CLIP text tower subset (``model_type='clip_text_model'``)."""

    vocab_size: int = 49408
    hidden_size: int = 1024
    intermediate_size: int = 4096
    projection_dim: int = 512
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    max_position_embeddings: int = 32
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0


@dataclass(frozen=True)
class Sam3GeometryEncoderConfig:
    """ROI geometry encoder (``model_type='sam3_geometry_encoder'``)."""

    hidden_size: int = 256
    num_layers: int = 3
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    roi_size: int = 7
    initializer_range: float = 0.02


@dataclass(frozen=True)
class Sam3DETREncoderConfig:
    """DETR transformer encoder (``model_type='sam3_detr_encoder'``)."""

    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02


@dataclass(frozen=True)
class Sam3DETRDecoderConfig:
    """DETR transformer decoder (``model_type='sam3_detr_decoder'``)."""

    hidden_size: int = 256
    num_layers: int = 6
    num_queries: int = 200
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02


@dataclass(frozen=True)
class Sam3MaskDecoderConfig:
    """FPN pixel/mask decoder (``model_type='sam3_mask_decoder'``)."""

    hidden_size: int = 256
    num_upsampling_stages: int = 3
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    num_attention_heads: int = 8
    initializer_range: float = 0.02


@dataclass(frozen=True)
class Sam3DetectorConfig:
    """Faithful SAM 3 image detector config (mirrors ``transformers.Sam3Config``)."""

    vision: Sam3VisionConfig = field(default_factory=Sam3VisionConfig)
    text: Sam3TextConfig = field(default_factory=Sam3TextConfig)
    geometry_encoder: Sam3GeometryEncoderConfig = field(default_factory=Sam3GeometryEncoderConfig)
    detr_encoder: Sam3DETREncoderConfig = field(default_factory=Sam3DETREncoderConfig)
    detr_decoder: Sam3DETRDecoderConfig = field(default_factory=Sam3DETRDecoderConfig)
    mask_decoder: Sam3MaskDecoderConfig = field(default_factory=Sam3MaskDecoderConfig)
    initializer_range: float = 0.02

    @property
    def image_size(self) -> int:
        return self.vision.image_size


# --- HF config.json ingestion -------------------------------------------------


def _int(data: Mapping[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    return int(value) if value is not None else default


def _float(data: Mapping[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    return float(value) if value is not None else default


def _str(data: Mapping[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    return str(value) if value is not None else default


def _int_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(int(v) for v in value)


def _build_vit(data: Mapping[str, Any]) -> Sam3ViTConfig:
    defaults = Sam3ViTConfig()
    layer_scale = data.get("layer_scale_init_value", defaults.layer_scale_init_value)
    return Sam3ViTConfig(
        hidden_size=_int(data, "hidden_size", defaults.hidden_size),
        intermediate_size=_int(data, "intermediate_size", defaults.intermediate_size),
        num_hidden_layers=_int(data, "num_hidden_layers", defaults.num_hidden_layers),
        num_attention_heads=_int(data, "num_attention_heads", defaults.num_attention_heads),
        num_channels=_int(data, "num_channels", defaults.num_channels),
        image_size=_int(data, "image_size", defaults.image_size),
        patch_size=_int(data, "patch_size", defaults.patch_size),
        hidden_act=_str(data, "hidden_act", defaults.hidden_act),
        layer_norm_eps=_float(data, "layer_norm_eps", defaults.layer_norm_eps),
        attention_dropout=_float(data, "attention_dropout", defaults.attention_dropout),
        rope_theta=_float(data, "rope_theta", defaults.rope_theta),
        window_size=_int(data, "window_size", defaults.window_size),
        global_attn_indexes=_int_tuple(data.get("global_attn_indexes"), defaults.global_attn_indexes),
        layer_scale_init_value=None if layer_scale is None else float(layer_scale),
        pretrain_image_size=_int(data, "pretrain_image_size", defaults.pretrain_image_size),
        hidden_dropout=_float(data, "hidden_dropout", defaults.hidden_dropout),
        initializer_range=_float(data, "initializer_range", defaults.initializer_range),
    )


def _build_vision(data: Mapping[str, Any]) -> Sam3VisionConfig:
    defaults = Sam3VisionConfig()
    backbone_data = data.get("backbone_config") or {}
    feature_sizes = data.get("backbone_feature_sizes")
    scale_factors = data.get("scale_factors")
    return Sam3VisionConfig(
        backbone=_build_vit(backbone_data),
        fpn_hidden_size=_int(data, "fpn_hidden_size", defaults.fpn_hidden_size),
        backbone_feature_sizes=(
            defaults.backbone_feature_sizes
            if feature_sizes is None
            else tuple((int(h), int(w)) for h, w in feature_sizes)
        ),
        scale_factors=(
            defaults.scale_factors if scale_factors is None else tuple(float(s) for s in scale_factors)
        ),
        hidden_act=_str(data, "hidden_act", defaults.hidden_act),
        layer_norm_eps=_float(data, "layer_norm_eps", defaults.layer_norm_eps),
        initializer_range=_float(data, "initializer_range", defaults.initializer_range),
    )


def _build_text(data: Mapping[str, Any]) -> Sam3TextConfig:
    defaults = Sam3TextConfig()
    return Sam3TextConfig(
        vocab_size=_int(data, "vocab_size", defaults.vocab_size),
        hidden_size=_int(data, "hidden_size", defaults.hidden_size),
        intermediate_size=_int(data, "intermediate_size", defaults.intermediate_size),
        projection_dim=_int(data, "projection_dim", defaults.projection_dim),
        num_hidden_layers=_int(data, "num_hidden_layers", defaults.num_hidden_layers),
        num_attention_heads=_int(data, "num_attention_heads", defaults.num_attention_heads),
        max_position_embeddings=_int(data, "max_position_embeddings", defaults.max_position_embeddings),
        hidden_act=_str(data, "hidden_act", defaults.hidden_act),
        layer_norm_eps=_float(data, "layer_norm_eps", defaults.layer_norm_eps),
        attention_dropout=_float(data, "attention_dropout", defaults.attention_dropout),
    )


def _build_geometry(data: Mapping[str, Any]) -> Sam3GeometryEncoderConfig:
    d = Sam3GeometryEncoderConfig()
    return Sam3GeometryEncoderConfig(
        hidden_size=_int(data, "hidden_size", d.hidden_size),
        num_layers=_int(data, "num_layers", d.num_layers),
        num_attention_heads=_int(data, "num_attention_heads", d.num_attention_heads),
        intermediate_size=_int(data, "intermediate_size", d.intermediate_size),
        dropout=_float(data, "dropout", d.dropout),
        hidden_act=_str(data, "hidden_act", d.hidden_act),
        hidden_dropout=_float(data, "hidden_dropout", d.hidden_dropout),
        layer_norm_eps=_float(data, "layer_norm_eps", d.layer_norm_eps),
        roi_size=_int(data, "roi_size", d.roi_size),
        initializer_range=_float(data, "initializer_range", d.initializer_range),
    )


def _build_detr_encoder(data: Mapping[str, Any]) -> Sam3DETREncoderConfig:
    d = Sam3DETREncoderConfig()
    return Sam3DETREncoderConfig(
        hidden_size=_int(data, "hidden_size", d.hidden_size),
        num_layers=_int(data, "num_layers", d.num_layers),
        num_attention_heads=_int(data, "num_attention_heads", d.num_attention_heads),
        intermediate_size=_int(data, "intermediate_size", d.intermediate_size),
        dropout=_float(data, "dropout", d.dropout),
        hidden_act=_str(data, "hidden_act", d.hidden_act),
        hidden_dropout=_float(data, "hidden_dropout", d.hidden_dropout),
        layer_norm_eps=_float(data, "layer_norm_eps", d.layer_norm_eps),
        initializer_range=_float(data, "initializer_range", d.initializer_range),
    )


def _build_detr_decoder(data: Mapping[str, Any]) -> Sam3DETRDecoderConfig:
    d = Sam3DETRDecoderConfig()
    return Sam3DETRDecoderConfig(
        hidden_size=_int(data, "hidden_size", d.hidden_size),
        num_layers=_int(data, "num_layers", d.num_layers),
        num_queries=_int(data, "num_queries", d.num_queries),
        num_attention_heads=_int(data, "num_attention_heads", d.num_attention_heads),
        intermediate_size=_int(data, "intermediate_size", d.intermediate_size),
        dropout=_float(data, "dropout", d.dropout),
        hidden_act=_str(data, "hidden_act", d.hidden_act),
        hidden_dropout=_float(data, "hidden_dropout", d.hidden_dropout),
        layer_norm_eps=_float(data, "layer_norm_eps", d.layer_norm_eps),
        initializer_range=_float(data, "initializer_range", d.initializer_range),
    )


def _build_mask_decoder(data: Mapping[str, Any]) -> Sam3MaskDecoderConfig:
    d = Sam3MaskDecoderConfig()
    return Sam3MaskDecoderConfig(
        hidden_size=_int(data, "hidden_size", d.hidden_size),
        num_upsampling_stages=_int(data, "num_upsampling_stages", d.num_upsampling_stages),
        layer_norm_eps=_float(data, "layer_norm_eps", d.layer_norm_eps),
        dropout=_float(data, "dropout", d.dropout),
        num_attention_heads=_int(data, "num_attention_heads", d.num_attention_heads),
        initializer_range=_float(data, "initializer_range", d.initializer_range),
    )


def from_hf_config(config: Mapping[str, Any]) -> Sam3DetectorConfig:
    """Build a faithful detector config from a ``facebook/sam3`` ``config.json``.

    Accepts the ``sam3_video`` wrapper (``{"detector_config": {...}, ...}``), a bare
    ``sam3`` detector config (``{"vision_config": {...}, ...}``), or a raw
    ``detector_config`` dict. Unknown keys are ignored; missing keys fall back to the
    architecture defaults.
    """

    if not isinstance(config, Mapping):
        raise TypeError(f"SAM3 config must be a mapping, got {type(config).__name__}")

    detector = config.get("detector_config", config)
    if not isinstance(detector, Mapping) or "vision_config" not in detector:
        raise ValueError(
            "SAM3 config is missing a detector with 'vision_config'; "
            "expected a facebook/sam3 config.json (sam3 or sam3_video)"
        )

    return Sam3DetectorConfig(
        vision=_build_vision(detector.get("vision_config") or {}),
        text=_build_text(detector.get("text_config") or {}),
        geometry_encoder=_build_geometry(detector.get("geometry_encoder_config") or {}),
        detr_encoder=_build_detr_encoder(detector.get("detr_encoder_config") or {}),
        detr_decoder=_build_detr_decoder(detector.get("detr_decoder_config") or {}),
        mask_decoder=_build_mask_decoder(detector.get("mask_decoder_config") or {}),
        initializer_range=_float(detector, "initializer_range", Sam3DetectorConfig().initializer_range),
    )


def from_hf_config_file(path: str | Path) -> Sam3DetectorConfig:
    """Load and ingest a ``config.json`` from disk."""

    return from_hf_config(json.loads(Path(path).read_text()))
