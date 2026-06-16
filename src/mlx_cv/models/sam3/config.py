"""SAM 3.1 image-mode model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...backbones.vision.sam3 import SAM3ImageBackboneConfig
from ...heads.segmentation import SAM3DecoderConfig
from .text import SAM3TextConfig

__all__ = ["SAM3Config"]


@dataclass(frozen=True)
class SAM3Config:
    image: SAM3ImageBackboneConfig = field(default_factory=SAM3ImageBackboneConfig)
    text: SAM3TextConfig = field(default_factory=SAM3TextConfig)
    decoder: SAM3DecoderConfig = field(default_factory=SAM3DecoderConfig)

    def __post_init__(self) -> None:
        if self.image.neck_channels != self.decoder.hidden_dim:
            raise ValueError("SAM3 image neck_channels must match decoder hidden_dim")
        if self.image.text_dim != self.text.d_model:
            raise ValueError("SAM3 image text_dim must match text encoder d_model")
        if self.decoder.text_dim != self.text.d_model:
            raise ValueError("SAM3 decoder text_dim must match text encoder d_model")
