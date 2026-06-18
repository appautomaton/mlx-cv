"""LocateAnything-3B configs (clean-room; values verified against the reference, §16).

``MoonViTConfig`` -> ``backbones/vision/moonvit``; ``Qwen2Config`` -> ``backbones/llm/qwen2``.
``LocateAnythingConfig`` binds the two and carries the grounding token ids that drive
PBD decoding (see :mod:`mlx_cv.models.locateanything.decode`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...backbones.llm.qwen2.config import Qwen2Config
from ...backbones.vision.moonvit.config import MoonViTConfig

__all__ = ["MoonViTConfig", "Qwen2Config", "LocateAnythingConfig"]


@dataclass
class LocateAnythingConfig:
    """The assembled grounding VLM: MoonViT + MLP projector + Qwen2.5 + PBD tokens."""

    vision_config: MoonViTConfig = field(default_factory=MoonViTConfig)
    text_config: Qwen2Config = field(default_factory=Qwen2Config)
    vocab_size: int = 152681
    mlp_connector_layers: int = 2
    n_future_tokens: int = 6
    # grounding token ids (drive the PBD parser)
    image_token_index: int = 151665
    box_start_token_id: int = 151668
    box_end_token_id: int = 151669
    ref_start_token_id: int = 151672
    ref_end_token_id: int = 151673
    coord_start_token_id: int = 151677
    coord_end_token_id: int = 152677     # coord_end - coord_start == 1000 (the [0,1000] grid)
    none_token_id: int = 4064
    null_token_id: int = 152678
    switch_token_id: int = 152679
    text_mask_token_id: int = 151676

    def __post_init__(self) -> None:
        if self.text_config.text_mask_token_id != self.text_mask_token_id:
            raise ValueError(
                "LocateAnythingConfig.text_mask_token_id must match "
                "text_config.text_mask_token_id"
            )
