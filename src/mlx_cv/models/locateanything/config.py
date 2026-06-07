"""LocateAnything-3B configs (clean-room; values verified against the reference, §16).

``MoonViTConfig`` -> ``backbones/vision/moonvit``; ``Qwen2Config`` -> ``backbones/llm/qwen2``.
``LocateAnythingConfig`` binds the two and carries the grounding token ids that drive
PBD decoding (see :mod:`mlx_cv.models.locateanything.decode`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MoonViTConfig", "Qwen2Config", "LocateAnythingConfig"]


@dataclass
class MoonViTConfig:
    """MoonViT-SO-400M native-resolution vision encoder."""

    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 14
    num_channels: int = 3
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    merge_kernel_size: tuple[int, int] = (2, 2)

    @property
    def spatial_merge_size(self) -> int:
        return self.merge_kernel_size[0]


@dataclass
class Qwen2Config:
    """Qwen2.5-3B-Instruct decoder (GQA), with PBD block fields."""

    hidden_size: int = 2048
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 11008
    vocab_size: int = 152681
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True
    # Parallel Box Decoding
    block_size: int = 6
    causal_attn: bool = False


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
