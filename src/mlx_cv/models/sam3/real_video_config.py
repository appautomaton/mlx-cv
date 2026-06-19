"""Faithful SAM 3 video tracker config (mirrors ``Sam3TrackerVideoConfig``).

Field-for-field with ``transformers.models.sam3_tracker_video`` so the real
``facebook/sam3`` tracker tensors ingest 1:1. Pure data only (no mlx/torch). The
detector half reuses :mod:`mlx_cv.models.sam3.real_config`; this covers the SAM2-style
tracker (memory encoder/attention, prompt encoder, tracker mask decoder).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sam3TrackerPromptEncoderConfig:
    hidden_size: int = 256
    image_size: int = 1024
    patch_size: int = 16
    mask_input_channels: int = 16
    num_point_embeddings: int = 4
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    scale: float = 1.0

    @property
    def image_embedding_size(self) -> int:
        return self.image_size // self.patch_size


@dataclass(frozen=True)
class Sam3TrackerMaskDecoderConfig:
    hidden_size: int = 256
    hidden_act: str = "gelu"
    mlp_dim: int = 2048
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    attention_downsample_rate: int = 2
    num_multimask_outputs: int = 3
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256
    layer_norm_eps: float = 1e-6
    dynamic_multimask_via_stability: bool = True
    dynamic_multimask_stability_delta: float = 0.05
    dynamic_multimask_stability_thresh: float = 0.98


@dataclass(frozen=True)
class Sam3TrackerVideoConfig:
    """SAM2-style tracker config (memory encoder/attention + prompt/mask decoders)."""

    prompt_encoder: Sam3TrackerPromptEncoderConfig = Sam3TrackerPromptEncoderConfig()
    mask_decoder: Sam3TrackerMaskDecoderConfig = Sam3TrackerMaskDecoderConfig()

    initializer_range: float = 0.02
    num_maskmem: int = 7
    sigmoid_scale_for_mem_enc: float = 20.0
    sigmoid_bias_for_mem_enc: float = -10.0
    enable_occlusion_spatial_embedding: bool = True
    max_object_pointers_in_encoder: int = 16
    max_cond_frame_num: int = 4
    enable_temporal_pos_encoding_for_object_pointers: bool = True

    # Memory attention (slice 9)
    memory_attention_hidden_size: int = 256
    memory_attention_num_layers: int = 4
    memory_attention_num_attention_heads: int = 1
    memory_attention_downsample_rate: int = 1
    memory_attention_feed_forward_hidden_size: int = 2048
    memory_attention_feed_forward_hidden_act: str = "relu"
    memory_attention_rope_theta: float = 10000.0
    memory_attention_rope_feat_sizes: tuple[int, int] = (72, 72)

    # Memory encoder (slice 8)
    memory_encoder_hidden_size: int = 256
    memory_encoder_output_channels: int = 64
    mask_downsampler_embed_dim: int = 256
    mask_downsampler_kernel_size: int = 3
    mask_downsampler_stride: int = 2
    mask_downsampler_padding: int = 1
    mask_downsampler_total_stride: int = 16
    mask_downsampler_hidden_act: str = "gelu"
    memory_fuser_num_layers: int = 2
    memory_fuser_embed_dim: int = 256
    memory_fuser_intermediate_dim: int = 1024
    memory_fuser_kernel_size: int = 7
    memory_fuser_padding: int = 3
    memory_fuser_layer_scale_init_value: float = 1e-6
    memory_fuser_hidden_act: str = "gelu"
