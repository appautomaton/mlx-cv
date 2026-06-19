"""Faithful MLX port of the SAM 3 video tracker — memory encoder + neck (slice 8).

Mirrors ``transformers.models.sam3_tracker_video.modeling_sam3_tracker_video`` so the
real ``facebook/sam3`` ``tracker_model.memory_encoder.*`` (40) tensors load 1:1; the
``tracker_neck`` (22) reuses the detector's :class:`Sam3VisionNeck` (it is the same
``Sam3VisionNeck`` class on ``detector_config.vision_config`` upstream).

SAM2-style memory encoder: a mask down-sampler (progressive stride-2 convs), a
1x1 feature projection, a memory fuser of depthwise-conv "CX" blocks, and a final
channel projection, plus a parameter-free sine position encoding. Channels-last
(NHWC) throughout (channels-first LayerNorm over C == ``nn.LayerNorm`` on the last
axis). No torch/transformers imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .real_video_config import Sam3TrackerVideoConfig
from .real_vision import Sam3SinePositionEmbedding, rotate_pairwise

__all__ = [
    "Sam3TrackerMemoryFuserCXBlock",
    "Sam3TrackerMemoryFuser",
    "Sam3TrackerMaskDownSampler",
    "Sam3TrackerMemoryEncoder",
    "Sam3TrackerMemoryEncoderOutput",
    "Sam3TrackerFeedForward",
    "Sam3TrackerVisionRotaryEmbedding",
    "Sam3TrackerRoPEAttention",
    "Sam3TrackerMemoryAttention",
]


class Sam3TrackerMemoryFuserCXBlock(nn.Module):
    """ConvNeXt-style block: depthwise conv -> LN -> 1x1 (Linear) MLP -> layer scale."""

    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        dim = config.memory_fuser_embed_dim
        self.depthwise_conv = nn.Conv2d(
            dim, dim, kernel_size=config.memory_fuser_kernel_size, padding=config.memory_fuser_padding, groups=dim
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, config.memory_fuser_intermediate_dim)
        self.pointwise_conv2 = nn.Linear(config.memory_fuser_intermediate_dim, dim)
        self.scale = config.memory_fuser_layer_scale_init_value * mx.ones((dim,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # NHWC throughout (upstream permutes NCHW<->NHWC around the pointwise convs).
        residual = hidden_states
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.scale * hidden_states
        return residual + hidden_states


class Sam3TrackerMemoryFuser(nn.Module):
    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        self.layers = [Sam3TrackerMemoryFuserCXBlock(config) for _ in range(config.memory_fuser_num_layers)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _MaskDownSamplerLayer(nn.Module):
    def __init__(self, config: Sam3TrackerVideoConfig, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=config.mask_downsampler_kernel_size,
            stride=config.mask_downsampler_stride,
            padding=config.mask_downsampler_padding,
        )
        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.layer_norm(self.conv(x)))


class Sam3TrackerMaskDownSampler(nn.Module):
    """Progressively downsample a single-channel mask by ``total_stride`` (NHWC)."""

    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        num_layers = int(
            math.log2(config.mask_downsampler_total_stride) // math.log2(config.mask_downsampler_stride)
        )
        layers: list[nn.Module] = []
        in_chans, out_chans = 1, 1
        for _ in range(num_layers):
            out_chans = in_chans * (config.mask_downsampler_stride**2)
            layers.append(_MaskDownSamplerLayer(config, in_chans, out_chans))
            in_chans = out_chans
        self.layers = layers
        self.final_conv = nn.Conv2d(out_chans, config.mask_downsampler_embed_dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


@dataclass
class Sam3TrackerMemoryEncoderOutput:
    vision_features: mx.array  # [B, H, W, output_channels]
    vision_pos_enc: mx.array  # [B, H, W, output_channels]


class Sam3TrackerMemoryEncoder(nn.Module):
    """Fuse pixel features with a downsampled mask into a compact memory (40 tensors)."""

    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        hidden_size = config.memory_encoder_hidden_size
        output_channels = config.memory_encoder_output_channels
        self.mask_downsampler = Sam3TrackerMaskDownSampler(config)
        self.feature_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.memory_fuser = Sam3TrackerMemoryFuser(config)
        self.position_encoding = Sam3SinePositionEmbedding(output_channels // 2, normalize=True)
        self.projection = nn.Conv2d(hidden_size, output_channels, kernel_size=1)

    def __call__(self, vision_features: mx.array, masks: mx.array) -> Sam3TrackerMemoryEncoderOutput:
        # vision_features: NHWC [B, H, W, hidden]; masks: NHWC [B, H*16, W*16, 1].
        masks = self.mask_downsampler(masks)
        vision_features = self.feature_projection(vision_features)
        vision_features = vision_features + masks
        vision_features = self.memory_fuser(vision_features)
        vision_features = self.projection(vision_features)

        batch_size, height, width, _ = vision_features.shape
        vision_pos_enc = self.position_encoding(batch_size, height, width)
        return Sam3TrackerMemoryEncoderOutput(vision_features=vision_features, vision_pos_enc=vision_pos_enc)


# --- memory attention (slice 9) -----------------------------------------------


class Sam3TrackerFeedForward(nn.Module):
    """proj_in -> (ReLU -> hidden Linear)* -> proj_out (object_pointer_proj, heads)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, activation: str = "relu"):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        self._act = nn.relu if activation == "relu" else nn.gelu

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self._act(self.proj_in(hidden_states))
        for layer in self.layers:
            hidden_states = self._act(layer(hidden_states))
        return self.proj_out(hidden_states)


class Sam3TrackerVisionRotaryEmbedding(nn.Module):
    """2D axial RoPE over a fixed feature grid (parameter-free; computed in-forward)."""

    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        self.dim = config.memory_attention_hidden_size // (
            config.memory_attention_downsample_rate * config.memory_attention_num_attention_heads
        )
        if self.dim % 4 != 0:
            raise ValueError("RoPE dimension must be divisible by 4")
        self.end_x, self.end_y = config.memory_attention_rope_feat_sizes
        self.theta = config.memory_attention_rope_theta

    def __call__(self) -> tuple[mx.array, mx.array]:
        dim = self.dim
        freqs = 1.0 / (self.theta ** (mx.arange(0, dim, 4)[: dim // 4].astype(mx.float32) / dim))
        idx = mx.arange(self.end_x * self.end_y)
        x_positions = (idx % self.end_x).astype(mx.float32)
        y_positions = (idx // self.end_x).astype(mx.float32)
        inv_freq = mx.concatenate([mx.outer(x_positions, freqs), mx.outer(y_positions, freqs)], axis=-1)
        inv_freq = mx.repeat(inv_freq, 2, axis=-1)
        return mx.cos(inv_freq), mx.sin(inv_freq)


def _apply_rope_video(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, num_k_exclude_rope: int = 0, repeat_freqs_k: bool = False
) -> tuple[mx.array, mx.array]:
    n = k.shape[-2] - num_k_exclude_rope
    k_rot, k_pass = k[..., :n, :], k[..., n:, :]
    q_embed = (q * cos) + (rotate_pairwise(q) * sin)
    if k_rot.shape[-2] == 0:
        return q_embed, mx.concatenate([k_rot, k_pass], axis=-2)
    if repeat_freqs_k and k_rot.shape[-2] != q.shape[-2]:
        repeat_factor = k_rot.shape[-2] // q.shape[-2]
        cos_k = mx.tile(cos, (repeat_factor, 1))
        sin_k = mx.tile(sin, (repeat_factor, 1))
    else:
        cos_k, sin_k = cos, sin
    k_embed = (k_rot * cos_k) + (rotate_pairwise(k_rot) * sin_k)
    return q_embed, mx.concatenate([k_embed, k_pass], axis=-2)


class Sam3TrackerRoPEAttention(nn.Module):
    """Attention with rotary position encoding (self + cross, kv may be 64-dim memory)."""

    def __init__(self, config: Sam3TrackerVideoConfig, kv_in_dim: int | None = None, rope_k_repeat: bool = False):
        super().__init__()
        hidden_size = config.memory_attention_hidden_size
        internal_dim = hidden_size // config.memory_attention_downsample_rate
        self.num_heads = config.memory_attention_num_attention_heads
        self.head_dim = internal_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        kv_in = kv_in_dim if kv_in_dim is not None else hidden_size
        self.q_proj = nn.Linear(hidden_size, internal_dim)
        self.k_proj = nn.Linear(kv_in, internal_dim)
        self.v_proj = nn.Linear(kv_in, internal_dim)
        self.o_proj = nn.Linear(internal_dim, hidden_size)
        self.rope_k_repeat = rope_k_repeat

    def __call__(self, query, key, value, position_embeddings, num_k_exclude_rope: int = 0) -> mx.array:
        batch_size, point_batch_size = query.shape[:2]
        merged = batch_size * point_batch_size

        def _heads(proj, x):
            x = proj(x).reshape(merged, -1, self.num_heads, self.head_dim)
            return x.transpose(0, 2, 1, 3)

        queries = _heads(self.q_proj, query)
        keys = _heads(self.k_proj, key)
        values = _heads(self.v_proj, value)

        cos, sin = position_embeddings
        queries, keys = _apply_rope_video(
            queries, keys, cos, sin, num_k_exclude_rope=num_k_exclude_rope, repeat_freqs_k=self.rope_k_repeat
        )

        attn = (queries @ keys.transpose(0, 1, 3, 2)) * self.scaling
        attn = mx.softmax(attn, axis=-1)
        out = attn @ values
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, point_batch_size, -1, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Sam3TrackerMemoryAttentionLayer(nn.Module):
    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        hidden_size = config.memory_attention_hidden_size
        self.self_attn = Sam3TrackerRoPEAttention(config)
        self.cross_attn_image = Sam3TrackerRoPEAttention(config, kv_in_dim=64, rope_k_repeat=True)
        self.linear1 = nn.Linear(hidden_size, config.memory_attention_feed_forward_hidden_size)
        self.linear2 = nn.Linear(config.memory_attention_feed_forward_hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self._act = nn.relu if config.memory_attention_feed_forward_hidden_act == "relu" else nn.gelu

    def __call__(self, queries, keys, key_point_embedding, rope_position_embeddings, num_k_exclude_rope=0):
        query = self.layer_norm1(queries)
        queries = queries + self.self_attn(query, query, query, rope_position_embeddings)

        query = self.layer_norm2(queries)
        query = self.cross_attn_image(
            query, keys + key_point_embedding, keys, rope_position_embeddings, num_k_exclude_rope=num_k_exclude_rope
        )
        queries = queries + query

        query = self.layer_norm3(queries)
        queries = queries + self.linear2(self._act(self.linear1(query)))
        return queries


class Sam3TrackerMemoryAttention(nn.Module):
    """4-layer memory attention transformer (self + image cross-attention + RoPE, 106)."""

    def __init__(self, config: Sam3TrackerVideoConfig):
        super().__init__()
        self.layers = [Sam3TrackerMemoryAttentionLayer(config) for _ in range(config.memory_attention_num_layers)]
        self.layer_norm = nn.LayerNorm(config.memory_attention_hidden_size)
        self.rotary_emb = Sam3TrackerVisionRotaryEmbedding(config)

    def __call__(
        self,
        current_vision_features: mx.array,  # [seq, B, C]
        memory: mx.array,  # [mem_seq, B, mem_dim]
        current_vision_position_embeddings: mx.array | None = None,
        memory_position_embeddings: mx.array | None = None,
        num_object_pointer_tokens: int = 0,
    ) -> mx.array:
        output = current_vision_features
        if current_vision_position_embeddings is not None:
            output = output + 0.1 * current_vision_position_embeddings

        output = output.transpose(1, 0, 2)  # [B, seq, C]
        memory = memory.transpose(1, 0, 2)[:, None]  # [B, 1, mem_seq, mem_dim]
        memory_position_embeddings = memory_position_embeddings.transpose(1, 0, 2)[:, None]
        rope_position_embeddings = self.rotary_emb()

        for layer in self.layers:
            queries = output[:, None] if output.ndim == 3 else output
            output = layer(
                queries=queries,
                keys=memory,
                key_point_embedding=memory_position_embeddings,
                rope_position_embeddings=rope_position_embeddings,
                num_k_exclude_rope=num_object_pointer_tokens,
            )

        normed_output = self.layer_norm(output)
        return normed_output.transpose(1, 0, 2, 3)
