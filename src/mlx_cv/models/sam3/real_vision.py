"""Faithful MLX port of the SAM 3 vision encoder (``Sam3VisionModel``).

Mirrors ``transformers.models.sam3.modeling_sam3`` module-for-module so the real
``facebook/sam3`` ``detector_model.vision_encoder.*`` tensors load 1:1 (538 keys):
a windowed/global 2D-axial-RoPE ViT backbone (``Sam3ViTModel``, 32 layers) plus an
FPN neck (``Sam3VisionNeck``, 4 levels). Parameter paths match the upstream state
dict exactly (minus the ``vision_encoder.`` prefix the converter strips).

All tensor compute is MLX-native and channels-last (NHWC); ``pixel_values`` enter
as NCHW ``[B, 3, H, W]`` (matching the transformers reference input) and are
transposed once at the patch embedding. No torch/transformers imports.

The numeric parity tap is ``Sam3VisionEncoderOutput.last_hidden_state`` (the ViT
backbone output ``[B, H*W, hidden]``), which is what ``Sam3Model.get_vision_features``
exposes as ``last_hidden_state``. See
``.agent/work/2026-06-18-sam3-real-architecture-port/PLAN.md`` (slice 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .real_config import Sam3ViTConfig, Sam3VisionConfig

__all__ = [
    "Sam3VisionEncoderOutput",
    "Sam3ViTModel",
    "Sam3VisionNeck",
    "Sam3VisionModel",
    "build_sam3_vision_real",
    "SAM31TriVisionModel",
    "SAM31TriVisionOutput",
]


# --- rotary position embedding (2D axial) -------------------------------------


def rotate_pairwise(x: mx.array) -> mx.array:
    """Pairwise rotation ``(a, b) -> (-b, a)`` over the last dim (head_dim)."""

    *lead, dim = x.shape
    x = x.reshape(*lead, dim // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    x = mx.stack([-x2, x1], axis=-1)
    return x.reshape(*lead, dim)


def apply_rotary_pos_emb_2d(
    query: mx.array, key: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    """Apply 2D RoPE to ``query``/``key`` ``[B, heads, seq, head_dim]``.

    ``cos``/``sin`` are ``[seq, head_dim]`` and broadcast over batch/heads.
    """

    q_embed = (query * cos) + (rotate_pairwise(query) * sin)
    k_embed = (key * cos) + (rotate_pairwise(key) * sin)
    return q_embed, k_embed


class Sam3ViTRotaryEmbedding(nn.Module):
    """Precomputes ``(cos, sin)`` for a fixed ``end_x``×``end_y`` grid.

    Parameter-free (upstream registers these as non-persistent buffers, so they are
    absent from the 538-key state dict); recomputed in :meth:`__call__` from scalars
    to keep the MLX parameter tree exactly 1:1 with the checkpoint.
    """

    def __init__(self, config: Sam3ViTConfig, end_x: int, end_y: int, scale: float = 1.0):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        if dim % 4 != 0:
            raise ValueError("head dimension must be divisible by 4 for axial RoPE")
        self.end_x = end_x
        self.end_y = end_y
        self.dim = dim
        self.rope_theta = config.rope_theta
        self.scale = scale

    def __call__(self) -> tuple[mx.array, mx.array]:
        dim = self.dim
        exponents = mx.arange(0, dim, 4)[: dim // 4].astype(mx.float32) / dim
        freqs = 1.0 / (self.rope_theta**exponents)

        flattened = mx.arange(self.end_x * self.end_y)
        x_positions = (flattened % self.end_x).astype(mx.float32) * self.scale
        y_positions = (flattened // self.end_x).astype(mx.float32) * self.scale
        freqs_x = mx.outer(x_positions, freqs)
        freqs_y = mx.outer(y_positions, freqs)
        inv_freq = mx.concatenate([freqs_x, freqs_y], axis=-1)
        inv_freq = mx.repeat(inv_freq, 2, axis=-1)
        return mx.cos(inv_freq), mx.sin(inv_freq)


# --- attention + MLP ----------------------------------------------------------


class Sam3ViTRoPEAttention(nn.Module):
    """Self-attention with 2D rotary position encoding (NHWC spatial input)."""

    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        hidden_size = config.hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states: mx.array, position_embeddings: tuple[mx.array, mx.array]) -> mx.array:
        batch_size, height, width, _ = hidden_states.shape
        seq_len = height * width
        shape = (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        query = self.q_proj(hidden_states).reshape(*shape).transpose(0, 2, 1, 3)
        key = self.k_proj(hidden_states).reshape(*shape).transpose(0, 2, 1, 3)
        value = self.v_proj(hidden_states).reshape(*shape).transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_2d(query, key, cos, sin)

        attn_weights = (query @ key.transpose(0, 1, 3, 2)) * self.scaling
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = attn_weights @ value  # [B, heads, seq, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, height, width, -1)
        return self.o_proj(attn_output)


class Sam3MLP(nn.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(hidden_states)))


# --- patch + position embeddings ----------------------------------------------


class Sam3ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        grid = config.pretrain_image_size // config.patch_size
        self.num_patches = grid * grid

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: NHWC [B, H, W, 3]
        embeddings = self.projection(pixel_values)  # [B, H', W', hidden]
        batch_size, height, width, hidden = embeddings.shape
        return embeddings.reshape(batch_size, height * width, hidden)


class Sam3ViTEmbeddings(nn.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.patch_embeddings = Sam3ViTPatchEmbeddings(config)
        self.position_embeddings = mx.zeros((1, self.patch_embeddings.num_patches, config.hidden_size))
        self.patch_size = config.patch_size

    def _tile_position_embeddings(self, position_embeddings: mx.array, height: int, width: int) -> mx.array:
        pretrain_size = int(round(position_embeddings.shape[1] ** 0.5))
        hidden = position_embeddings.shape[-1]
        if pretrain_size == height and pretrain_size == width:
            return position_embeddings.reshape(1, height * width, hidden)
        pos = position_embeddings.reshape(1, pretrain_size, pretrain_size, hidden)  # NHWC
        repeat_h = height // pretrain_size + 1
        repeat_w = width // pretrain_size + 1
        pos = mx.tile(pos, (1, repeat_h, repeat_w, 1))[:, :height, :width, :]
        return pos.reshape(1, height * width, hidden)

    def __call__(self, pixel_values: mx.array, height_patches: int, width_patches: int) -> mx.array:
        embeddings = self.patch_embeddings(pixel_values)
        position_embeddings = self._tile_position_embeddings(
            self.position_embeddings, height_patches, width_patches
        )
        return embeddings + position_embeddings


# --- windowing helpers (NHWC) -------------------------------------------------


def window_partition(hidden_state: mx.array, window_size: int) -> tuple[mx.array, tuple[int, int]]:
    batch_size, height, width, num_channels = hidden_state.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    if pad_height or pad_width:
        hidden_state = mx.pad(hidden_state, [(0, 0), (0, pad_height), (0, pad_width), (0, 0)])
    padded_height, padded_width = height + pad_height, width + pad_width
    hidden_state = hidden_state.reshape(
        batch_size, padded_height // window_size, window_size, padded_width // window_size, window_size, num_channels
    )
    windows = hidden_state.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, num_channels)
    return windows, (padded_height, padded_width)


def window_unpartition(
    windows: mx.array, window_size: int, pad_height_width: tuple[int, int], height_width: tuple[int, int]
) -> mx.array:
    padded_height, padded_width = pad_height_width
    height, width = height_width
    batch_size = windows.shape[0] // (padded_height * padded_width // window_size // window_size)
    hidden_state = windows.reshape(
        batch_size, padded_height // window_size, padded_width // window_size, window_size, window_size, -1
    )
    hidden_state = hidden_state.transpose(0, 1, 3, 2, 4, 5).reshape(batch_size, padded_height, padded_width, -1)
    return hidden_state[:, :height, :width, :]


# --- ViT layer + backbone -----------------------------------------------------


class Sam3ViTLayer(nn.Module):
    def __init__(self, config: Sam3ViTConfig, window_size: int = 0):
        super().__init__()
        grid = config.image_size // config.patch_size
        input_size = (grid, grid)
        rotary_input_size = input_size if window_size == 0 else (window_size, window_size)
        rotary_scale = config.window_size / rotary_input_size[0]

        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = Sam3ViTRotaryEmbedding(
            config, end_x=rotary_input_size[0], end_y=rotary_input_size[1], scale=rotary_scale
        )
        self.attention = Sam3ViTRoPEAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Sam3MLP(config)
        self.window_size = window_size

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        pad_height_width = None
        height = width = 0
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_height_width = window_partition(hidden_states, self.window_size)

        position_embeddings = self.rotary_emb()
        hidden_states = self.attention(hidden_states, position_embeddings)

        if self.window_size > 0:
            hidden_states = window_unpartition(hidden_states, self.window_size, pad_height_width, (height, width))

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Sam3ViTModel(nn.Module):
    """Windowed/global 2D-RoPE ViT backbone (``model_type='sam3_vit_model'``)."""

    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.config = config
        self.embeddings = Sam3ViTEmbeddings(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [
            Sam3ViTLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            for i in range(config.num_hidden_layers)
        ]

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: NHWC [B, H, W, 3]
        batch_size = pixel_values.shape[0]
        height = pixel_values.shape[1] // self.config.patch_size
        width = pixel_values.shape[2] // self.config.patch_size

        hidden_states = self.embeddings(pixel_values, height, width)  # [B, H*W, C]
        hidden = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(batch_size, height, width, hidden)

        hidden_states = self.layer_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states.reshape(batch_size, height * width, hidden)


# --- FPN neck -----------------------------------------------------------------


class Sam3SinePositionEmbedding(nn.Module):
    """Parameter-free sine position embedding for FPN levels (returns NHWC)."""

    def __init__(self, num_position_features: int, temperature: int = 10000, normalize: bool = True, scale: float | None = None):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_position_features = num_position_features
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def __call__(self, batch_size: int, height: int, width: int) -> mx.array:
        ones = mx.ones((batch_size, height, width), dtype=mx.float32)
        y_embed = mx.cumsum(ones, axis=1)
        x_embed = mx.cumsum(ones, axis=2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = mx.arange(self.num_position_features).astype(mx.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_position_features)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = mx.stack([mx.sin(pos_x[..., 0::2]), mx.cos(pos_x[..., 1::2])], axis=-1).reshape(
            batch_size, height, width, -1
        )
        pos_y = mx.stack([mx.sin(pos_y[..., 0::2]), mx.cos(pos_y[..., 1::2])], axis=-1).reshape(
            batch_size, height, width, -1
        )
        return mx.concatenate([pos_y, pos_x], axis=-1)  # NHWC [B, H, W, 2*num_position_features]

    def _dim_t(self) -> mx.array:
        dim_t = mx.arange(self.num_position_features).astype(mx.float32)
        return self.temperature ** (2 * (dim_t // 2) / self.num_position_features)

    def encode_1d_positions(self, x: mx.array, y: mx.array) -> tuple[mx.array, mx.array]:
        """Sine/cosine embed 1D coordinate pairs (used by the geometry box encoder)."""

        dim_t = self._dim_t()
        pos_x = (x * self.scale)[:, None] / dim_t
        pos_y = (y * self.scale)[:, None] / dim_t
        pos_x = mx.stack([mx.sin(pos_x[:, 0::2]), mx.cos(pos_x[:, 1::2])], axis=-1).reshape(x.shape[0], -1)
        pos_y = mx.stack([mx.sin(pos_y[:, 0::2]), mx.cos(pos_y[:, 1::2])], axis=-1).reshape(y.shape[0], -1)
        return pos_x, pos_y

    def encode_boxes(self, boxes: mx.array) -> mx.array:
        """Sine/cosine embed 4D boxes ``[B, Q, 4]`` -> ``[B, Q, 4*num_position_features]``."""

        dim_t = self._dim_t()
        batch_size, num_boxes, _ = boxes.shape

        def _embed(values: mx.array) -> mx.array:
            pos = (values * self.scale)[:, :, None] / dim_t
            return mx.stack([mx.sin(pos[:, :, 0::2]), mx.cos(pos[:, :, 1::2])], axis=-1).reshape(
                batch_size, num_boxes, -1
            )

        pos_x = _embed(boxes[:, :, 0])
        pos_y = _embed(boxes[:, :, 1])
        pos_w = _embed(boxes[:, :, 2])
        pos_h = _embed(boxes[:, :, 3])
        return mx.concatenate([pos_y, pos_x, pos_w, pos_h], axis=-1)


class Sam3FPNLayer(nn.Module):
    def __init__(self, in_channels: int, fpn_dim: int, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        scale_layers: list[nn.Module] = []
        if scale_factor == 4.0:
            scale_layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            scale_layers.append(nn.GELU())
            scale_layers.append(nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2))
            intermediate_channels = in_channels // 4
        elif scale_factor == 2.0:
            scale_layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            intermediate_channels = in_channels // 2
        elif scale_factor == 1.0:
            intermediate_channels = in_channels
        elif scale_factor == 0.5:
            scale_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            intermediate_channels = in_channels
        else:
            raise NotImplementedError(f"scale_factor={scale_factor} is not supported")
        self.scale_layers = scale_layers
        self.proj1 = nn.Conv2d(intermediate_channels, fpn_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.scale_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.proj1(hidden_states)
        return self.proj2(hidden_states)


class Sam3VisionNeck(nn.Module):
    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.position_encoding = Sam3SinePositionEmbedding(
            num_position_features=config.fpn_hidden_size // 2, normalize=True
        )
        self.fpn_layers = [
            Sam3FPNLayer(config.backbone.hidden_size, config.fpn_hidden_size, scale)
            for scale in config.scale_factors
        ]

    def __call__(self, hidden_states: mx.array) -> tuple[tuple[mx.array, ...], tuple[mx.array, ...]]:
        fpn_hidden_states: list[mx.array] = []
        fpn_position_encoding: list[mx.array] = []
        for fpn_layer in self.fpn_layers:
            fpn_output = fpn_layer(hidden_states)
            fpn_hidden_states.append(fpn_output)
            batch_size, height, width, _ = fpn_output.shape
            fpn_position_encoding.append(self.position_encoding(batch_size, height, width))
        return tuple(fpn_hidden_states), tuple(fpn_position_encoding)


# --- top-level vision model ---------------------------------------------------


@dataclass
class Sam3VisionEncoderOutput:
    last_hidden_state: mx.array
    fpn_hidden_states: tuple[mx.array, ...]
    fpn_position_encoding: tuple[mx.array, ...]


class Sam3VisionModel(nn.Module):
    """Faithful SAM 3 vision encoder: ViT backbone + FPN neck (538 tensors)."""

    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.config = config
        self.backbone = Sam3ViTModel(config.backbone)
        self.neck = Sam3VisionNeck(config)

    def __call__(self, pixel_values: mx.array) -> Sam3VisionEncoderOutput:
        # pixel_values: NCHW [B, 3, H, W] (matches the transformers reference input)
        nhwc = pixel_values.transpose(0, 2, 3, 1)
        last_hidden_state = self.backbone(nhwc)  # [B, H*W, hidden]

        batch_size = last_hidden_state.shape[0]
        height = pixel_values.shape[-2] // self.config.backbone.patch_size
        width = pixel_values.shape[-1] // self.config.backbone.patch_size
        hidden = last_hidden_state.shape[-1]
        spatial = last_hidden_state.reshape(batch_size, height, width, hidden)  # NHWC

        fpn_hidden_states, fpn_position_encoding = self.neck(spatial)
        return Sam3VisionEncoderOutput(
            last_hidden_state=last_hidden_state,
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
        )


def build_sam3_vision_real(config: Sam3VisionConfig) -> Sam3VisionModel:
    return Sam3VisionModel(config)


@dataclass
class SAM31TriVisionOutput:
    """Shared SAM 3.1 trunk with detector, interactive, and propagation heads."""

    last_hidden_state: mx.array
    fpn_hidden_states: tuple[mx.array, ...]
    fpn_position_encoding: tuple[mx.array, ...]
    interactive_hidden_states: tuple[mx.array, ...]
    interactive_position_encoding: tuple[mx.array, ...]
    propagation_hidden_states: tuple[mx.array, ...]
    propagation_position_encoding: tuple[mx.array, ...]


class SAM31TriVisionModel(nn.Module):
    """Official SAM 3.1 TriHead vision backbone.

    The ViT trunk is shared. Each head owns the three 4x/2x/1x FPN projections
    shipped as ``convs``, ``interactive_convs``, and ``propagation_convs`` in
    the merged checkpoint.
    """

    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        if tuple(config.scale_factors) != (4.0, 2.0, 1.0):
            raise ValueError(
                "SAM 3.1 TriHead vision requires scale_factors=(4.0, 2.0, 1.0)"
            )
        self.config = config
        self.backbone = Sam3ViTModel(config.backbone)
        self.neck = Sam3VisionNeck(config)
        self.interactive_neck = Sam3VisionNeck(config)
        self.propagation_neck = Sam3VisionNeck(config)

    def __call__(self, pixel_values: mx.array) -> SAM31TriVisionOutput:
        nhwc = pixel_values.transpose(0, 2, 3, 1)
        last_hidden_state = self.backbone(nhwc)
        batch_size = last_hidden_state.shape[0]
        height = pixel_values.shape[-2] // self.config.backbone.patch_size
        width = pixel_values.shape[-1] // self.config.backbone.patch_size
        hidden = last_hidden_state.shape[-1]
        spatial = last_hidden_state.reshape(batch_size, height, width, hidden)

        detector, detector_pos = self.neck(spatial)
        interactive, interactive_pos = self.interactive_neck(spatial)
        propagation, propagation_pos = self.propagation_neck(spatial)
        return SAM31TriVisionOutput(
            last_hidden_state=last_hidden_state,
            fpn_hidden_states=detector,
            fpn_position_encoding=detector_pos,
            interactive_hidden_states=interactive,
            interactive_position_encoding=interactive_pos,
            propagation_hidden_states=propagation,
            propagation_position_encoding=propagation_pos,
        )
