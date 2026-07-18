"""Official SAM 3.1 16-slot multiplex tracker modules for MLX."""

from __future__ import annotations

import math
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from .real_tracker_decoder import (
    Sam3TrackerMaskDecoder,
    Sam3TrackerMaskEmbedding,
)
from .real_video import Sam3TrackerFeedForward
from .real_video_config import (
    Sam3TrackerMaskDecoderConfig,
    Sam3TrackerPromptEncoderConfig,
)
from .real_vision import Sam3SinePositionEmbedding, rotate_pairwise

__all__ = ["SAM31MultiplexTracker", "SAM31MultiplexMaskDecoder"]


class _PositionEmbeddingRandom(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.positional_encoding_gaussian_matrix = mx.zeros((2, hidden_size // 2))

    def __call__(self, coords: mx.array) -> mx.array:
        coords = 2.0 * coords - 1.0
        angles = 2.0 * math.pi * (
            coords @ self.positional_encoding_gaussian_matrix
        )
        return mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)

    def dense(self, height: int, width: int) -> mx.array:
        ys = (mx.arange(height, dtype=mx.float32) + 0.5) / height
        xs = (mx.arange(width, dtype=mx.float32) + 0.5) / width
        grid = mx.stack(
            [
                mx.broadcast_to(xs[None], (height, width)),
                mx.broadcast_to(ys[:, None], (height, width)),
            ],
            axis=-1,
        )
        return self(grid)[None]


class _InteractivePromptEncoder(nn.Module):
    """SAM prompt encoder retaining the official four separate embeddings."""

    def __init__(self, config: Sam3TrackerPromptEncoderConfig):
        super().__init__()
        self.pe_layer = _PositionEmbeddingRandom(config.hidden_size)
        self.point_embeddings = [
            nn.Embedding(1, config.hidden_size)
            for _ in range(config.num_point_embeddings)
        ]
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)
        self.mask_downscaling = Sam3TrackerMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)
        self.image_size = config.image_size
        self.image_embedding_size = config.image_embedding_size

    def encode_sparse(self, coords: mx.array, labels: mx.array) -> mx.array:
        normalized = (coords + 0.5) / float(self.image_size)
        embeddings = self.pe_layer(normalized)
        labels_expanded = labels[..., None]
        embeddings = mx.where(
            labels_expanded == -1,
            mx.broadcast_to(self.not_a_point_embed.weight, embeddings.shape),
            embeddings,
        )
        for label, learned in enumerate(self.point_embeddings):
            embeddings = mx.where(
                labels_expanded == label,
                embeddings + learned.weight[0],
                embeddings,
            )
        return embeddings

    def encode_dense(self, masks: mx.array | None, batch_size: int) -> mx.array:
        if masks is not None:
            return self.mask_downscaling(masks)
        dense = self.no_mask_embed.weight.reshape(1, 1, 1, -1)
        size = self.image_embedding_size
        return mx.broadcast_to(dense, (batch_size, size, size, dense.shape[-1]))

    def get_dense_pe(self) -> mx.array:
        size = self.image_embedding_size
        return self.pe_layer.dense(size, size)


class _CXBlock(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.gamma = mx.ones((dim,)) * 1e-6
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.pwconv2 = nn.Linear(dim * 4, dim)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv2(nn.gelu(self.pwconv1(x)))
        return residual + self.gamma * x


class _Fuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_CXBlock(), _CXBlock()]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class _MaskDownsampler(nn.Module):
    """32 multiplex mask channels -> stride-16 256-channel embedding."""

    def __init__(self):
        super().__init__()
        channels = ((32, 16), (16, 64), (64, 256), (256, 1024))
        encoder: list[nn.Module] = []
        for input_channels, output_channels in channels:
            encoder.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.LayerNorm(output_channels, eps=1e-6),
                    nn.GELU(),
                ]
            )
        encoder.append(nn.Conv2d(1024, 256, kernel_size=1))
        self.encoder = encoder

    def __call__(self, masks: mx.array) -> mx.array:
        if masks.shape[1:3] != (1152, 1152):
            masks = _resize_bilinear(masks, 1152, 1152)
        for layer in self.encoder:
            masks = layer(masks)
        return masks


def _resize_axis_half_pixel(x: mx.array, out_size: int, axis: int) -> mx.array:
    in_size = int(x.shape[axis])
    if in_size == out_size:
        return x
    coordinates = (mx.arange(out_size, dtype=mx.float32) + 0.5) * (
        in_size / out_size
    ) - 0.5
    lower_raw = mx.floor(coordinates).astype(mx.int32)
    upper_raw = lower_raw + 1
    lower = mx.clip(lower_raw, 0, in_size - 1)
    upper = mx.clip(upper_raw, 0, in_size - 1)
    weight = coordinates - lower_raw.astype(mx.float32)
    left = mx.take(x, lower, axis=axis)
    right = mx.take(x, upper, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = out_size
    weight = weight.reshape(shape)
    return left * (1.0 - weight) + right * weight


def _resize_bilinear(
    x: mx.array, out_height: int, out_width: int
) -> mx.array:
    x = _resize_axis_half_pixel(x, out_height, axis=1)
    return _resize_axis_half_pixel(x, out_width, axis=2)


class _MaskMemoryBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_downsampler = _MaskDownsampler()
        self.pix_feat_proj = nn.Conv2d(256, 256, kernel_size=1)
        self.fuser = _Fuser()
        self.position_encoding = Sam3SinePositionEmbedding(128, normalize=True)

    def __call__(self, pixel_features: mx.array, masks: mx.array):
        memory = self.pix_feat_proj(pixel_features) + self.mask_downsampler(masks)
        memory = self.fuser(memory)
        batch, height, width, _ = memory.shape
        return memory, self.position_encoding(batch, height, width)


def _rope_frequencies(seq_len: int, head_dim: int = 32) -> tuple[mx.array, mx.array]:
    side = int(math.sqrt(seq_len))
    if side * side != seq_len:
        raise ValueError(f"SAM 3.1 tracker RoPE requires a square grid, got {seq_len}")
    frequency = 1.0 / (
        10000.0
        ** (mx.arange(0, head_dim, 4)[: head_dim // 4].astype(mx.float32) / head_dim)
    )
    index = mx.arange(seq_len)
    x = (index % side).astype(mx.float32)
    y = (index // side).astype(mx.float32)
    angles = mx.concatenate([mx.outer(x, frequency), mx.outer(y, frequency)], axis=-1)
    angles = mx.repeat(angles, 2, axis=-1)
    return mx.cos(angles), mx.sin(angles)


def _attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    num_heads: int = 8,
    rope: bool,
    repeat_key_rope: bool = False,
    num_key_tokens_without_rope: int = 0,
) -> mx.array:
    batch, query_length, channels = query.shape
    key_length = key.shape[1]
    head_dim = channels // num_heads
    query = query.reshape(batch, query_length, num_heads, head_dim).transpose(0, 2, 1, 3)
    key = key.reshape(key.shape[0], key_length, num_heads, head_dim).transpose(0, 2, 1, 3)
    value = value.reshape(value.shape[0], key_length, num_heads, head_dim).transpose(0, 2, 1, 3)
    if rope:
        cos, sin = _rope_frequencies(query_length, head_dim)
        query = query * cos + rotate_pairwise(query) * sin
        rotate_length = key_length - num_key_tokens_without_rope
        key_rotated = key[..., :rotate_length, :]
        if repeat_key_rope:
            repeats = rotate_length // query_length
            key_cos = mx.tile(cos, (repeats, 1))
            key_sin = mx.tile(sin, (repeats, 1))
        else:
            key_cos, key_sin = cos, sin
        key_rotated = key_rotated * key_cos + rotate_pairwise(key_rotated) * key_sin
        key = mx.concatenate([key_rotated, key[..., rotate_length:, :]], axis=-2)
    output = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=head_dim**-0.5
    )
    return output.transpose(0, 2, 1, 3).reshape(batch, query_length, channels)


class _DecoupledLayer(nn.Module):
    def __init__(self):
        super().__init__()
        for stem in (
            "self_attn_q_proj",
            "self_attn_k_proj",
            "self_attn_v_proj",
            "self_attn_out_proj",
            "cross_attn_q_proj",
            "cross_attn_k_proj",
            "cross_attn_v_proj",
            "cross_attn_out_proj",
            "image_cross_attn_q_proj",
            "image_cross_attn_k_proj",
        ):
            setattr(self, stem, nn.Linear(256, 256))
        self.linear1 = nn.Linear(256, 2048)
        self.linear2 = nn.Linear(2048, 256)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)

    def __call__(
        self,
        image: mx.array,
        objects: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        memory_image_pos: mx.array | None,
        num_object_pointer_tokens: int,
    ) -> mx.array:
        normalized = self.norm1(objects)
        query = self.self_attn_q_proj(normalized)
        key = self.self_attn_k_proj(normalized)
        value = self.self_attn_v_proj(normalized)
        objects = objects + self.self_attn_out_proj(
            _attention(query, key, value, rope=True)
        )

        normalized = self.norm2(objects)
        query = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(normalized)
        key = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        if memory_image_pos is not None:
            key = key + memory_image_pos
        value = self.cross_attn_v_proj(memory)
        objects = objects + self.cross_attn_out_proj(
            _attention(
                query,
                key,
                value,
                rope=True,
                repeat_key_rope=True,
                num_key_tokens_without_rope=num_object_pointer_tokens,
            )
        )
        normalized = self.norm3(objects)
        return objects + self.linear2(nn.gelu(self.linear1(normalized)))


class _DecoupledEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_DecoupledLayer() for _ in range(4)]
        self.norm = nn.LayerNorm(256)

    def __call__(
        self,
        image: mx.array,
        objects: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        object_pos: mx.array | None = None,
        memory_image_pos: mx.array | None = None,
        num_object_pointer_tokens: int = 0,
    ) -> mx.array:
        if object_pos is not None:
            objects = objects + 0.1 * object_pos
        for layer in self.layers:
            objects = layer(
                image,
                objects,
                memory_image,
                memory,
                memory_image_pos,
                num_object_pointer_tokens,
            )
        return self.norm(objects)


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _DecoupledEncoder()


class SAM31MultiplexMaskDecoder(Sam3TrackerMaskDecoder):
    """Three masks per object across one 16-slot multiplex bucket."""

    def __init__(self, config: Sam3TrackerMaskDecoderConfig):
        config = replace(config, num_multimask_outputs=2)
        super().__init__(config)
        self.iou_token = nn.Embedding(16, config.hidden_size)
        self.obj_score_token = nn.Embedding(16, config.hidden_size)
        self.mask_tokens = nn.Embedding(48, config.hidden_size)

    def __call__(
        self,
        image_embeddings: mx.array,
        image_positional_embeddings: mx.array,
        high_resolution_features: list[mx.array],
        extra_per_object_embeddings: mx.array | None = None,
        multimask_output: bool = True,
    ):
        """Run the official shared-image, 16-object propagation decoder."""

        batch, height, width, channels = image_embeddings.shape
        if image_positional_embeddings.shape[0] == 1 and batch != 1:
            image_positional_embeddings = mx.repeat(
                image_positional_embeddings, batch, axis=0
            )
        mask_tokens = self.mask_tokens.weight.reshape(16, 3, channels)
        if extra_per_object_embeddings is not None:
            mask_tokens = mask_tokens[None] + extra_per_object_embeddings[:, :, None]
        else:
            mask_tokens = mx.broadcast_to(mask_tokens[None], (batch, 16, 3, channels))
        tokens = mx.concatenate(
            [
                mx.broadcast_to(self.obj_score_token.weight[None], (batch, 16, channels)),
                mx.broadcast_to(self.iou_token.weight[None], (batch, 16, channels)),
                mask_tokens.reshape(batch, 48, channels),
            ],
            axis=1,
        )

        token_output, keys = self.transformer(
            tokens[:, None], image_embeddings, image_positional_embeddings
        )
        token_output = token_output[:, 0]
        keys = keys[:, 0].reshape(batch, height, width, channels)
        object_tokens = token_output[:, :16]
        iou_tokens = token_output[:, 16:32]
        mask_tokens_out = token_output[:, 32:].reshape(batch, 16, 3, channels)

        feat_s0, feat_s1 = high_resolution_features
        upscaled = nn.gelu(
            self.upscale_layer_norm(self.upscale_conv1(keys) + feat_s1)
        )
        upscaled = nn.gelu(self.upscale_conv2(upscaled) + feat_s0)
        hyper = mx.stack(
            [
                self.output_hypernetworks_mlps[index](mask_tokens_out[:, :, index])
                for index in range(3)
            ],
            axis=2,
        )
        _, out_height, out_width, out_channels = upscaled.shape
        pixels = upscaled.reshape(batch, out_height * out_width, out_channels)
        masks = (hyper @ pixels.transpose(0, 2, 1)).reshape(
            batch, 16, 3, out_height, out_width
        )
        iou_pred = self.iou_prediction_head(iou_tokens).reshape(batch, 16, 3)
        object_score_logits = self.pred_obj_score_head(object_tokens)
        if not multimask_output:
            masks = masks[:, :, :1]
            iou_pred = iou_pred[:, :, :1]
            mask_tokens_out = mask_tokens_out[:, :, :1]
        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "object_score_logits": object_score_logits,
            "sam_tokens_out": mask_tokens_out,
        }


class SAM31MultiplexTracker(nn.Module):
    """Complete 457-source-tensor SAM 3.1 tracker parameter assembly."""

    def __init__(self):
        super().__init__()
        prompt_config = Sam3TrackerPromptEncoderConfig(
            image_size=1008,
            patch_size=14,
            layer_norm_eps=1e-6,
        )
        decoder_config = Sam3TrackerMaskDecoderConfig(
            layer_norm_eps=1e-6, iou_prediction_use_sigmoid=False
        )
        self.maskmem_tpos_enc = mx.zeros((7, 1, 1, 256))
        self.interactivity_no_mem_embed = mx.zeros((1, 1, 256))
        self.no_obj_embed_spatial = mx.zeros((16, 256))
        self.output_valid_embed = mx.zeros((16, 256))
        self.output_invalid_embed = mx.zeros((16, 256))
        self.transformer = _Transformer()
        self.maskmem_backbone = _MaskMemoryBackbone()
        self.interactive_mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.no_obj_ptr_linear = nn.Linear(256, 256)
        self.image_pe_layer = _PositionEmbeddingRandom(256)
        self.interactive_sam_prompt_encoder = _InteractivePromptEncoder(prompt_config)
        self.interactive_sam_mask_decoder = Sam3TrackerMaskDecoder(decoder_config)
        self.sam_mask_decoder = SAM31MultiplexMaskDecoder(decoder_config)
        self.obj_ptr_proj = Sam3TrackerFeedForward(256, 256, 256, 3)
        self.interactive_obj_ptr_proj = Sam3TrackerFeedForward(256, 256, 256, 3)
        self.obj_ptr_tpos_proj = nn.Linear(256, 256)
