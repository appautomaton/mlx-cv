"""Faithful MLX port of the SAM 3 tracker prompt encoder + SAM2 mask decoder (slice 10).

Mirrors ``transformers.models.sam3_tracker_video`` so the real ``facebook/sam3``
``tracker_model.prompt_encoder.*`` (14) and ``tracker_model.mask_decoder.*`` (131)
tensors load 1:1. Channels-last (NHWC); channels-first LayerNorm over C maps to
``nn.LayerNorm`` on the last axis. ``upscale_conv*`` are ConvTranspose2d. No
torch/transformers imports.

The forwards mirror the SAM2 prompt encoder / TwoWay-transformer mask decoder. They
are driven by the per-frame streaming loop (assembled + numerically gated
out-of-sandbox in slice 11); here they are structurally 1:1 and shape-verified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .real_video import Sam3TrackerFeedForward
from .real_video_config import Sam3TrackerMaskDecoderConfig, Sam3TrackerPromptEncoderConfig

__all__ = [
    "Sam3TrackerPositionalEmbedding",
    "Sam3TrackerMaskEmbedding",
    "Sam3TrackerPromptEncoder",
    "Sam3TrackerAttention",
    "Sam3TrackerTwoWayTransformer",
    "Sam3TrackerMaskDecoder",
    "Sam3TrackerMaskDecoderOutput",
]


class Sam3TrackerPositionalEmbedding(nn.Module):
    """Random-Fourier positional embedding (``shared_embedding``)."""

    def __init__(self, config: Sam3TrackerPromptEncoderConfig):
        super().__init__()
        self.scale = config.scale
        self.positional_embedding = self.scale * mx.zeros((2, config.hidden_size // 2))

    def __call__(self, input_coords: mx.array, input_shape: tuple[int, int] | None = None) -> mx.array:
        coords = input_coords
        if input_shape is not None:
            x = coords[..., 0] / input_shape[1]
            y = coords[..., 1] / input_shape[0]
            coords = mx.stack([x, y], axis=-1)
        coords = 2 * coords - 1
        coords = coords @ self.positional_embedding
        coords = 2 * math.pi * coords
        return mx.concatenate([mx.sin(coords), mx.cos(coords)], axis=-1)


class Sam3TrackerMaskEmbedding(nn.Module):
    """Downsample a dense mask prompt to the image-embedding grid (``mask_embed``)."""

    def __init__(self, config: Sam3TrackerPromptEncoderConfig):
        super().__init__()
        mask_channels = config.mask_input_channels // 4
        self.conv1 = nn.Conv2d(1, mask_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(mask_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = nn.LayerNorm(mask_channels, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.mask_input_channels, eps=config.layer_norm_eps)

    def __call__(self, masks: mx.array) -> mx.array:
        hidden = nn.gelu(self.layer_norm1(self.conv1(masks)))
        hidden = nn.gelu(self.layer_norm2(self.conv2(hidden)))
        return self.conv3(hidden)


class Sam3TrackerPromptEncoder(nn.Module):
    """Embed point / box / mask prompts into sparse + dense embeddings (14 tensors)."""

    def __init__(self, config: Sam3TrackerPromptEncoderConfig):
        super().__init__()
        self.shared_embedding = Sam3TrackerPositionalEmbedding(config)
        self.mask_embed = Sam3TrackerMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)
        self.point_embed = nn.Embedding(config.num_point_embeddings, config.hidden_size)
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)
        self.hidden_size = config.hidden_size
        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.image_size

    def __call__(self, input_masks: mx.array | None, batch_size: int = 1) -> mx.array:
        """Dense-embedding path (the streaming tracker's reconditioning input).

        Sparse point/box embedding uses ``shared_embedding``/``point_embed`` and is
        exercised by the slice-11 streaming assembly; the dense path is shape-tested
        here. Returns dense embeddings ``[B, H, W, hidden]`` (NHWC).
        """

        if input_masks is not None:
            return self.mask_embed(input_masks)
        height, width = self.image_embedding_size
        dense = self.no_mask_embed.weight.reshape(1, 1, 1, -1)
        return mx.broadcast_to(dense, (batch_size, height, width, self.hidden_size))

    def get_dense_pe(self) -> mx.array:
        """Positional encoding over the image-embedding grid -> ``[1, H, W, hidden]`` (NHWC)."""
        height, width = self.image_embedding_size
        ys = (mx.arange(height).astype(mx.float32) + 0.5) / height
        xs = (mx.arange(width).astype(mx.float32) + 0.5) / width
        grid = mx.stack(
            [mx.broadcast_to(xs[None, :], (height, width)), mx.broadcast_to(ys[:, None], (height, width))],
            axis=-1,
        )  # [H, W, 2], already normalized -> shared_embedding treats it as such (input_shape=None)
        return self.shared_embedding(grid)[None]

    def encode_sparse(self, points: mx.array, labels: mx.array) -> mx.array:
        """Embed point / box-corner prompts into sparse embeddings ``[B, N, hidden]``.

        ``labels``: -1 padding (not-a-point), 0/1 negative/positive point, 2/3 box top-left/
        bottom-right corner. Mirrors the SAM2 prompt encoder's point path.
        """
        coords = points + 0.5  # shift to pixel centers
        embeddings = self.shared_embedding(coords, input_shape=(self.input_image_size, self.input_image_size))
        labels_e = labels[..., None]
        point_embed = self.point_embed.weight  # [num_point_embeddings, hidden]
        embeddings = mx.where(labels_e == -1, mx.zeros_like(embeddings) + self.not_a_point_embed.weight, embeddings)
        for label_value in range(point_embed.shape[0]):
            embeddings = mx.where(labels_e == label_value, embeddings + point_embed[label_value], embeddings)
        return embeddings


class Sam3TrackerAttention(nn.Module):
    """Attention with optional internal downsampling (q/k/v -> internal, o -> hidden)."""

    def __init__(self, config: Sam3TrackerMaskDecoderConfig, downsample_rate: int | None = None):
        super().__init__()
        rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate
        hidden_size = config.hidden_size
        self.internal_dim = hidden_size // rate
        self.num_heads = config.num_attention_heads
        self.head_dim = self.internal_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(hidden_size, self.internal_dim)
        self.o_proj = nn.Linear(self.internal_dim, hidden_size)

    def __call__(self, query: mx.array, key: mx.array, value: mx.array) -> mx.array:
        batch_size, point_batch_size = query.shape[:2]
        merged = batch_size * point_batch_size

        def _heads(proj, x):
            return proj(x).reshape(merged, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries, keys, values = _heads(self.q_proj, query), _heads(self.k_proj, key), _heads(self.v_proj, value)
        attn = mx.softmax((queries @ keys.transpose(0, 1, 3, 2)) * self.scaling, axis=-1)
        out = (attn @ values).transpose(0, 2, 1, 3).reshape(batch_size, point_batch_size, -1, self.num_heads * self.head_dim)
        return self.o_proj(out)


class _TwoWayAttentionBlock(nn.Module):
    def __init__(self, config: Sam3TrackerMaskDecoderConfig, skip_first_layer_pe: bool):
        super().__init__()
        self.self_attn = Sam3TrackerAttention(config, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.cross_attn_token_to_image = Sam3TrackerAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = Sam3TrackerFeedForward(config.hidden_size, config.mlp_dim, config.hidden_size, config.num_hidden_layers)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size)
        self.layer_norm4 = nn.LayerNorm(config.hidden_size)
        self.cross_attn_image_to_token = Sam3TrackerAttention(config)
        self.skip_first_layer_pe = skip_first_layer_pe

    def __call__(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = queries + self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q, q, queries)
        queries = self.layer_norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        queries = queries + self.cross_attn_token_to_image(q, k, keys)
        queries = self.layer_norm2(queries)

        queries = queries + self.mlp(queries)
        queries = self.layer_norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        keys = keys + self.cross_attn_image_to_token(k, q, queries)
        keys = self.layer_norm4(keys)
        return queries, keys


class Sam3TrackerTwoWayTransformer(nn.Module):
    def __init__(self, config: Sam3TrackerMaskDecoderConfig):
        super().__init__()
        self.layers = [_TwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0)) for i in range(config.num_hidden_layers)]
        self.final_attn_token_to_image = Sam3TrackerAttention(config)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def __call__(self, point_embeddings, image_embeddings, image_positional_embeddings):
        # image_embeddings: [B*pb, H, W, C] -> [B*pb, 1, H*W, C]
        merged, height, width, channels = image_embeddings.shape
        keys = image_embeddings.reshape(merged, 1, height * width, channels)
        image_pe = image_positional_embeddings.reshape(merged, 1, height * width, channels)

        queries = point_embeddings
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embeddings, image_pe)

        query = queries + point_embeddings
        key = keys + image_pe
        queries = queries + self.final_attn_token_to_image(query, key, keys)
        queries = self.layer_norm_final_attn(queries)
        return queries, keys


@dataclass
class Sam3TrackerMaskDecoderOutput:
    masks: mx.array
    iou_pred: mx.array
    object_score_logits: mx.array
    sam_tokens_out: mx.array  # [B, point_batch, num_masks, C] — mask-token outputs (obj-pointer source)


class Sam3TrackerMaskDecoder(nn.Module):
    """SAM2-style mask decoder (TwoWay transformer + hypernetwork masks, 131 tensors)."""

    def __init__(self, config: Sam3TrackerMaskDecoderConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_mask_tokens = config.num_multimask_outputs + 1
        self.iou_token = nn.Embedding(1, hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, hidden_size)
        self.obj_score_token = nn.Embedding(1, hidden_size)
        self.transformer = Sam3TrackerTwoWayTransformer(config)
        self.upscale_conv1 = nn.ConvTranspose2d(hidden_size, hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(hidden_size // 4, hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = nn.LayerNorm(hidden_size // 4)
        self.output_hypernetworks_mlps = [
            Sam3TrackerFeedForward(hidden_size, hidden_size, hidden_size // 8, 3) for _ in range(self.num_mask_tokens)
        ]
        self.iou_prediction_head = Sam3TrackerFeedForward(
            hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )
        self.conv_s0 = nn.Conv2d(hidden_size, hidden_size // 8, kernel_size=1)
        self.conv_s1 = nn.Conv2d(hidden_size, hidden_size // 4, kernel_size=1)
        self.pred_obj_score_head = Sam3TrackerFeedForward(hidden_size, hidden_size, 1, 3)

    def __call__(
        self,
        image_embeddings: mx.array,  # NHWC [B, H, W, C]
        image_positional_embeddings: mx.array,  # NHWC [B, H, W, C]
        sparse_prompt_embeddings: mx.array,  # [B, pb, N, C]
        dense_prompt_embeddings: mx.array,  # NHWC [B, H, W, C]
        high_resolution_features: list[mx.array],  # [feat_s0 (4x), feat_s1 (2x)] NHWC
        multimask_output: bool = True,
    ) -> Sam3TrackerMaskDecoderOutput:
        batch_size, height, width, channels = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]

        output_tokens = mx.concatenate(
            [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], axis=0
        )
        output_tokens = mx.broadcast_to(
            output_tokens[None, None], (batch_size, point_batch_size, output_tokens.shape[0], channels)
        )
        tokens = mx.concatenate([output_tokens, sparse_prompt_embeddings], axis=2)

        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = mx.repeat(image_embeddings, point_batch_size, axis=0)
        image_positional_embeddings = mx.repeat(image_positional_embeddings, point_batch_size, axis=0)

        point_embeddings, keys = self.transformer(tokens, image_embeddings, image_positional_embeddings)
        iou_token_out = point_embeddings[:, :, 1, :]
        mask_tokens_out = point_embeddings[:, :, 2 : 2 + self.num_mask_tokens, :]

        merged = batch_size * point_batch_size
        upscaled = keys.reshape(merged, height, width, channels)
        feat_s0, feat_s1 = high_resolution_features
        feat_s0 = mx.repeat(feat_s0, point_batch_size, axis=0)
        feat_s1 = mx.repeat(feat_s1, point_batch_size, axis=0)
        upscaled = nn.gelu(self.upscale_layer_norm(self.upscale_conv1(upscaled) + feat_s1))
        upscaled = nn.gelu(self.upscale_conv2(upscaled) + feat_s0)

        hyper_in = mx.stack(
            [self.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i, :]) for i in range(self.num_mask_tokens)],
            axis=2,
        )  # [B, pb, num_mask, C/8]

        _, up_h, up_w, up_c = upscaled.shape
        upscaled = upscaled.reshape(batch_size, point_batch_size, up_h * up_w, up_c).transpose(0, 1, 3, 2)
        masks = (hyper_in @ upscaled).reshape(batch_size, point_batch_size, -1, up_h, up_w)

        iou_pred = mx.sigmoid(self.iou_prediction_head(iou_token_out))
        object_score_logits = self.pred_obj_score_head(point_embeddings[:, :, 0, :])

        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks = masks[:, :, mask_slice]
        iou_pred = iou_pred[:, :, mask_slice]
        sam_tokens_out = mask_tokens_out[:, :, mask_slice]
        return Sam3TrackerMaskDecoderOutput(
            masks=masks,
            iou_pred=iou_pred,
            object_score_logits=object_score_logits,
            sam_tokens_out=sam_tokens_out,
        )
