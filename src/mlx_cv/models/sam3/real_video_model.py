"""Faithful MLX assembly of the SAM 3 video tracker (slice 11).

Wires the ported tracker subsystems (slices 8-10) into ``Sam3TrackerVideoModel``
(the ``tracker_model.*`` namespace, incl. the scalar memory/object-pointer
embeddings) and the top-level ``Sam3VideoModel`` (``detector_model`` + ``tracker_model``
+ ``tracker_neck``), mirroring ``transformers.models.sam3_video.Sam3VideoModel`` so
the full ``facebook/sam3`` video checkpoint loads 1:1 (1797 tensors).

``get_vision_features_for_tracker`` (detector vision encoder -> tracker FPN neck) is
ported and shape-verified. The per-frame streaming / memory-propagation / association
loop (``Sam3VideoModel.forward`` over an inference session) is the remaining
numerically-gated piece and runs out-of-sandbox with the gated checkpoint; it is not
implemented here (no synthetic pass). No torch/transformers imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .real_config import Sam3DetectorConfig
from .real_modeling import Sam3Model
from .real_tracker_decoder import Sam3TrackerMaskDecoder, Sam3TrackerPositionalEmbedding, Sam3TrackerPromptEncoder
from .real_video import Sam3TrackerFeedForward, Sam3TrackerMemoryAttention, Sam3TrackerMemoryEncoder
from .real_video_config import Sam3TrackerVideoConfig
from .real_vision import Sam3VisionConfig, Sam3VisionNeck

__all__ = [
    "Sam3TrackerStageOutput",
    "Sam3TrackerVideoModel",
    "Sam3VideoModel",
    "build_sam3_video_real",
]


def _upsample_nhwc(x: mx.array, factor: int) -> mx.array:
    """Nearest-neighbour integer upsample of an NHWC tensor along H and W."""
    if factor == 1:
        return x
    return mx.repeat(mx.repeat(x, factor, axis=1), factor, axis=2)


def _get_1d_sine_pe(positions: mx.array, dim: int, temperature: float = 10000.0) -> mx.array:
    """1D sinusoidal positional encoding of ``positions`` -> ``[..., dim]`` (SAM2 ``get_1d_sine_pe``)."""
    half = dim // 2
    dim_t = temperature ** (2.0 * (mx.arange(half).astype(mx.float32) // 1) / dim)
    angles = positions[..., None] / dim_t
    return mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)


@dataclass
class Sam3TrackerStageOutput:
    """One frame's faithful tracker output (object batch ``B`` along axis 0)."""

    low_res_masks: mx.array  # NHWC [B, 4g, 4g, 1]
    high_res_masks: mx.array  # NHWC [B, 16g, 16g, 1]
    iou_pred: mx.array  # [B, 1, 1]
    object_score_logits: mx.array  # [B, 1, 1]
    obj_ptr: mx.array  # [B, hidden_dim]
    maskmem_features: mx.array | None = None  # NHWC [B, g, g, mem_dim]
    maskmem_pos_enc: mx.array | None = None  # NHWC [B, g, g, mem_dim]
    extra: dict[str, Any] = field(default_factory=dict)


class Sam3TrackerVideoModel(nn.Module):
    """The SAM2-style tracker (``tracker_model.*``, 307 tensors; no vision encoder)."""

    def __init__(self, config: Sam3TrackerVideoConfig, hidden_dim: int = 256, mem_dim: int = 64):
        super().__init__()
        self.config = config
        self.shared_image_embedding = Sam3TrackerPositionalEmbedding(config.prompt_encoder)
        self.prompt_encoder = Sam3TrackerPromptEncoder(config.prompt_encoder)
        self.mask_decoder = Sam3TrackerMaskDecoder(config.mask_decoder)
        self.memory_attention = Sam3TrackerMemoryAttention(config)
        self.memory_encoder = Sam3TrackerMemoryEncoder(config)
        self.object_pointer_proj = Sam3TrackerFeedForward(hidden_dim, hidden_dim, hidden_dim, 3)
        self.temporal_positional_encoding_projection_layer = nn.Linear(hidden_dim, mem_dim)
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

        # Scalar memory / object-pointer embeddings (top-level tracker parameters).
        self.no_memory_embedding = mx.zeros((1, 1, hidden_dim))
        self.no_memory_positional_encoding = mx.zeros((1, 1, hidden_dim))
        self.memory_temporal_positional_encoding = mx.zeros((config.num_maskmem, 1, 1, mem_dim))
        self.no_object_pointer = mx.zeros((1, hidden_dim))
        self.occlusion_spatial_embedding_parameter = mx.zeros((1, mem_dim))

        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.num_maskmem = config.num_maskmem

    # ---- faithful per-frame streaming step (slice 12) ------------------------

    def _assemble_memory(
        self, previous_frames: list[Sam3TrackerStageOutput], batch: int
    ) -> tuple[mx.array, mx.array, int]:
        """Stack prior-frame spatial memory + object pointers -> ``([mem_seq,B,mem_dim], pos, num_ptr_tokens)``.

        Spatial memory carries the learned ``memory_temporal_positional_encoding`` per slot; object
        pointers are split into ``hidden_dim // mem_dim`` tokens (appended last so memory-attention can
        exclude them from RoPE) with a sine temporal encoding projected to ``mem_dim``.
        """
        recent = previous_frames[-(self.num_maskmem - 1) :]
        spatial_mem: list[mx.array] = []
        spatial_pos: list[mx.array] = []
        for offset, frame in enumerate(reversed(recent)):
            t_pos = offset + 1  # 1 == most recent prior frame
            feats = frame.maskmem_features
            b, h, w, c = feats.shape
            spatial_mem.append(feats.reshape(b, h * w, c).transpose(1, 0, 2))
            pos = frame.maskmem_pos_enc.reshape(b, h * w, c).transpose(1, 0, 2)
            spatial_pos.append(pos + self.memory_temporal_positional_encoding[self.num_maskmem - t_pos - 1])

        obj_tokens: list[mx.array] = []
        obj_token_pos: list[mx.array] = []
        num_obj_ptr_tokens = 0
        if recent:
            stacked = mx.stack([frame.obj_ptr for frame in recent], axis=0)  # [n, B, hidden_dim]
            n = stacked.shape[0]
            tokens_per_ptr = self.hidden_dim // self.mem_dim
            stacked = stacked.reshape(n, batch, tokens_per_ptr, self.mem_dim)
            stacked = stacked.transpose(0, 2, 1, 3).reshape(n * tokens_per_ptr, batch, self.mem_dim)
            rel = mx.arange(n).astype(mx.float32)
            sine = self.temporal_positional_encoding_projection_layer(_get_1d_sine_pe(rel, self.hidden_dim))
            sine = mx.repeat(sine, tokens_per_ptr, axis=0)
            obj_tokens.append(stacked)
            obj_token_pos.append(mx.broadcast_to(sine[:, None, :], (n * tokens_per_ptr, batch, self.mem_dim)))
            num_obj_ptr_tokens = n * tokens_per_ptr

        memory = mx.concatenate(spatial_mem + obj_tokens, axis=0)
        memory_pos = mx.concatenate(spatial_pos + obj_token_pos, axis=0)
        return memory, memory_pos, num_obj_ptr_tokens

    def _condition_on_memory(
        self,
        vision_features: mx.array,
        vision_pos: mx.array,
        previous_frames: list[Sam3TrackerStageOutput] | None,
        is_init_cond_frame: bool,
    ) -> mx.array:
        """Fuse current NHWC features with the memory bank -> NHWC ``[B, g, g, C]``."""
        batch, height, width, channels = vision_features.shape
        if is_init_cond_frame or not previous_frames:
            # First frame: directly add the no-memory embedding (no transformer pass).
            return vision_features + self.no_memory_embedding.reshape(1, 1, 1, channels)

        memory, memory_pos, num_obj_ptr_tokens = self._assemble_memory(previous_frames, batch)
        current = vision_features.reshape(batch, height * width, channels).transpose(1, 0, 2)
        current_pos = vision_pos.reshape(batch, height * width, channels).transpose(1, 0, 2)
        conditioned = self.memory_attention(
            current_vision_features=current,
            memory=memory,
            current_vision_position_embeddings=current_pos,
            memory_position_embeddings=memory_pos,
            num_object_pointer_tokens=num_obj_ptr_tokens,
        )  # [1, B, seq, C]
        return conditioned.reshape(batch, height, width, channels)

    def track_step(
        self,
        *,
        vision_features: mx.array,  # NHWC [B, g, g, C]
        vision_pos: mx.array,  # NHWC [B, g, g, C]
        high_res_features: list[mx.array] | tuple[mx.array, mx.array],  # [4g-res, 2g-res] raw FPN, NHWC
        is_init_cond_frame: bool = False,
        point_inputs: tuple[mx.array, mx.array] | None = None,  # (coords [B,N,2], labels [B,N])
        mask_inputs: mx.array | None = None,  # NHWC [B, Hm, Wm, 1]
        previous_frames: list[Sam3TrackerStageOutput] | None = None,
        run_mem_encoder: bool = True,
    ) -> Sam3TrackerStageOutput:
        """One faithful SAM2-style tracker step over a single frame (object batch ``B``)."""
        batch, grid_h, grid_w, channels = vision_features.shape

        conditioned = self._condition_on_memory(vision_features, vision_pos, previous_frames, is_init_cond_frame)

        # High-res guides for mask upscaling (conv_s0 / conv_s1 own the channel projections).
        feat_s0 = self.mask_decoder.conv_s0(high_res_features[0])
        feat_s1 = self.mask_decoder.conv_s1(high_res_features[1])

        # Sparse prompt: explicit points/box, else a single padding point (label -1).
        if point_inputs is not None:
            coords, labels = point_inputs
        else:
            coords = mx.zeros((batch, 1, 2))
            labels = -mx.ones((batch, 1))
        sparse = self.prompt_encoder.encode_sparse(coords, labels)[:, None]  # [B, 1, N, C]

        # Dense prompt + dense positional encoding.
        dense = self.prompt_encoder(mask_inputs, batch_size=batch)  # NHWC [B, g, g, C]
        image_pe = mx.broadcast_to(self.prompt_encoder.get_dense_pe(), (batch, grid_h, grid_w, channels))

        decoded = self.mask_decoder(conditioned, image_pe, sparse, dense, [feat_s0, feat_s1], multimask_output=False)
        low_res_masks = decoded.masks[:, 0, 0][..., None]  # NHWC [B, 4g, 4g, 1]
        high_res_masks = _upsample_nhwc(low_res_masks, 4)  # NHWC [B, 16g, 16g, 1] for the memory encoder

        # Object pointer from the SAM token, gated by object presence (occlusion handling).
        sam_token = decoded.sam_tokens_out[:, 0, 0]  # [B, C]
        obj_ptr = self.object_pointer_proj(sam_token)
        is_obj = (decoded.object_score_logits[:, 0] > 0).astype(obj_ptr.dtype)  # [B, 1]
        obj_ptr = is_obj * obj_ptr + (1.0 - is_obj) * self.no_object_pointer

        maskmem_features = maskmem_pos_enc = None
        if run_mem_encoder:
            mask_for_mem = mx.sigmoid(high_res_masks)
            mask_for_mem = mask_for_mem * self.config.sigmoid_scale_for_mem_enc + self.config.sigmoid_bias_for_mem_enc
            mem = self.memory_encoder(conditioned, mask_for_mem)
            maskmem_features = mem.vision_features
            maskmem_pos_enc = mem.vision_pos_enc

        return Sam3TrackerStageOutput(
            low_res_masks=low_res_masks,
            high_res_masks=high_res_masks,
            iou_pred=decoded.iou_pred,
            object_score_logits=decoded.object_score_logits,
            obj_ptr=obj_ptr,
            maskmem_features=maskmem_features,
            maskmem_pos_enc=maskmem_pos_enc,
        )


class Sam3VideoModel(nn.Module):
    """Faithful SAM 3 video model: detector + tracker + tracker neck (1797 tensors)."""

    def __init__(
        self,
        detector_config: Sam3DetectorConfig | None = None,
        tracker_config: Sam3TrackerVideoConfig | None = None,
    ):
        super().__init__()
        detector_config = detector_config or Sam3DetectorConfig()
        tracker_config = tracker_config or Sam3TrackerVideoConfig()
        self.detector_model = Sam3Model(detector_config)
        self.tracker_model = Sam3TrackerVideoModel(tracker_config)
        self.tracker_neck = Sam3VisionNeck(detector_config.vision)

    def get_vision_features_for_tracker(self, pixel_values: mx.array) -> tuple[mx.array, ...]:
        """Detector vision encoder -> tracker FPN neck (shape-verified component path).

        Returns the tracker FPN feature maps (NHWC) used as the tracker's image
        embeddings during streaming.
        """

        vision = self.detector_model.vision_encoder(pixel_values)
        hidden_states = vision.last_hidden_state  # [B, H*W, C]
        batch_size, seq_len, channels = hidden_states.shape
        side = int(round(seq_len**0.5))
        spatial = hidden_states.reshape(batch_size, side, side, channels)
        fpn_hidden_states, _ = self.tracker_neck(spatial)
        return fpn_hidden_states


def build_sam3_video_real(
    detector_config: Sam3DetectorConfig | None = None,
    tracker_config: Sam3TrackerVideoConfig | None = None,
) -> Sam3VideoModel:
    return Sam3VideoModel(detector_config, tracker_config)
