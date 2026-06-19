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

import mlx.core as mx
import mlx.nn as nn

from .real_config import Sam3DetectorConfig
from .real_modeling import Sam3Model
from .real_tracker_decoder import Sam3TrackerMaskDecoder, Sam3TrackerPositionalEmbedding, Sam3TrackerPromptEncoder
from .real_video import Sam3TrackerFeedForward, Sam3TrackerMemoryAttention, Sam3TrackerMemoryEncoder
from .real_video_config import Sam3TrackerVideoConfig
from .real_vision import Sam3VisionConfig, Sam3VisionNeck

__all__ = ["Sam3TrackerVideoModel", "Sam3VideoModel", "build_sam3_video_real"]


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
