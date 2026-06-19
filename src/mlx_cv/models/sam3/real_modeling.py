"""Faithful MLX assembly of the SAM 3 image detector (``Sam3Model``).

Wires the ported subsystems (slices 2-6) into the end-to-end text-prompt detection
pipeline, mirroring ``transformers.models.sam3.modeling_sam3.Sam3Model.forward``.
Child module names match the upstream state dict exactly so the full
``detector_model.*`` checkpoint loads 1:1 (1468 tensors):

  vision_encoder, text_encoder, text_projection, detr_encoder, geometry_encoder,
  dot_product_scoring, detr_decoder, mask_decoder.

Outputs (taps for the slice-7 image parity gate): ``pred_logits`` ``[B, Q]``,
``pred_boxes`` ``[B, Q, 4]`` (xyxy), ``presence_logits`` ``[B, 1]``, ``pred_masks``
``[B, Q, H, W]``, ``semantic_seg`` ``[B, 1, H, W]``, plus ``vision_last_hidden_state``.

Text-prompt path only (box/geometry prompts are deferred with the geometry encoder).
No torch/transformers imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .real_config import Sam3DetectorConfig
from .real_decoder import Sam3DetrDecoder, box_cxcywh_to_xyxy, inverse_sigmoid
from .real_detr import Sam3DetrEncoder, Sam3DotProductScoring
from .real_geometry import Sam3GeometryEncoder
from .real_mask import Sam3MaskDecoder
from .real_text import Sam3CLIPTextModelWithProjection
from .real_vision import Sam3VisionModel

__all__ = ["Sam3Model", "Sam3ImageOutput", "build_sam3_detector_real"]


@dataclass
class Sam3ImageOutput:
    pred_logits: mx.array  # [B, Q]
    pred_boxes: mx.array  # [B, Q, 4] (xyxy)
    presence_logits: mx.array  # [B, 1]
    pred_masks: mx.array  # [B, Q, H, W]
    semantic_seg: mx.array  # [B, 1, H, W]
    vision_last_hidden_state: mx.array  # [B, H*W, hidden]


def _key_padding_mask(attention_mask: mx.array | None) -> mx.array | None:
    """Additive [B, 1, 1, seq] padding mask over text keys, or None if all valid."""

    if attention_mask is None:
        return None
    valid = attention_mask.astype(mx.float32)
    if bool(mx.all(valid == 1.0).item()):
        return None
    return (1.0 - valid)[:, None, None, :] * -1e9


class Sam3Model(nn.Module):
    """Faithful SAM 3 image detector (1468 tensors), text-prompt path."""

    def __init__(self, config: Sam3DetectorConfig):
        super().__init__()
        self.config = config
        detr_hidden = config.detr_encoder.hidden_size
        self.vision_encoder = Sam3VisionModel(config.vision)
        self.text_encoder = Sam3CLIPTextModelWithProjection(config.text)
        self.text_projection = nn.Linear(config.text.hidden_size, detr_hidden)
        self.detr_encoder = Sam3DetrEncoder(config.detr_encoder)
        self.geometry_encoder = Sam3GeometryEncoder(config.geometry_encoder)
        self.dot_product_scoring = Sam3DotProductScoring(config.detr_decoder)
        self.detr_decoder = Sam3DetrDecoder(config.detr_decoder)
        self.mask_decoder = Sam3MaskDecoder(config.mask_decoder)

    def get_text_features(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        last_hidden_state, _ = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(last_hidden_state)  # [B, seq, detr_hidden]

    def __call__(
        self,
        pixel_values: mx.array,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> Sam3ImageOutput:
        vision_outputs = self.vision_encoder(pixel_values)
        fpn_hidden_states = list(vision_outputs.fpn_hidden_states[:-1])  # first 3 FPN levels (NHWC)
        fpn_position_encoding = list(vision_outputs.fpn_position_encoding[:-1])

        text_features = self.get_text_features(input_ids, attention_mask)
        text_pad_mask = _key_padding_mask(attention_mask)

        # DETR encoder over the single finest-stride level (scale 1.0).
        encoder_outputs = self.detr_encoder(
            vision_features=[fpn_hidden_states[-1]],
            text_features=text_features,
            vision_pos_embeds=[fpn_position_encoding[-1]],
            prompt_cross_attn_mask=text_pad_mask,
        )

        height, width = fpn_hidden_states[-1].shape[1], fpn_hidden_states[-1].shape[2]
        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs.last_hidden_state,
            text_features=encoder_outputs.text_features,
            vision_pos_encoding=encoder_outputs.pos_embeds_flattened,
            text_cross_attn_mask=text_pad_mask,
            spatial_shapes=[(height, width)],
        )

        # Box refinement across all decoder layers, then take the last.
        offsets = self.detr_decoder.box_head(decoder_outputs.intermediate_hidden_states)
        boxes_cxcywh = mx.sigmoid(inverse_sigmoid(decoder_outputs.reference_boxes) + offsets)
        all_pred_boxes = box_cxcywh_to_xyxy(boxes_cxcywh)
        pred_boxes = all_pred_boxes[-1]

        all_logits = self.dot_product_scoring(
            decoder_outputs.intermediate_hidden_states, encoder_outputs.text_features
        )[..., 0]
        pred_logits = all_logits[-1]

        decoder_queries = decoder_outputs.intermediate_hidden_states[-1]
        presence_logits = decoder_outputs.presence_logits[-1]

        mask_outputs = self.mask_decoder(
            decoder_queries=decoder_queries,
            backbone_features=fpn_hidden_states,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            prompt_features=text_features,
            prompt_cross_attn_mask=text_pad_mask,
        )

        return Sam3ImageOutput(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            presence_logits=presence_logits,
            pred_masks=mask_outputs.pred_masks,
            semantic_seg=mask_outputs.semantic_seg,
            vision_last_hidden_state=vision_outputs.last_hidden_state,
        )


def build_sam3_detector_real(config: Sam3DetectorConfig) -> Sam3Model:
    return Sam3Model(config)
