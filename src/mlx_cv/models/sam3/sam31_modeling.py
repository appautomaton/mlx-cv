"""Official SAM 3.1 detector assembly for MLX."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn

from .real_config import Sam3DetectorConfig
from .real_decoder import Sam3DetrDecoder, inverse_sigmoid
from .real_detr import Sam3DetrEncoder, Sam3DotProductScoring
from .real_geometry import Sam3GeometryEncoder
from .real_mask import Sam3MaskDecoder
from .real_text import Sam3CLIPTextModelWithProjection
from .real_vision import SAM31TriVisionModel

__all__ = ["SAM3ImageOutput", "SAM3Model", "sam31_detector_config"]


def sam31_detector_config() -> Sam3DetectorConfig:
    """Return the fixed public SAM 3.1 detector configuration."""

    config = Sam3DetectorConfig()
    vision = replace(
        config.vision,
        backbone=replace(config.vision.backbone, layer_norm_eps=1e-5),
        scale_factors=(4.0, 2.0, 1.0),
        backbone_feature_sizes=((288, 288), (144, 144), (72, 72)),
        layer_norm_eps=1e-5,
    )
    return replace(
        config,
        vision=vision,
        geometry_encoder=replace(config.geometry_encoder, layer_norm_eps=1e-5),
        detr_encoder=replace(config.detr_encoder, layer_norm_eps=1e-5),
        detr_decoder=replace(config.detr_decoder, layer_norm_eps=1e-5),
        mask_decoder=replace(config.mask_decoder, layer_norm_eps=1e-5),
    )


@dataclass
class SAM3ImageOutput:
    pred_logits: mx.array
    pred_boxes: mx.array
    presence_logits: mx.array
    pred_masks: mx.array
    semantic_seg: mx.array
    vision_last_hidden_state: mx.array


def _key_padding_mask(attention_mask: mx.array | None) -> mx.array | None:
    if attention_mask is None:
        return None
    valid = attention_mask.astype(mx.float32)
    if bool(mx.all(valid == 1.0).item()):
        return None
    return (1.0 - valid)[:, None, None, :] * -1e9


class SAM31GeometryEncoder(Sam3GeometryEncoder):
    """SAM 3.1 geometry encoder, including the checkpoint's point projections."""

    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.hidden_size
        self.points_direct_project = nn.Linear(2, hidden_size)
        self.points_pool_project = nn.Linear(hidden_size, hidden_size)
        self.points_pos_enc_project = nn.Linear(hidden_size, hidden_size)

    def encode_empty_prompt(
        self,
        vision_feats: mx.array,
        vision_pos_encoding: mx.array,
    ) -> mx.array:
        """Encode the official always-present geometry CLS token."""

        batch_size = vision_feats.shape[0]
        cls = mx.broadcast_to(
            self.cls_embed.weight[None],
            (batch_size, 1, self.cls_embed.weight.shape[-1]),
        )
        return self.run_layers(cls, vision_feats, vision_pos_encoding)


class SAM3Model(nn.Module):
    """SAM 3.1 text-prompt detector using the shared TriHead vision trunk."""

    def __init__(self, config: Sam3DetectorConfig | None = None):
        super().__init__()
        self.config = sam31_detector_config() if config is None else config
        if tuple(self.config.vision.scale_factors) != (4.0, 2.0, 1.0):
            raise ValueError("SAM3Model requires the SAM 3.1 three-level vision config")
        hidden = self.config.detr_encoder.hidden_size
        self.vision_encoder = SAM31TriVisionModel(self.config.vision)
        self.text_encoder = Sam3CLIPTextModelWithProjection(self.config.text)
        self.text_projection = nn.Linear(self.config.text.hidden_size, hidden)
        self.detr_encoder = Sam3DetrEncoder(self.config.detr_encoder)
        self.geometry_encoder = SAM31GeometryEncoder(self.config.geometry_encoder)
        self.dot_product_scoring = Sam3DotProductScoring(self.config.detr_decoder)
        self.detr_decoder = Sam3DetrDecoder(self.config.detr_decoder)
        self.mask_decoder = Sam3MaskDecoder(self.config.mask_decoder)

    def get_text_features(
        self, input_ids: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        last_hidden_state, _ = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(last_hidden_state)

    def __call__(
        self,
        pixel_values: mx.array,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> SAM3ImageOutput:
        vision = self.vision_encoder(pixel_values)
        fpn = list(vision.fpn_hidden_states)
        fpn_pos = list(vision.fpn_position_encoding)
        # The official CLIP tower uses only its causal mask; padding is masked
        # when the resulting prompt sequence is consumed by the SAM encoder.
        text = self.get_text_features(input_ids, None)
        batch_size = text.shape[0]
        text_valid = (
            mx.ones(input_ids.shape, dtype=mx.bool_)
            if attention_mask is None
            else attention_mask.astype(mx.bool_)
        )

        vision_low = fpn[-1].reshape(batch_size, -1, fpn[-1].shape[-1])
        vision_pos_low = fpn_pos[-1].reshape(
            batch_size, -1, fpn_pos[-1].shape[-1]
        )
        geometry = self.geometry_encoder.encode_empty_prompt(
            vision_low, vision_pos_low
        )
        prompt = mx.concatenate([text, geometry], axis=1)
        prompt_valid = mx.concatenate(
            [text_valid, mx.ones((batch_size, 1), dtype=mx.bool_)], axis=1
        )
        prompt_mask = _key_padding_mask(prompt_valid)

        encoder = self.detr_encoder(
            vision_features=[fpn[-1]],
            text_features=prompt,
            vision_pos_embeds=[fpn_pos[-1]],
            prompt_cross_attn_mask=prompt_mask,
        )
        height, width = fpn[-1].shape[1:3]
        decoder = self.detr_decoder(
            vision_features=encoder.last_hidden_state,
            text_features=prompt,
            vision_pos_encoding=encoder.pos_embeds_flattened,
            text_cross_attn_mask=prompt_mask,
            spatial_shapes=[(height, width)],
        )

        offsets = self.detr_decoder.box_head(decoder.intermediate_hidden_states)
        boxes = mx.sigmoid(inverse_sigmoid(decoder.reference_boxes) + offsets)[-1]
        logits = self.dot_product_scoring(
            decoder.intermediate_hidden_states,
            prompt,
            prompt_valid,
        )[-1, ..., 0]
        presence_logits = decoder.presence_logits[-1]
        # The official 3.1 multiplex detector supervises a joint box score. Its
        # public processor subsequently applies the presence probability again;
        # preserve that exact (slightly unusual) contract for parity.
        logits = mx.clip(
            inverse_sigmoid(mx.sigmoid(logits) * mx.sigmoid(presence_logits)),
            -10.0,
            10.0,
        )
        masks = self.mask_decoder(
            decoder_queries=decoder.intermediate_hidden_states[-1],
            backbone_features=fpn,
            encoder_hidden_states=encoder.last_hidden_state,
            prompt_features=prompt,
            prompt_cross_attn_mask=prompt_mask,
        )
        return SAM3ImageOutput(
            pred_logits=logits,
            pred_boxes=boxes,
            presence_logits=presence_logits,
            pred_masks=masks.pred_masks,
            semantic_seg=masks.semantic_seg,
            vision_last_hidden_state=vision.last_hidden_state,
        )
