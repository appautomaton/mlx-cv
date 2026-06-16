"""SAM 3.1 image-mode segmentation model assembly."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...backbones.vision.necks import SAM3FeatureNeck
from ...backbones.vision.sam3 import SAM3ImageBackbone
from ...core.features import HeadOutput
from ...core.geometry import SpatialTransform
from ...heads.segmentation import SAM3ImageDecoder, SAM3MaskDecoder
from ...prompts import PointPrompt
from .config import SAM3Config
from .prompts import SAM3PreparedPrompt, prepare_sam3_prompt
from .text import SAM3TextEncoder, SAM3TextOutput
from .tokenizer import SAM3Tokenizer

__all__ = ["SAM3FeatureExtractor", "SAM3Model"]


class SAM3FeatureExtractor(nn.Module):
    """SAM3 image/VL backbone plus feature neck."""

    def __init__(self, cfg: SAM3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = SAM3ImageBackbone(cfg.image)
        self.neck = SAM3FeatureNeck(
            in_channels=(cfg.image.embed_dim,) * len(cfg.image.out_layers),
            out_channels=cfg.image.neck_channels,
            scale_factors=cfg.image.neck_scales,
        )

    def __call__(self, image: mx.array, *, text_features=None, capture_taps: bool = False):
        features = self.backbone(image, text_features=text_features, capture_taps=capture_taps)
        pyramid = self.neck(features)
        return features, pyramid


class SAM3Model(nn.Module):
    """Raw SAM3 image segmentation path: image + prompt -> mask logits."""

    def __init__(self, cfg: SAM3Config, *, tokenizer: SAM3Tokenizer | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.text_encoder = SAM3TextEncoder(cfg.text, tokenizer=tokenizer)
        self.feature_extractor = SAM3FeatureExtractor(cfg)
        self.decoder = SAM3ImageDecoder(cfg.decoder)
        self.mask_decoder = SAM3MaskDecoder(cfg.decoder)

    def _prepare_prompt(self, image: mx.array, prompt) -> SAM3PreparedPrompt:
        if isinstance(prompt, SAM3PreparedPrompt):
            return prompt
        if isinstance(prompt, PointPrompt):
            raise NotImplementedError("SAM 3.1 image-mode model does not support interactive point prompts")
        if len(image.shape) != 4:
            raise ValueError(f"SAM3Model expects NCHW image input, got {image.shape}")
        model_size = (int(image.shape[2]), int(image.shape[3]))
        return prepare_sam3_prompt(
            prompt,
            transform=SpatialTransform.resize(model_size, model_size),
            model_size=model_size,
        )

    def _encode_text(self, prepared: SAM3PreparedPrompt) -> SAM3TextOutput | None:
        if not prepared.texts:
            return None
        return self.text_encoder(list(prepared.texts))

    def __call__(self, image: mx.array, prompt=None, *, capture_taps: bool = False) -> HeadOutput:
        prepared = self._prepare_prompt(image, prompt)
        text_output = self._encode_text(prepared)
        features, pyramid = self.feature_extractor(
            image,
            text_features=None if text_output is None else text_output.language_features,
            capture_taps=capture_taps,
        )
        decoder_out = self.decoder(
            pyramid,
            prompt=prepared.geometry,
            text_output=text_output,
            capture_taps=capture_taps,
        )
        out = self.mask_decoder(decoder_out, pyramid)
        out.data["text_fused"] = features.extras["text_fused"]
        out.data["prompt_texts"] = prepared.texts
        out.data["prepared_geometry"] = prepared.geometry
        if capture_taps:
            out.data["pyramid"] = [level.data for level in pyramid.levels]
            out.data["decoder_memory"] = decoder_out.get("memory")
        return out
