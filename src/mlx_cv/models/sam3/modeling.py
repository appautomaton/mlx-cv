"""SAM 3.1 image-mode segmentation model assembly."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

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
            text_features=text_output,
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
            taps = {}
            if text_output is not None:
                taps["text.token_ids"] = text_output.token_ids
                taps["text.language_features"] = text_output.language_features
                taps["text.language_embeds"] = text_output.language_embeds
            if prepared.geometry is not None:
                taps["prompt.boxes_cxcywh"] = np.asarray(prepared.geometry.boxes_cxcywh, dtype=np.float32)
                exemplar = prepared.geometry.exemplar_boxes_cxcywh
                if exemplar is None:
                    exemplar = np.zeros((0, 4), dtype=np.float32)
                taps["prompt.exemplar_boxes_cxcywh"] = np.asarray(exemplar, dtype=np.float32)
            taps["backbone.patch_tokens"] = features.patch_tokens.data
            for i, level in enumerate(pyramid.levels):
                taps[f"neck.level_{i}"] = level.data
            taps["decoder.hidden_states"] = out["hidden_states"]
            taps["head.mask_logits"] = out["mask_logits"]
            taps["head.object_scores"] = out["object_scores"]
            taps["head.boxes"] = out["boxes"]
            out.data["taps"] = taps
            out.data["pyramid"] = [level.data for level in pyramid.levels]
            out.data["decoder_memory"] = decoder_out.get("memory")
        return out

    def predict(self, image, prompt=None, *, processor=None, labels=None, **opts):
        """Run ``preprocess -> model -> postprocess`` for one image."""

        if processor is None:
            from .processor import SAM3Processor, SAM3ProcessorConfig

            processor = SAM3Processor(SAM3ProcessorConfig(labels=labels, **opts))
        elif labels is not None or opts:
            raise ValueError("SAM3Model.predict accepts labels/options only when processor is not provided")
        model_inputs, ctx = processor.preprocess({"image": image, "prompt": prompt})
        return processor.postprocess(self(model_inputs["pixel_values"], model_inputs["prompt"]), ctx)
