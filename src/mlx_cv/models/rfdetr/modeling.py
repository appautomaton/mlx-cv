"""RF-DETR DINOv2 feature path."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...backbones.vision.dinov2 import DINOv2ViT
from ...backbones.vision.necks import RFDETRFeaturePyramid, RFDETRMultiScaleProjector
from ...backbones.vision.necks.rfdetr import RFDETRP4C2fProjector
from ...core.features import BackboneFeatures, HeadOutput
from ...core.types import Result
from ...heads.detection import RFDETRDetectionHead, RFDETRQueryDecoder
from .config import RFDETRConfig

__all__ = ["RFDETRDINOv2Adapter", "RFDETRFeatureExtractor", "RFDETRModel"]


class RFDETRDINOv2Adapter(nn.Module):
    """RF-DETR-owned wrapper around the local DINOv2 backbone."""

    def __init__(self, cfg: RFDETRConfig) -> None:
        super().__init__()
        if not cfg.out_layers:
            raise ValueError("RF-DETR requires at least one DINOv2 output layer")
        invalid = [i for i in cfg.out_layers if i < 0 or i >= cfg.backbone.depth]
        if invalid:
            raise ValueError(f"RF-DETR out_layers outside DINOv2 depth: {invalid}")
        self.cfg = cfg
        self.backbone = DINOv2ViT(cfg.backbone)

    def __call__(self, x: mx.array, *, capture_taps: bool = False) -> BackboneFeatures:
        if len(x.shape) != 4:
            raise ValueError(f"RF-DETR DINOv2 adapter expects NCHW input, got {x.shape}")
        if x.shape[1] != self.cfg.backbone.in_chans:
            raise ValueError(
                f"RF-DETR DINOv2 adapter expects {self.cfg.backbone.in_chans} channels, got {x.shape}"
            )
        patch = self.cfg.backbone.patch_size
        divisor = patch * max(1, self.cfg.backbone.num_windows)
        if x.shape[2] % divisor or x.shape[3] % divisor:
            if divisor == patch:
                raise ValueError(f"RF-DETR input height/width must be divisible by patch size {patch}, got {x.shape}")
            raise ValueError(
                "RF-DETR windowed DINOv2 input height/width must be divisible by "
                f"patch_size * num_windows ({patch} * {self.cfg.backbone.num_windows} = {divisor}), got {x.shape}"
            )
        return self.backbone.forward_features(
            x,
            intermediate_layers=self.cfg.out_layers,
            capture_taps=capture_taps,
        )


class RFDETRFeatureExtractor(nn.Module):
    """DINOv2 adapter plus multi-scale projector for RF-DETR."""

    def __init__(self, cfg: RFDETRConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = RFDETRDINOv2Adapter(cfg)
        in_channels = (cfg.backbone.embed_dim,) * len(cfg.out_layers)
        if cfg.projector_kind == "resize_fuse":
            self.projector = RFDETRMultiScaleProjector(
                in_channels=in_channels,
                out_channels=cfg.projector_out_channels,
                scale_factors=cfg.projector_scale_factors,
            )
        elif cfg.projector_kind == "p4_c2f":
            if tuple(cfg.projector_scale_factors) != (1.0,):
                raise ValueError("RF-DETR p4_c2f projector requires projector_scale_factors=(1.0,)")
            self.projector = RFDETRP4C2fProjector(
                in_channels=in_channels,
                out_channels=cfg.projector_out_channels,
                layer_norm=cfg.projector_layer_norm,
            )
        else:
            raise ValueError(f"unsupported RF-DETR projector_kind: {cfg.projector_kind!r}")

    def __call__(self, x: mx.array, *, capture_taps: bool = False) -> RFDETRFeaturePyramid:
        features = self.backbone(x, capture_taps=capture_taps)
        pyramid = self.projector(features)
        if capture_taps:
            features.extras["rfdetr_pyramid"] = [level.data for level in pyramid.levels]
        return pyramid


class RFDETRModel(nn.Module):
    """Raw RF-DETR detection path: image -> pyramid -> decoder -> logits/boxes."""

    def __init__(self, cfg: RFDETRConfig) -> None:
        super().__init__()
        if cfg.projector_out_channels != cfg.decoder.hidden_dim:
            raise ValueError("projector_out_channels must match decoder.hidden_dim")
        if len(cfg.projector_scale_factors) <= 0:
            raise ValueError("RF-DETR requires at least one projected feature level")
        self.cfg = cfg
        self.feature_extractor = RFDETRFeatureExtractor(cfg)
        self.decoder = RFDETRQueryDecoder(cfg.decoder, num_levels=len(cfg.projector_scale_factors))
        self.head = RFDETRDetectionHead(cfg.decoder)

    def __call__(self, x: mx.array, *, capture_taps: bool = False) -> HeadOutput:
        pyramid = self.feature_extractor(x, capture_taps=capture_taps)
        decoder_out = self.decoder(pyramid, capture_taps=capture_taps)
        out = self.head(decoder_out)
        if capture_taps:
            out.data["pyramid"] = [level.data for level in pyramid.levels]
            taps = {f"projector.level_{i}": level.data for i, level in enumerate(pyramid.levels)}
            attended_layers = decoder_out["deformable_attention"]
            for i in range(attended_layers.shape[0]):
                taps[f"decoder.deformable_attention_{i}"] = attended_layers[i]
            taps["decoder.hidden_states"] = decoder_out["hidden_states"]
            taps["head.logits"] = out["logits"]
            taps["head.boxes"] = out["boxes"]
            out.data["taps"] = taps
        return out

    def predict(self, image, *, processor=None, labels=None, **opts) -> Result:
        """Run ``preprocess -> model -> postprocess`` for one image."""
        if processor is None:
            from .processor import RFDETRProcessor, RFDETRProcessorConfig

            processor = RFDETRProcessor(RFDETRProcessorConfig(labels=labels, **opts))
        elif labels is not None or opts:
            raise ValueError("RFDETRModel.predict accepts labels/options only when processor is not provided")
        model_inputs, ctx = processor.preprocess(image)
        return processor.postprocess(self(model_inputs["pixel_values"]), ctx)
