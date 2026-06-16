"""RF-DETR configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ...backbones.vision.dinov2 import DINOv2Config
from ...heads.detection import RFDETRDecoderConfig

__all__ = ["RFDETRConfig"]


@dataclass(frozen=True)
class RFDETRConfig:
    backbone: DINOv2Config
    out_layers: tuple[int, ...]
    projector_out_channels: int = 256
    projector_scale_factors: tuple[float, ...] = (2.0, 1.0, 0.5)
    projector_kind: str = "resize_fuse"
    projector_layer_norm: bool = True
    decoder: RFDETRDecoderConfig = RFDETRDecoderConfig()

    @classmethod
    def rfdetr_nano(cls) -> "RFDETRConfig":
        """RF-DETR Nano real-checkpoint inference configuration."""
        return cls(
            backbone=DINOv2Config.rfdetr_nano(),
            out_layers=(2, 5, 8, 11),
            projector_out_channels=256,
            projector_scale_factors=(1.0,),
            projector_kind="p4_c2f",
            projector_layer_norm=True,
            decoder=RFDETRDecoderConfig(
                hidden_dim=256,
                num_queries=300,
                num_heads=16,
                num_layers=2,
                num_points=2,
                num_classes=91,
                ffn_hidden_dim=2048,
                self_attn_heads=8,
                group_detr=13,
                query_dim=4,
                use_self_attention=True,
                two_stage=True,
                bbox_reparam=True,
                lite_refpoint_refine=True,
                decoder_final_norm=True,
            ),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "RFDETRConfig":
        backbone = d["backbone"]
        if not isinstance(backbone, DINOv2Config):
            backbone = DINOv2Config.from_dict(backbone)
        return cls(
            backbone=backbone,
            out_layers=tuple(int(i) for i in d.get("out_layers", (2, 4, 5, 9))),
            projector_out_channels=int(d.get("projector_out_channels", 256)),
            projector_scale_factors=tuple(float(s) for s in d.get("projector_scale_factors", (2.0, 1.0, 0.5))),
            projector_kind=str(d.get("projector_kind", "resize_fuse")),
            projector_layer_norm=bool(d.get("projector_layer_norm", True)),
            decoder=(
                d["decoder"]
                if isinstance(d.get("decoder"), RFDETRDecoderConfig)
                else RFDETRDecoderConfig(**d.get("decoder", {}))
            ),
        )
