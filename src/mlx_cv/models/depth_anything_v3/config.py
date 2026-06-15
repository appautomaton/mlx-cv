"""Depth Anything V3 monocular configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...backbones.vision.dinov2 import DINOv2Config
from ...heads.dense import DPTConfig

__all__ = ["DA3MonocularConfig"]


@dataclass(frozen=True)
class DA3MonocularConfig:
    backbone: DINOv2Config = field(
        default_factory=lambda: DINOv2Config(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            patch_size=14,
            n_register_tokens=0,
            pretrain_grid=37,
            layer_norm_eps=1e-6,
            final_norm_eps=1e-5,
        )
    )
    head: DPTConfig = field(
        default_factory=lambda: DPTConfig(
            dim_in=1024,
            patch_size=14,
            output_dim=1,
            features=256,
            out_channels=(256, 512, 1024, 1024),
            use_sky_head=False,
            pos_embed=False,
            down_ratio=1,
            norm_type="idt",
        )
    )
    out_layers: tuple[int, int, int, int] = (4, 11, 17, 23)

    @classmethod
    def from_dict(cls, d: dict) -> "DA3MonocularConfig":
        backbone = d.get("backbone", d.get("backbone_config"))
        head = d.get("head", d.get("head_config"))
        return cls(
            backbone=backbone if isinstance(backbone, DINOv2Config) else DINOv2Config.from_dict(backbone),
            head=head if isinstance(head, DPTConfig) else DPTConfig.from_dict(head),
            out_layers=tuple(d.get("out_layers", (4, 11, 17, 23))),
        )

    @classmethod
    def tiny_fixture(cls) -> "DA3MonocularConfig":
        return cls(
            backbone=DINOv2Config(
                embed_dim=32,
                depth=4,
                num_heads=4,
                patch_size=14,
                n_register_tokens=0,
                pretrain_grid=2,
                layer_norm_eps=1e-6,
                final_norm_eps=1e-5,
            ),
            head=DPTConfig(
                dim_in=32,
                patch_size=14,
                output_dim=2,
                features=16,
                out_channels=(8, 8, 8, 8),
                use_sky_head=False,
                pos_embed=False,
                down_ratio=1,
                norm_type="idt",
            ),
            out_layers=(0, 1, 2, 3),
        )
