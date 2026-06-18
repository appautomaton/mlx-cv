"""Depth Anything V3 configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...backbones.vision.dinov2 import DA3AnyViewDINOv2Config, DINOv2Config
from ...heads.dense import DA3DualDPTConfig, DPTConfig
from .camera import DA3CameraDecoderConfig, DA3CameraEncoderConfig

__all__ = ["DA3MonocularConfig", "DA3MultiViewConfig"]

_DA3_SMALL_DUALDPT_FEATURES = 64
_DA3_SMALL_DUALDPT_OUT_CHANNELS = (48, 96, 192, 384)
_DA3_BASE_DUALDPT_FEATURES = 128
_DA3_BASE_DUALDPT_OUT_CHANNELS = (96, 192, 384, 768)


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


@dataclass(frozen=True)
class DA3MultiViewConfig:
    """Opt-in DA3 any-view depth and camera geometry configuration."""

    backbone: DA3AnyViewDINOv2Config = field(default_factory=DA3AnyViewDINOv2Config.small)
    head: DA3DualDPTConfig = field(
        default_factory=lambda: DA3DualDPTConfig(
            dim_in=DA3AnyViewDINOv2Config.small().head_input_dim,
            patch_size=14,
            output_dim=2,
            features=_DA3_SMALL_DUALDPT_FEATURES,
            out_channels=_DA3_SMALL_DUALDPT_OUT_CHANNELS,
            pos_embed=True,
            head_names=("depth", "ray"),
        )
    )
    cam_enc: DA3CameraEncoderConfig = field(
        default_factory=lambda: DA3CameraEncoderConfig(
            dim_out=DA3AnyViewDINOv2Config.small().embed_dim,
            num_heads=16,
        )
    )
    cam_dec: DA3CameraDecoderConfig = field(
        default_factory=lambda: DA3CameraDecoderConfig(dim_in=DA3AnyViewDINOv2Config.small().head_input_dim)
    )
    extrinsics_convention: str = "w2c"

    def __post_init__(self) -> None:
        if self.head.dim_in != self.backbone.head_input_dim:
            raise ValueError(
                f"DA3MultiViewConfig head.dim_in={self.head.dim_in} must equal "
                f"backbone.head_input_dim={self.backbone.head_input_dim}"
            )
        if self.cam_enc.dim_out != self.backbone.embed_dim:
            raise ValueError(
                f"DA3MultiViewConfig cam_enc.dim_out={self.cam_enc.dim_out} must equal "
                f"backbone.embed_dim={self.backbone.embed_dim}"
            )
        if self.cam_dec.dim_in != self.backbone.head_input_dim:
            raise ValueError(
                f"DA3MultiViewConfig cam_dec.dim_in={self.cam_dec.dim_in} must equal "
                f"backbone.head_input_dim={self.backbone.head_input_dim}"
            )
        if self.extrinsics_convention != "w2c":
            raise ValueError("DA3MultiViewConfig currently returns final w2c extrinsics")

    @classmethod
    def from_dict(cls, d: dict) -> "DA3MultiViewConfig":
        backbone = d.get("backbone", d.get("backbone_config", d.get("net")))
        head = d.get("head", d.get("head_config"))
        cam_enc = d.get("cam_enc", d.get("camera_encoder"))
        cam_dec = d.get("cam_dec", d.get("camera_decoder"))
        if backbone is None or head is None or cam_enc is None or cam_dec is None:
            raise ValueError("DA3MultiViewConfig requires backbone, head, cam_enc, and cam_dec")
        backbone_cfg = (
            backbone if isinstance(backbone, DA3AnyViewDINOv2Config) else DA3AnyViewDINOv2Config.from_dict(backbone)
        )
        return cls(
            backbone=backbone_cfg,
            head=head if isinstance(head, DA3DualDPTConfig) else DA3DualDPTConfig.from_dict(head),
            cam_enc=cam_enc
            if isinstance(cam_enc, DA3CameraEncoderConfig)
            else DA3CameraEncoderConfig.from_dict(cam_enc),
            cam_dec=cam_dec
            if isinstance(cam_dec, DA3CameraDecoderConfig)
            else DA3CameraDecoderConfig.from_dict(cam_dec),
            extrinsics_convention=d.get("extrinsics_convention", "w2c"),
        )

    @classmethod
    def small(cls) -> "DA3MultiViewConfig":
        backbone = DA3AnyViewDINOv2Config.small()
        return cls(
            backbone=backbone,
            head=DA3DualDPTConfig(
                dim_in=backbone.head_input_dim,
                features=_DA3_SMALL_DUALDPT_FEATURES,
                out_channels=_DA3_SMALL_DUALDPT_OUT_CHANNELS,
            ),
            cam_enc=DA3CameraEncoderConfig(dim_out=backbone.embed_dim, num_heads=16),
            cam_dec=DA3CameraDecoderConfig(dim_in=backbone.head_input_dim),
        )

    @classmethod
    def base(cls) -> "DA3MultiViewConfig":
        backbone = DA3AnyViewDINOv2Config.base()
        return cls(
            backbone=backbone,
            head=DA3DualDPTConfig(
                dim_in=backbone.head_input_dim,
                features=_DA3_BASE_DUALDPT_FEATURES,
                out_channels=_DA3_BASE_DUALDPT_OUT_CHANNELS,
            ),
            cam_enc=DA3CameraEncoderConfig(dim_out=backbone.embed_dim, num_heads=16),
            cam_dec=DA3CameraDecoderConfig(dim_in=backbone.head_input_dim),
        )

    @classmethod
    def tiny_fixture(cls) -> "DA3MultiViewConfig":
        backbone = DA3AnyViewDINOv2Config(
            embed_dim=8,
            depth=4,
            num_heads=2,
            patch_size=2,
            pretrain_grid=2,
            out_layers=(0, 1, 2, 3),
            alt_start=2,
            qknorm_start=2,
            rope_start=2,
            cat_token=True,
        )
        return cls(
            backbone=backbone,
            head=DA3DualDPTConfig(
                dim_in=backbone.head_input_dim,
                patch_size=2,
                features=16,
                out_channels=(8, 8, 8, 8),
                pos_embed=True,
            ),
            cam_enc=DA3CameraEncoderConfig(dim_out=backbone.embed_dim, num_heads=2, trunk_depth=1),
            cam_dec=DA3CameraDecoderConfig(dim_in=backbone.head_input_dim),
        )
