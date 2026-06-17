"""Depth Anything V3 model assembly."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...backbones.vision.dinov2 import DA3AnyViewDINOv2, DINOv2ViT
from ...core.features import HeadInput, HeadOutput
from ...core.registry import register_model
from ...heads.dense import DA3DualDPT, DPTHead
from .camera import DA3CameraDecoder, DA3CameraEncoder
from .config import DA3MonocularConfig, DA3MultiViewConfig

__all__ = [
    "DepthAnythingV3Monocular",
    "DepthAnythingV3MultiView",
    "build_depth_anything_v3_monocular",
    "build_depth_anything_v3_multiview",
]


class DepthAnythingV3Monocular(nn.Module):
    """DA3 monocular path: DINOv2 selected intermediates -> DPT depth head."""

    def __init__(self, cfg: DA3MonocularConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DINOv2ViT(cfg.backbone)
        self.head = DPTHead(cfg.head)

    def __call__(self, x: mx.array, *, capture_taps: bool = False) -> HeadOutput:
        if x.ndim != 4:
            raise ValueError(f"DepthAnythingV3Monocular expects NCHW input, got shape {x.shape}")
        if x.shape[1] != self.cfg.backbone.in_chans:
            raise ValueError(
                f"DepthAnythingV3Monocular expects NCHW input with {self.cfg.backbone.in_chans} "
                f"channels at axis 1, got shape {x.shape}"
            )
        if x.shape[2] % self.cfg.backbone.patch_size or x.shape[3] % self.cfg.backbone.patch_size:
            raise ValueError(
                f"DepthAnythingV3Monocular input height/width must be divisible by patch size "
                f"{self.cfg.backbone.patch_size}, got shape {x.shape}"
            )
        feats = self.backbone.forward_features(
            x,
            intermediate_layers=self.cfg.out_layers,
            capture_taps=capture_taps,
        )
        out = self.head(
            HeadInput(features=feats, image_size=(int(x.shape[2]), int(x.shape[3]))),
            capture_taps=capture_taps,
        )
        if capture_taps:
            taps = {}
            taps.update({f"dinov2.{k}": v for k, v in feats.extras.get("taps", {}).items()})
            taps.update({f"dpt.{k}": v for k, v in out.data.get("taps", {}).items()})
            out.data["taps"] = taps
        return out


class DepthAnythingV3MultiView(nn.Module):
    """DA3 any-view path: any-view DINOv2 -> DualDPT depth/ray -> camera decoder.

    Final ``extrinsics`` are returned in DA3's default ``w2c`` convention after
    decoding an intermediate ``c2w`` pose and applying ``affine_inverse``.
    """

    def __init__(self, cfg: DA3MultiViewConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DA3AnyViewDINOv2(cfg.backbone)
        self.head = DA3DualDPT(cfg.head)
        self.cam_enc = DA3CameraEncoder(cfg.cam_enc)
        self.cam_dec = DA3CameraDecoder(cfg.cam_dec)

    def __call__(
        self,
        x: mx.array,
        *,
        extrinsics: mx.array | None = None,
        intrinsics: mx.array | None = None,
        capture_taps: bool = False,
        reference_view_strategy: str = "saddle_balanced",
    ) -> HeadOutput:
        if x.ndim != 5:
            raise ValueError(f"DepthAnythingV3MultiView expects BVCHW input, got shape {tuple(x.shape)}")
        batch, views, channels, height, width = x.shape
        if channels != self.cfg.backbone.in_chans:
            raise ValueError(
                f"DepthAnythingV3MultiView expects {self.cfg.backbone.in_chans} channels at axis 2, "
                f"got shape {tuple(x.shape)}"
            )
        if height % self.cfg.backbone.patch_size or width % self.cfg.backbone.patch_size:
            raise ValueError(
                "DepthAnythingV3MultiView input height/width must be divisible by patch size "
                f"{self.cfg.backbone.patch_size}, got shape {tuple(x.shape)}"
            )
        if (extrinsics is None) != (intrinsics is None):
            raise ValueError("pose-conditioned DA3 input requires both extrinsics and intrinsics")

        cam_token = None
        conditioning_pose = None
        if extrinsics is not None and intrinsics is not None:
            extrinsics = mx.array(extrinsics)
            intrinsics = mx.array(intrinsics)
            if extrinsics.shape[:2] != (batch, views):
                raise ValueError(
                    f"pose-conditioned extrinsics must share input B,V axes {(batch, views)}, "
                    f"got {tuple(extrinsics.shape)}"
                )
            if intrinsics.shape[:2] != (batch, views):
                raise ValueError(
                    f"pose-conditioned intrinsics must share input B,V axes {(batch, views)}, "
                    f"got {tuple(intrinsics.shape)}"
                )
            cam_token, conditioning_pose = self.cam_enc(
                extrinsics,
                intrinsics,
                (int(height), int(width)),
                return_pose_encoding=True,
            )

        feats = self.backbone.forward_features(
            x,
            intermediate_layers=self.cfg.backbone.out_layers,
            capture_taps=capture_taps,
            cam_token=cam_token,
            reference_view_strategy=reference_view_strategy,
        )
        out = self.head(
            HeadInput(features=feats, image_size=(int(height), int(width))),
            capture_taps=capture_taps,
        )

        camera_tokens = feats.extras.get("camera_tokens")
        if not camera_tokens:
            raise ValueError("DA3 any-view backbone did not provide camera tokens for camera decoding")
        camera_data = self.cam_dec.decode_camera(camera_tokens[-1], (int(height), int(width)))
        out.data.update(camera_data)
        out.data["extrinsics_convention"] = self.cfg.extrinsics_convention

        if capture_taps:
            taps = {}
            taps.update({f"anyview.{k}": v for k, v in feats.extras.get("taps", {}).items()})
            taps.update({f"dualdpt.{k}": v for k, v in out.data.get("taps", {}).items()})
            if cam_token is not None:
                taps["camera_enc.tokens"] = cam_token
                taps["camera_enc.pose_encoding"] = conditioning_pose
            taps["camera_dec.input_tokens"] = camera_tokens[-1]
            taps["camera_dec.pose_encoding"] = camera_data["pose_encoding"]
            taps["camera_dec.extrinsics_w2c"] = camera_data["extrinsics"]
            taps["camera_dec.intrinsics"] = camera_data["intrinsics"]
            out.data["taps"] = taps
        return out


@register_model("depth-anything-v3-monocular")
def build_depth_anything_v3_monocular(config) -> DepthAnythingV3Monocular:
    cfg = config if isinstance(config, DA3MonocularConfig) else DA3MonocularConfig.from_dict(config)
    return DepthAnythingV3Monocular(cfg)


@register_model("depth-anything-v3-multiview")
def build_depth_anything_v3_multiview(config) -> DepthAnythingV3MultiView:
    cfg = config if isinstance(config, DA3MultiViewConfig) else DA3MultiViewConfig.from_dict(config)
    return DepthAnythingV3MultiView(cfg)
