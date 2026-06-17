"""Depth Anything 3 camera pose utilities and camera token heads."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ...backbones.vision.dinov2 import DA3TransformerBlock

__all__ = [
    "DA3CameraDecoder",
    "DA3CameraDecoderConfig",
    "DA3CameraEncoder",
    "DA3CameraEncoderConfig",
    "affine_inverse",
    "as_homogeneous",
    "extri_intri_to_pose_encoding",
    "fov_to_intrinsics",
    "intrinsics_to_fov",
    "mat_to_quat",
    "pose_encoding_to_extri_intri",
    "quat_to_mat",
    "standardize_quaternion",
]


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _transpose_last_two(x: mx.array) -> mx.array:
    axes = list(range(x.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return mx.transpose(x, tuple(axes))


def as_homogeneous(extrinsics: mx.array) -> mx.array:
    """Accept ``(...,3,4)`` or ``(...,4,4)`` and return homogeneous ``(...,4,4)``."""

    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsics must have shape (...,3,4) or (...,4,4), got {tuple(extrinsics.shape)}")
    bottom = mx.zeros((*extrinsics.shape[:-2], 1, 4), dtype=extrinsics.dtype)
    one = mx.ones((*extrinsics.shape[:-2], 1, 1), dtype=extrinsics.dtype)
    bottom = mx.concatenate([bottom[..., :, :3], one], axis=-1)
    return mx.concatenate([extrinsics, bottom], axis=-2)


def affine_inverse(A: mx.array) -> mx.array:
    """Invert a rigid affine matrix in DA3 convention, preserving 3x4 vs 4x4 shape."""

    if A.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"affine_inverse expects (...,3,4) or (...,4,4), got {tuple(A.shape)}")
    R = A[..., :3, :3]
    T = A[..., :3, 3:4]
    Rt = _transpose_last_two(R)
    top = mx.concatenate([Rt, -(Rt @ T)], axis=-1)
    if A.shape[-2] == 3:
        return top
    return mx.concatenate([top, A[..., 3:4, :]], axis=-2)


def standardize_quaternion(quaternions: mx.array) -> mx.array:
    """Return scalar-last quaternions with non-negative scalar component."""

    return mx.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def quat_to_mat(quaternions: mx.array) -> mx.array:
    """Convert scalar-last ``xyzw`` quaternions to rotation matrices."""

    if quaternions.shape[-1] != 4:
        raise ValueError(f"quaternions must have final dimension 4, got {tuple(quaternions.shape)}")
    i, j, k, r = [quaternions[..., idx] for idx in range(4)]
    denom = mx.maximum(mx.sum(quaternions * quaternions, axis=-1), 1e-8)
    two_s = 2.0 / denom
    mats = mx.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return mats.reshape(*quaternions.shape[:-1], 3, 3)


def _sqrt_positive_part(x: mx.array) -> mx.array:
    return mx.sqrt(mx.maximum(x, 0))


def mat_to_quat(matrix: mx.array) -> mx.array:
    """Convert rotation matrices to scalar-last ``xyzw`` quaternions."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must have shape (...,3,3), got {tuple(matrix.shape)}")
    flat = matrix.reshape(*matrix.shape[:-2], 9)
    m00, m01, m02 = flat[..., 0], flat[..., 1], flat[..., 2]
    m10, m11, m12 = flat[..., 3], flat[..., 4], flat[..., 5]
    m20, m21, m22 = flat[..., 6], flat[..., 7], flat[..., 8]

    q_abs = _sqrt_positive_part(
        mx.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )
    quat_by_rijk = mx.stack(
        [
            mx.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            mx.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            mx.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            mx.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )
    candidates = quat_by_rijk / (2.0 * mx.maximum(q_abs, 0.1)[..., None])
    idx = mx.argmax(q_abs, axis=-1).astype(mx.int32)
    gather_idx = mx.broadcast_to(idx[..., None, None], (*idx.shape, 1, 4))
    rijk = mx.take_along_axis(candidates, gather_idx, axis=-2)[..., 0, :]
    xyzw = rijk[..., [1, 2, 3, 0]]
    return standardize_quaternion(xyzw)


def intrinsics_to_fov(intrinsics: mx.array, image_size_hw: tuple[int, int]) -> tuple[mx.array, mx.array]:
    """Return ``(fov_h, fov_w)`` from DA3 intrinsics and ``(H,W)`` image size."""

    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must have shape (...,3,3), got {tuple(intrinsics.shape)}")
    height, width = image_size_hw
    fov_h = 2 * mx.arctan((height / 2.0) / intrinsics[..., 1, 1])
    fov_w = 2 * mx.arctan((width / 2.0) / intrinsics[..., 0, 0])
    return fov_h, fov_w


def fov_to_intrinsics(fov_h: mx.array, fov_w: mx.array, image_size_hw: tuple[int, int]) -> mx.array:
    """Build DA3 intrinsics from horizontal/vertical FOV tensors."""

    height, width = image_size_hw
    fy = (height / 2.0) / mx.maximum(mx.tan(fov_h / 2.0), 1e-6)
    fx = (width / 2.0) / mx.maximum(mx.tan(fov_w / 2.0), 1e-6)
    zeros = mx.zeros_like(fx)
    ones = mx.ones_like(fx)
    cx = zeros + (width / 2.0)
    cy = zeros + (height / 2.0)
    row0 = mx.stack([fx, zeros, cx], axis=-1)
    row1 = mx.stack([zeros, fy, cy], axis=-1)
    row2 = mx.stack([zeros, zeros, ones], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


def extri_intri_to_pose_encoding(
    extrinsics: mx.array,
    intrinsics: mx.array,
    image_size_hw: tuple[int, int],
) -> mx.array:
    """Convert ``c2w`` extrinsics and intrinsics to DA3's 9D pose encoding."""

    if extrinsics.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"extrinsics must have shape (...,3,4) or (...,4,4), got {tuple(extrinsics.shape)}")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must have shape (...,3,3), got {tuple(intrinsics.shape)}")
    R = extrinsics[..., :3, :3]
    T = extrinsics[..., :3, 3]
    quat = mat_to_quat(R)
    fov_h, fov_w = intrinsics_to_fov(intrinsics, image_size_hw)
    return mx.concatenate([T, quat, fov_h[..., None], fov_w[..., None]], axis=-1).astype(mx.float32)


def pose_encoding_to_extri_intri(
    pose_encoding: mx.array,
    image_size_hw: tuple[int, int],
) -> tuple[mx.array, mx.array]:
    """Decode DA3's 9D pose encoding to ``c2w`` extrinsics and intrinsics."""

    if pose_encoding.shape[-1] != 9:
        raise ValueError(f"pose encoding must have final dimension 9, got {tuple(pose_encoding.shape)}")
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    R = quat_to_mat(quat)
    extrinsics = mx.concatenate([R, T[..., None]], axis=-1)
    intrinsics = fov_to_intrinsics(pose_encoding[..., 7], pose_encoding[..., 8], image_size_hw)
    return extrinsics, intrinsics


class PoseMlp(nn.Module):
    def __init__(self, dim_in: int, hidden: int, dim_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden)
        self.fc2 = nn.Linear(hidden, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


@dataclass(frozen=True)
class DA3CameraEncoderConfig:
    dim_out: int
    dim_in: int = 9
    trunk_depth: int = 4
    target_dim: int = 9
    num_heads: int = 16
    mlp_ratio: int = 4
    init_values: float = 0.01
    norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, d: dict) -> "DA3CameraEncoderConfig":
        return cls(
            dim_out=int(d["dim_out"]),
            dim_in=int(d.get("dim_in", 9)),
            trunk_depth=int(d.get("trunk_depth", 4)),
            target_dim=int(d.get("target_dim", 9)),
            num_heads=int(d.get("num_heads", 16)),
            mlp_ratio=int(d.get("mlp_ratio", 4)),
            init_values=float(d.get("init_values", 0.01)),
            norm_eps=float(d.get("norm_eps", d.get("layer_norm_eps", 1e-5))),
        )


class DA3CameraEncoder(nn.Module):
    """Pose-conditioned camera token encoder for DA3 any-view DINOv2."""

    def __init__(self, cfg: DA3CameraEncoderConfig) -> None:
        super().__init__()
        if cfg.dim_out % cfg.num_heads:
            raise ValueError("DA3CameraEncoder dim_out must be divisible by num_heads")
        self.cfg = cfg
        self.target_dim = cfg.target_dim
        self.trunk_depth = cfg.trunk_depth
        self.trunk = [
            DA3TransformerBlock(
                cfg.dim_out,
                cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qk_norm=False,
                rope_frequency=None,
                norm_eps=cfg.norm_eps,
                layerscale_init=cfg.init_values,
            )
            for _ in range(cfg.trunk_depth)
        ]
        self.token_norm = nn.LayerNorm(cfg.dim_out)
        self.trunk_norm = nn.LayerNorm(cfg.dim_out)
        self.pose_branch = PoseMlp(cfg.dim_in, cfg.dim_out // 2, cfg.dim_out)

    def pose_encoding(
        self,
        extrinsics: mx.array,
        intrinsics: mx.array,
        image_size_hw: tuple[int, int],
    ) -> mx.array:
        if extrinsics.ndim != 4 or extrinsics.shape[-2:] not in ((3, 4), (4, 4)):
            raise ValueError(
                "pose-conditioned extrinsics must have shape (B,V,3,4) or (B,V,4,4), "
                f"got {tuple(extrinsics.shape)}"
            )
        if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError(f"pose-conditioned intrinsics must have shape (B,V,3,3), got {tuple(intrinsics.shape)}")
        if intrinsics.shape[:2] != extrinsics.shape[:2]:
            raise ValueError(
                "pose-conditioned extrinsics and intrinsics must share B,V axes, "
                f"got {tuple(extrinsics.shape[:2])} and {tuple(intrinsics.shape[:2])}"
            )
        c2w = affine_inverse(extrinsics)
        return extri_intri_to_pose_encoding(c2w, intrinsics, image_size_hw)

    def __call__(
        self,
        extrinsics: mx.array,
        intrinsics: mx.array,
        image_size_hw: tuple[int, int],
        *,
        return_pose_encoding: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        pose = self.pose_encoding(extrinsics, intrinsics, image_size_hw)
        tokens = self.token_norm(self.pose_branch(pose))
        for block in self.trunk:
            tokens = block(tokens)
        tokens = self.trunk_norm(tokens)
        if return_pose_encoding:
            return tokens, pose
        return tokens


@dataclass(frozen=True)
class DA3CameraDecoderConfig:
    dim_in: int

    @classmethod
    def from_dict(cls, d: dict) -> "DA3CameraDecoderConfig":
        return cls(dim_in=int(d["dim_in"]))


class DA3CameraDecoder(nn.Module):
    """DA3 camera decoder producing 9D pose encodings from camera tokens."""

    def __init__(self, cfg: DA3CameraDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = [
            nn.Linear(cfg.dim_in, cfg.dim_in),
            Identity(),
            nn.Linear(cfg.dim_in, cfg.dim_in),
            Identity(),
        ]
        self.fc_t = nn.Linear(cfg.dim_in, 3)
        self.fc_qvec = nn.Linear(cfg.dim_in, 4)
        self.fc_fov = [nn.Linear(cfg.dim_in, 2), Identity()]

    def __call__(self, feat: mx.array, camera_encoding: mx.array | None = None) -> mx.array:
        if feat.ndim != 3 or feat.shape[-1] != self.cfg.dim_in:
            raise ValueError(f"DA3CameraDecoder expects (B,V,{self.cfg.dim_in}) features, got {tuple(feat.shape)}")
        batch, views = int(feat.shape[0]), int(feat.shape[1])
        x = feat.reshape(batch * views, self.cfg.dim_in)
        x = _relu(self.backbone[0](x))
        x = _relu(self.backbone[2](x))
        out_t = self.fc_t(x).reshape(batch, views, 3)
        if camera_encoding is None:
            out_qvec = self.fc_qvec(x).reshape(batch, views, 4)
            out_fov = _relu(self.fc_fov[0](x)).reshape(batch, views, 2)
        else:
            if camera_encoding.shape != (batch, views, 9):
                raise ValueError(
                    f"camera_encoding must have shape {(batch, views, 9)}, got {tuple(camera_encoding.shape)}"
                )
            out_qvec = camera_encoding[..., 3:7]
            out_fov = camera_encoding[..., 7:9]
        return mx.concatenate([out_t, out_qvec, out_fov], axis=-1)

    def decode_camera(
        self,
        feat: mx.array,
        image_size_hw: tuple[int, int],
        camera_encoding: mx.array | None = None,
    ) -> dict[str, mx.array]:
        pose = self(feat, camera_encoding=camera_encoding)
        c2w, intrinsics = pose_encoding_to_extri_intri(pose, image_size_hw)
        return {
            "pose_encoding": pose,
            "extrinsics": affine_inverse(c2w),
            "intrinsics": intrinsics,
        }


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x
