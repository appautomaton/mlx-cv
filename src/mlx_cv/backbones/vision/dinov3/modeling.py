"""DINOv3 ViT in MLX — a faithful port of the official PyTorch reference.

Ported from `references/dinov3` (`dinov3/models/vision_transformer.py` + layers):
patch-embed → prepend ``[cls, storage…]`` → 12× pre-norm SelfAttention blocks with
axial 2D-RoPE on the patch tokens (cls/storage skipped) → final LayerNorm → split
into cls / storage / patch tokens. No LayerScale (``layerscale_init=None``), plain
Mlp/GELU FFN, qkv bias on. The forward output satisfies the spine ``BackboneFeatures``
contract; numerical parity vs the reference is asserted in `tests/test_dinov3_parity.py`.

`core/` stays mlx-free: mlx lives only here, behind the ``[mlx]`` extra.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ....core.features import BackboneFeatures, FeatureMap, Layout, TokenLayout
from ....core.registry import register_backbone
from .config import DINOv3Config

__all__ = ["DINOv3ViT", "build_dinov3"]


def _rope_periods(head_dim: int, base: float) -> mx.array:
    n = head_dim // 4
    return base ** (2.0 * mx.arange(n, dtype=mx.float32) / (head_dim // 2))


def _rope_sincos(periods: mx.array, h: int, w: int) -> tuple[mx.array, mx.array]:
    """Axial RoPE sin/cos for an ``h×w`` patch grid (normalize_coords='separate')."""
    coords_h = mx.arange(0.5, float(h), 1.0, dtype=mx.float32) / h          # (h,)
    coords_w = mx.arange(0.5, float(w), 1.0, dtype=mx.float32) / w          # (w,)
    gh = mx.broadcast_to(coords_h.reshape(h, 1), (h, w))
    gw = mx.broadcast_to(coords_w.reshape(1, w), (h, w))
    coords = mx.stack([gh, gw], axis=-1).reshape(h * w, 2)                  # (HW, 2)
    coords = 2.0 * coords - 1.0
    angles = 2.0 * math.pi * coords[:, :, None] / periods[None, None, :]    # (HW, 2, D//4)
    angles = angles.reshape(h * w, -1)                                      # (HW, D//2)
    angles = mx.concatenate([angles, angles], axis=-1)                      # (HW, D)
    return mx.sin(angles), mx.cos(angles)


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rope(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    return x * cos + _rotate_half(x) * sin


class DINOv3PatchEmbed(nn.Module):
    def __init__(self, cfg: DINOv3Config) -> None:
        super().__init__()
        self.proj = nn.Conv2d(cfg.in_chans, cfg.embed_dim, cfg.patch_size, stride=cfg.patch_size)

    def __call__(self, x: mx.array) -> tuple[mx.array, tuple[int, int]]:
        x = mx.transpose(x, (0, 2, 3, 1))          # NCHW -> NHWC (mlx conv layout)
        x = self.proj(x)                           # (B, Hp, Wp, D)
        b, hp, wp, d = x.shape
        return x.reshape(b, hp * wp, d), (hp, wp)


class DINOv3Attention(nn.Module):
    def __init__(self, cfg: DINOv3Config) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.scale = cfg.head_dim**-0.5
        self.qkv = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=cfg.qkv_bias)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    def __call__(self, x: mx.array, sin: mx.array, cos: mx.array, n_prefix: int) -> mx.array:
        b, n, c = x.shape
        h, dh = self.num_heads, c // self.num_heads
        qkv = self.qkv(x).reshape(b, n, 3, h, dh)
        q = mx.transpose(qkv[:, :, 0], (0, 2, 1, 3))   # (B, h, N, Dh)
        k = mx.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        v = mx.transpose(qkv[:, :, 2], (0, 2, 1, 3))
        q = self._rope(q, sin, cos, n_prefix)
        k = self._rope(k, sin, cos, n_prefix)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v                                  # (B, h, N, Dh)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, n, c)
        return self.proj(out)

    @staticmethod
    def _rope(t: mx.array, sin: mx.array, cos: mx.array, prefix: int) -> mx.array:
        pre = t[:, :, :prefix, :]
        suf = _apply_rope(t[:, :, prefix:, :], sin, cos)
        return mx.concatenate([pre, suf], axis=2)


class DINOv3Mlp(nn.Module):
    def __init__(self, cfg: DINOv3Config) -> None:
        super().__init__()
        hidden = int(cfg.embed_dim * cfg.ffn_ratio)
        self.fc1 = nn.Linear(cfg.embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, cfg.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))           # exact (erf) GELU, matching torch nn.GELU()


class DINOv3Block(nn.Module):
    def __init__(self, cfg: DINOv3Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.attn = DINOv3Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.mlp = DINOv3Mlp(cfg)

    def __call__(self, x: mx.array, sin: mx.array, cos: mx.array, n_prefix: int) -> mx.array:
        x = x + self.attn(self.norm1(x), sin, cos, n_prefix)   # ls1 = Identity
        x = x + self.mlp(self.norm2(x))                        # ls2 = Identity
        return x


class DINOv3ViT(nn.Module):
    """DINOv3 vision transformer (MLX). ``__call__`` == ``forward_features``."""

    def __init__(self, cfg: DINOv3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = DINOv3PatchEmbed(cfg)
        self.cls_token = mx.zeros((1, 1, cfg.embed_dim))
        if cfg.n_storage_tokens > 0:
            self.storage_tokens = mx.zeros((1, cfg.n_storage_tokens, cfg.embed_dim))
        self.periods = _rope_periods(cfg.head_dim, cfg.rope_base)
        self.blocks = [DINOv3Block(cfg) for _ in range(cfg.depth)]
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)

    def __call__(self, x: mx.array) -> BackboneFeatures:
        return self.forward_features(x)

    def forward_features(self, x: mx.array, *, capture_taps: bool = False) -> BackboneFeatures:
        cfg = self.cfg
        taps: dict[str, mx.array] = {}
        patches, (hp, wp) = self.patch_embed(x)              # (B, P, D)
        b, _, d = patches.shape
        cls = mx.broadcast_to(self.cls_token, (b, 1, d))
        parts = [cls]
        if cfg.n_storage_tokens > 0:
            parts.append(mx.broadcast_to(self.storage_tokens, (b, cfg.n_storage_tokens, d)))
        parts.append(patches)
        tokens = mx.concatenate(parts, axis=1)               # (B, 1+R+P, D)
        if capture_taps:
            taps["patch_embed"] = tokens
        sin, cos = _rope_sincos(self.periods, hp, wp)
        if capture_taps:
            taps["rope_sincos"] = mx.stack([sin, cos])
        n_prefix = 1 + cfg.n_storage_tokens
        z = tokens
        for i, blk in enumerate(self.blocks):
            z = blk(z, sin, cos, n_prefix)
            if capture_taps:
                taps[f"block_{i:02d}"] = z
        z_norm = self.norm(z)
        cls_out = z_norm[:, 0]
        storage_out = z_norm[:, 1:n_prefix]
        patch_out = z_norm[:, n_prefix:]
        if capture_taps:
            taps["norm"] = z_norm
            taps["cls"] = cls_out
            taps["storage"] = storage_out
            taps["patch"] = patch_out
        extras: dict = {"x_prenorm": z}
        if capture_taps:
            extras["taps"] = taps
        return BackboneFeatures(
            patch_tokens=FeatureMap(patch_out, layout=Layout.BNC, grid=(hp, wp), stride=cfg.patch_size),
            cls_token=cls_out,
            storage_tokens=storage_out if cfg.n_storage_tokens > 0 else None,
            token_layout=TokenLayout.vit(n_storage=cfg.n_storage_tokens),
            extras=extras,
        )


@register_backbone("dinov3", kind="vision")
def build_dinov3(config) -> DINOv3ViT:
    """Registry builder: a `parity.fixtures` config dict or a `DINOv3Config`."""
    cfg = config if isinstance(config, DINOv3Config) else DINOv3Config.from_dict(config)
    return DINOv3ViT(cfg)
