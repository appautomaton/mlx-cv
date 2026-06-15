"""2D positional-encoding suite (build-once family).

Holds the variants the current backbones consume:

* **Axial 2D-RoPE** (DINOv3) — ``rope_axial_periods`` / ``rope_axial_sincos``
  build the per-grid sin/cos; ``apply_rope`` / ``apply_rope_prefixed`` rotate
  q/k (prefix tokens — cls/storage — skipped).

The DINOv2 ``learned-absolute + interpolation`` variant is added in the slice
that introduces its consumer (foundation-forward, not built speculatively).

Faithful extraction of the official DINOv3 RoPE math (`references/dinov3`):
``normalize_coords='separate'``, half-rotation layout, periods over ``head_dim``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "rope_axial_periods",
    "rope_axial_sincos",
    "rotate_half",
    "apply_rope",
    "apply_rope_prefixed",
    "LearnedAbsPosEmb",
]


def rope_axial_periods(head_dim: int, base: float) -> mx.array:
    """RoPE periods buffer for one head (``head_dim // 4`` entries)."""
    n = head_dim // 4
    return base ** (2.0 * mx.arange(n, dtype=mx.float32) / (head_dim // 2))


def rope_axial_sincos(periods: mx.array, h: int, w: int) -> tuple[mx.array, mx.array]:
    """Axial RoPE sin/cos for an ``h×w`` patch grid (``normalize_coords='separate'``)."""
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


def rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    return x * cos + rotate_half(x) * sin


def apply_rope_prefixed(t: mx.array, sin: mx.array, cos: mx.array, prefix: int) -> mx.array:
    """Apply RoPE to the suffix (patch) tokens only; the ``prefix`` (cls/storage) pass through."""
    pre = t[:, :, :prefix, :]
    suf = apply_rope(t[:, :, prefix:, :], sin, cos)
    return mx.concatenate([pre, suf], axis=2)


class LearnedAbsPosEmb(nn.Module):
    """Learned absolute positional embedding over ``[cls, patch]`` (DINOv2).

    Holds a ``(1, 1 + Gh*Gw, dim)`` table for the pretrain grid and bicubic-
    interpolates the patch part to the runtime ``(Hp, Wp)`` grid (cubic `Upsample`),
    re-joining the cls slot. **Registers/storage tokens are intentionally not
    covered** — the ViT assembly adds this to ``[cls, patch]`` *before* inserting
    storage, so registers receive no positional embedding (eng-review B2, matching
    the DINOv2-with-registers reference).
    """

    def __init__(self, dim: int, pretrain_grid: int | tuple[int, int]) -> None:
        super().__init__()
        gh, gw = (pretrain_grid, pretrain_grid) if isinstance(pretrain_grid, int) else pretrain_grid
        self.pretrain_grid = (gh, gw)
        self.table = mx.zeros((1, 1 + gh * gw, dim))   # [cls] + patch grid

    def __call__(self, grid: tuple[int, int]) -> mx.array:
        th, tw = grid
        gh, gw = self.pretrain_grid
        if (th, tw) == (gh, gw):
            return self.table
        d = self.table.shape[-1]
        cls_pos = self.table[:, :1]                                 # (1, 1, D)
        patch_pos = self.table[:, 1:].reshape(1, gh, gw, d)         # (1, Gh, Gw, D)
        up = nn.Upsample(scale_factor=(th / gh, tw / gw), mode="cubic")
        patch_pos = up(patch_pos).reshape(1, th * tw, d)            # (1, Th*Tw, D)
        return mx.concatenate([cls_pos, patch_pos], axis=1)         # (1, 1 + Th*Tw, D)
