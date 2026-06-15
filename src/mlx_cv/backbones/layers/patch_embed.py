"""Patch embedding (build-once family).

Conv2d patchifier shared across ViT backbones. Input is NCHW (spine
convention); mlx conv wants NHWC, so we transpose in, then flatten the
``Hp×Wp`` grid to a ``(B, Hp*Wp, C)`` token sequence and return the grid.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = ["PatchEmbed"]


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> tuple[mx.array, tuple[int, int]]:
        x = mx.transpose(x, (0, 2, 3, 1))          # NCHW -> NHWC (mlx conv layout)
        x = self.proj(x)                           # (B, Hp, Wp, D)
        b, hp, wp, d = x.shape
        return x.reshape(b, hp * wp, d), (hp, wp)
