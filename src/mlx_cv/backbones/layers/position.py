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


def _cubic_weight(t: mx.array) -> mx.array:
    at = mx.abs(t)
    at2 = at * at
    at3 = at2 * at
    a = -0.75
    w1 = (a + 2.0) * at3 - (a + 3.0) * at2 + 1.0
    w2 = a * at3 - 5.0 * a * at2 + 8.0 * a * at - 4.0 * a
    return mx.where(at <= 1.0, w1, mx.where(at < 2.0, w2, mx.zeros_like(t)))


def _torch_bicubic_resize_nhwc(
    x: mx.array,
    *,
    size: tuple[int, int],
    scale_factor: tuple[float, float] | None = None,
) -> mx.array:
    """PyTorch-compatible non-antialiased bicubic resize for NHWC tensors."""

    batch, in_h, in_w, channels = x.shape
    out_h, out_w = int(size[0]), int(size[1])
    input_dtype = x.dtype
    x = x.astype(mx.float32)

    if scale_factor is None:
        scale_h = out_h / in_h
        scale_w = out_w / in_w
    else:
        scale_h = float(scale_factor[0])
        scale_w = float(scale_factor[1])

    y_out = (mx.arange(out_h, dtype=mx.float32) + 0.5) / scale_h - 0.5
    x_out = (mx.arange(out_w, dtype=mx.float32) + 0.5) / scale_w - 0.5

    def weights_1d(coords: mx.array, in_size: int) -> tuple[mx.array, mx.array]:
        start = mx.floor(coords - 2.0).astype(mx.int32) + 1
        offsets = mx.arange(4, dtype=mx.int32)
        pix = start[:, None] + offsets[None, :]
        dist = coords[:, None] - pix.astype(mx.float32)
        weight = _cubic_weight(dist)
        pix = mx.clip(pix, 0, in_size - 1)
        weight = weight / (mx.sum(weight, axis=-1, keepdims=True) + 1e-8)
        return pix, weight

    pix_y, wy = weights_1d(y_out, in_h)
    pix_x, wx = weights_1d(x_out, in_w)
    taps_h = pix_y.shape[1]
    taps_w = pix_x.shape[1]

    gathered_y = x[:, pix_y.reshape(-1), :, :].reshape(batch, out_h, taps_h, in_w, channels)
    tmp = mx.sum(gathered_y * wy[None, :, :, None, None], axis=2)
    gathered_x = tmp[:, :, pix_x.reshape(-1), :].reshape(batch, out_h, out_w, taps_w, channels)
    result = mx.sum(gathered_x * wx[None, None, :, :, None], axis=3)
    return result.astype(input_dtype) if input_dtype != mx.float32 else result


class LearnedAbsPosEmb(nn.Module):
    """Learned absolute positional embedding over ``[cls, patch]`` (DINOv2).

    Holds a ``(1, 1 + Gh*Gw, dim)`` table for the pretrain grid and bicubic-
    interpolates the patch part to the runtime ``(Hp, Wp)`` grid (cubic `Upsample`),
    re-joining the cls slot. **Registers/storage tokens are intentionally not
    covered** — the ViT assembly adds this to ``[cls, patch]`` *before* inserting
    storage, so registers receive no positional embedding (eng-review B2, matching
    the DINOv2-with-registers reference).
    """

    def __init__(
        self,
        dim: int,
        pretrain_grid: int | tuple[int, int],
        *,
        interpolate_offset: float = 0.0,
    ) -> None:
        super().__init__()
        gh, gw = (pretrain_grid, pretrain_grid) if isinstance(pretrain_grid, int) else pretrain_grid
        self.pretrain_grid = (gh, gw)
        self.interpolate_offset = float(interpolate_offset)
        self.table = mx.zeros((1, 1 + gh * gw, dim))   # [cls] + patch grid

    def __call__(self, grid: tuple[int, int]) -> mx.array:
        th, tw = grid
        gh, gw = self.pretrain_grid
        if (th, tw) == (gh, gw):
            return self.table
        d = self.table.shape[-1]
        cls_pos = self.table[:, :1]                                 # (1, 1, D)
        patch_pos = self.table[:, 1:].reshape(1, gh, gw, d)         # (1, Gh, Gw, D)
        if self.interpolate_offset:
            patch_pos = _torch_bicubic_resize_nhwc(
                patch_pos,
                size=(th, tw),
                scale_factor=((th + self.interpolate_offset) / gh, (tw + self.interpolate_offset) / gw),
            )
        else:
            up = nn.Upsample(scale_factor=(th / gh, tw / gw), mode="cubic")
            patch_pos = up(patch_pos)
        patch_pos = patch_pos.reshape(1, th * tw, d)                # (1, Th*Tw, D)
        return mx.concatenate([cls_pos, patch_pos], axis=1)         # (1, 1 + Th*Tw, D)
