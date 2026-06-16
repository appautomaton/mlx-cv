"""MLX sampling ops used by detector and segmentation heads."""

from __future__ import annotations

import mlx.core as mx

__all__ = ["bilinear_grid_sample_nchw"]


def _check_grid_sample_inputs(x: mx.array, grid: mx.array) -> None:
    if len(x.shape) != 4:
        raise ValueError(f"input must have shape (N,C,H,W), got {x.shape}")
    if len(grid.shape) != 4 or grid.shape[-1] != 2:
        raise ValueError(f"grid must have shape (N,Hg,Wg,2), got {grid.shape}")
    if grid.shape[0] != x.shape[0]:
        raise ValueError(f"grid batch {grid.shape[0]} must match input batch {x.shape[0]}")


def _gather_nchw(flat: mx.array, y: mx.array, x: mx.array, width: int, out_hw: tuple[int, int]) -> mx.array:
    batch, channels, _ = flat.shape
    out_h, out_w = out_hw
    idx = (y * width + x).reshape(batch, out_h * out_w).astype(mx.int32)
    idx = mx.broadcast_to(mx.expand_dims(idx, axis=1), (batch, channels, out_h * out_w))
    return mx.take_along_axis(flat, idx, axis=2).reshape(batch, channels, out_h, out_w)


def bilinear_grid_sample_nchw(
    x: mx.array,
    grid: mx.array,
    *,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> mx.array:
    """Bilinear sample ``x`` at normalized ``grid`` coordinates.

    This matches the RF-DETR reference path's
    ``torch.nn.functional.grid_sample(..., mode="bilinear")`` semantics for NCHW
    tensors. ``grid`` coordinates are in ``[-1, 1]`` with ``(..., 0)`` as x and
    ``(..., 1)`` as y. ``padding_mode="zeros"`` is the deformable-attention path;
    ``"border"`` is included because the reference helper supports it.
    """

    _check_grid_sample_inputs(x, grid)
    if padding_mode not in {"zeros", "border"}:
        raise ValueError("padding_mode must be 'zeros' or 'border'")

    batch, channels, height, width = x.shape
    out_h, out_w = grid.shape[1], grid.shape[2]
    gx = grid[..., 0]
    gy = grid[..., 1]

    if align_corners:
        ix = (gx + 1) * (width - 1) / 2
        iy = (gy + 1) * (height - 1) / 2
    else:
        ix = (gx + 1) * width / 2 - 0.5
        iy = (gy + 1) * height / 2 - 0.5

    ix0 = mx.floor(ix).astype(mx.int32)
    iy0 = mx.floor(iy).astype(mx.int32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    wx1 = mx.expand_dims((ix - ix0.astype(ix.dtype)).astype(x.dtype), axis=1)
    wy1 = mx.expand_dims((iy - iy0.astype(iy.dtype)).astype(x.dtype), axis=1)
    one = mx.array(1.0, dtype=x.dtype)
    wx0 = one - wx1
    wy0 = one - wy1

    if padding_mode == "border":
        ix0 = mx.clip(ix0, 0, width - 1)
        ix1 = mx.clip(ix1, 0, width - 1)
        iy0 = mx.clip(iy0, 0, height - 1)
        iy1 = mx.clip(iy1, 0, height - 1)
        in_x0 = in_x1 = in_y0 = in_y1 = None
    else:
        in_x0 = (ix0 >= 0) & (ix0 < width)
        in_x1 = (ix1 >= 0) & (ix1 < width)
        in_y0 = (iy0 >= 0) & (iy0 < height)
        in_y1 = (iy1 >= 0) & (iy1 < height)
        ix0 = mx.clip(ix0, 0, width - 1)
        ix1 = mx.clip(ix1, 0, width - 1)
        iy0 = mx.clip(iy0, 0, height - 1)
        iy1 = mx.clip(iy1, 0, height - 1)

    flat = x.reshape(batch, channels, height * width)
    v00 = _gather_nchw(flat, iy0, ix0, width, (out_h, out_w))
    v10 = _gather_nchw(flat, iy0, ix1, width, (out_h, out_w))
    v01 = _gather_nchw(flat, iy1, ix0, width, (out_h, out_w))
    v11 = _gather_nchw(flat, iy1, ix1, width, (out_h, out_w))

    if padding_mode == "zeros":
        v00 = v00 * mx.expand_dims(in_x0 & in_y0, axis=1)
        v10 = v10 * mx.expand_dims(in_x1 & in_y0, axis=1)
        v01 = v01 * mx.expand_dims(in_x0 & in_y1, axis=1)
        v11 = v11 * mx.expand_dims(in_x1 & in_y1, axis=1)

    return wx0 * wy0 * v00 + wx1 * wy0 * v10 + wx0 * wy1 * v01 + wx1 * wy1 * v11
