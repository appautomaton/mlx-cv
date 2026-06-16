"""MLX deformable-attention primitives."""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .sampling import bilinear_grid_sample_nchw

__all__ = ["ms_deform_attn_core"]


def _spatial_shapes(value_spatial_shapes) -> list[tuple[int, int]]:
    shapes = np.asarray(value_spatial_shapes, dtype=np.int64)
    if shapes.ndim != 2 or shapes.shape[1] != 2:
        raise ValueError(f"value_spatial_shapes must have shape (L,2), got {shapes.shape}")
    out = [(int(h), int(w)) for h, w in shapes.tolist()]
    if any(h <= 0 or w <= 0 for h, w in out):
        raise ValueError("value_spatial_shapes entries must be positive")
    return out


def ms_deform_attn_core(
    value: mx.array,
    value_spatial_shapes,
    sampling_locations: mx.array,
    attention_weights: mx.array,
) -> mx.array:
    """RF-DETR multi-scale deformable attention core.

    Args:
        value: ``(batch, heads, head_dim, sum(H_l * W_l))``.
        value_spatial_shapes: ``(levels, 2)`` ``(H, W)`` metadata.
        sampling_locations: ``(batch, queries, heads, levels, points, 2)`` in
            normalized ``[0, 1]`` coordinates.
        attention_weights: ``(batch, queries, heads, levels, points)``.

    Returns:
        ``(batch, queries, heads * head_dim)``.
    """

    if len(value.shape) != 4:
        raise ValueError(f"value must have shape (B,H,D,S), got {value.shape}")
    if len(sampling_locations.shape) != 6 or sampling_locations.shape[-1] != 2:
        raise ValueError(
            "sampling_locations must have shape (B,Q,H,L,P,2), "
            f"got {sampling_locations.shape}"
        )
    if len(attention_weights.shape) != 5:
        raise ValueError(f"attention_weights must have shape (B,Q,H,L,P), got {attention_weights.shape}")

    batch, num_heads, head_dim, spatial_size = value.shape
    loc_batch, num_queries, loc_heads, num_levels, num_points, _ = sampling_locations.shape
    if (loc_batch, loc_heads) != (batch, num_heads):
        raise ValueError("sampling_locations batch/head dimensions must match value")
    if attention_weights.shape != (batch, num_queries, num_heads, num_levels, num_points):
        raise ValueError(
            "attention_weights shape must match sampling_locations without the coordinate axis"
        )

    shapes = _spatial_shapes(value_spatial_shapes)
    if len(shapes) != num_levels:
        raise ValueError(f"value_spatial_shapes has {len(shapes)} levels, expected {num_levels}")
    expected_spatial_size = sum(h * w for h, w in shapes)
    if expected_spatial_size != spatial_size:
        raise ValueError(
            f"value spatial size {spatial_size} does not match value_spatial_shapes "
            f"total {expected_spatial_size}"
        )

    sampled_levels: list[mx.array] = []
    start = 0
    for level_index, (height, width) in enumerate(shapes):
        size = height * width
        level_value = value[..., start : start + size]
        level_value = level_value.reshape(batch, num_heads, head_dim, height, width)
        level_value = level_value.reshape(batch * num_heads, head_dim, height, width)

        grid = 2 * sampling_locations[:, :, :, level_index] - 1
        grid = mx.transpose(grid, (0, 2, 1, 3, 4)).reshape(
            batch * num_heads, num_queries, num_points, 2
        )
        sampled = bilinear_grid_sample_nchw(
            level_value,
            grid,
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_levels.append(sampled)
        start += size

    sampled_values = mx.stack(sampled_levels, axis=-2).reshape(
        batch * num_heads, head_dim, num_queries, num_levels * num_points
    )
    weights = mx.transpose(attention_weights, (0, 2, 1, 3, 4)).reshape(
        batch * num_heads, 1, num_queries, num_levels * num_points
    )
    out = mx.sum(sampled_values * weights, axis=-1).reshape(
        batch, num_heads * head_dim, num_queries
    )
    return mx.transpose(out, (0, 2, 1))
