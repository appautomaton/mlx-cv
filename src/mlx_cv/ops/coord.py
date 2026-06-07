"""Coordinate-token decoding — e.g. LocateAnything's vocab-token scheme (§16.2).

LocateAnything emits coordinates as dedicated vocabulary tokens ``<0>``..``<1000>``;
the integer value is ``token_id - coord_start`` (e.g. ``<64>`` = 151741 - 151677).
Those normalized ``[0, range]`` values are then scaled to original pixels via the
``SpatialTransform`` ctx. This op is the reusable, model-agnostic piece.
"""

from __future__ import annotations

import numpy as np

__all__ = ["token_to_coord", "coords_to_pixels"]


def token_to_coord(token_id: int, coord_start: int, coord_max: int = 1000) -> int:
    """Map a coordinate *token id* to its integer value in ``[0, coord_max]``."""
    return max(0, min(coord_max, int(token_id) - int(coord_start)))


def coords_to_pixels(coords, size_hw: tuple[int, int], coord_range: int = 1000) -> np.ndarray:
    """Scale normalized ``[0, coord_range]`` ``(..., 2)`` xy coords to pixels in ``(H, W)``."""
    h, w = size_hw
    c = np.asarray(coords, dtype=np.float64)
    out = c.copy()
    out[..., 0] = c[..., 0] / coord_range * w
    out[..., 1] = c[..., 1] / coord_range * h
    return out
