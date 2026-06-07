"""Load arbitrary image inputs to a canonical ``(H, W, 3)`` uint8 RGB array + size."""

from __future__ import annotations

import numpy as np

__all__ = ["load_image"]


def load_image(src) -> tuple[np.ndarray, tuple[int, int]]:
    """Return ``(rgb_uint8_HxWx3, (H, W))`` from a path, numpy array, or PIL image."""
    if isinstance(src, np.ndarray):
        arr = src
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"expected (H, W), (H, W, 3) or (H, W, 4); got {arr.shape}")
        arr = np.ascontiguousarray(arr[..., :3])
        h, w = arr.shape[:2]
        return arr, (h, w)

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        Image = None

    if Image is not None and isinstance(src, Image.Image):
        arr = np.asarray(src.convert("RGB"))
        return arr, (arr.shape[0], arr.shape[1])

    if isinstance(src, str) or hasattr(src, "__fspath__"):
        from PIL import Image as _Image
        arr = np.asarray(_Image.open(src).convert("RGB"))
        return arr, (arr.shape[0], arr.shape[1])

    raise TypeError(f"unsupported image source: {type(src)!r}")
