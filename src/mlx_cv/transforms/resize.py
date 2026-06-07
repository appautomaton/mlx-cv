"""Resize / letterbox transforms — each returns ``(image, SpatialTransform)``."""

from __future__ import annotations

import numpy as np

from ..core.geometry import SpatialTransform

__all__ = ["Resize", "Letterbox"]


def _as_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    return (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))


class Resize:
    """Anisotropic resize to ``size`` ``(H, W)`` (bicubic)."""

    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = _as_hw(size)

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, SpatialTransform]:
        from PIL import Image

        h, w = image.shape[:2]
        nh, nw = self.size
        out = np.asarray(Image.fromarray(image).resize((nw, nh), Image.BICUBIC))
        return out, SpatialTransform.resize((h, w), (nh, nw))


class Letterbox:
    """Uniform resize to fit ``size`` ``(H, W)``, padded (centered) to ``size``."""

    def __init__(self, size: int | tuple[int, int], pad_value: int = 114) -> None:
        self.size = _as_hw(size)
        self.pad_value = pad_value

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, SpatialTransform]:
        from PIL import Image

        h, w = image.shape[:2]
        nh, nw = self.size
        s = min(nw / w, nh / h)
        rw, rh = int(round(w * s)), int(round(h * s))
        resized = np.asarray(Image.fromarray(image).resize((rw, rh), Image.BICUBIC))
        canvas = np.full((nh, nw, 3), self.pad_value, dtype=np.uint8)
        ox, oy = (nw - rw) // 2, (nh - rh) // 2
        canvas[oy:oy + rh, ox:ox + rw] = resized
        ctx = SpatialTransform(orig_size=(h, w), scale=(s, s),
                               offset=(float(ox), float(oy)), model_size=(nh, nw))
        return canvas, ctx
