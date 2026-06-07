"""Pixel-space normalization helpers (no coordinate effect, so no ctx)."""

from __future__ import annotations

import numpy as np

__all__ = ["normalize", "to_chw"]


def normalize(image: np.ndarray,
              mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
              std: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> np.ndarray:
    """Scale ``uint8`` HWC to ``[0,1]`` then ``(x - mean) / std`` (float32)."""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return (arr - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)


def to_chw(image: np.ndarray) -> np.ndarray:
    """HWC -> CHW."""
    return np.transpose(image, (2, 0, 1))
