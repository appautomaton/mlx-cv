"""A transform maps an image to model input and returns ``(array, ctx)`` (§5.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..core.geometry import SpatialTransform

__all__ = ["Transform"]


@runtime_checkable
class Transform(Protocol):
    """Spatial transforms emit a :class:`SpatialTransform` recording their geometry."""

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, SpatialTransform]: ...
