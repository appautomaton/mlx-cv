"""Prompt taxonomy (opt-in, §5.5): Text / Point / Box / Exemplar.

A promptable model accepts one of these; others ignore the ``prompt`` arg. The
grounding anchor (LocateAnything) exercises only :class:`TextPrompt` (§16.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

__all__ = ["TextPrompt", "PointPrompt", "BoxPrompt", "ExemplarPrompt", "Prompt"]


@dataclass
class TextPrompt:
    """Free text: category names, referring expressions, OCR/GUI/pointing queries."""

    text: str


@dataclass
class PointPrompt:
    """Click points ``(N, 2)`` xy, with optional ``labels`` (1 = fg, 0 = bg)."""

    points: np.ndarray
    labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        if self.labels is not None:
            self.labels = np.asarray(self.labels).reshape(-1)


@dataclass
class BoxPrompt:
    """Prompt boxes ``(N, 4)`` xyxy."""

    boxes: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)


@dataclass
class ExemplarPrompt:
    """Visual exemplar(s): an image and ``(N, 4)`` boxes marking the example object(s)."""

    image: np.ndarray
    boxes: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)


Prompt = Union[TextPrompt, PointPrompt, BoxPrompt, ExemplarPrompt]
