"""Official SAM 3.1 image/video preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ...core.geometry import SpatialTransform
from ...core.image import load_image

__all__ = [
    "SAM3FrameContext",
    "SAM3ProcessorContext",
    "SAM3VideoProcessor",
    "SAM3VideoProcessorConfig",
]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _as_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    return (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))


@dataclass(frozen=True)
class SAM3VideoProcessorConfig:
    image_size: int | tuple[int, int] = 1008
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __post_init__(self) -> None:
        if min(_as_hw(self.image_size)) <= 0:
            raise ValueError("SAM 3.1 image_size must be positive")
        if len(self.mean) != 3 or len(self.std) != 3 or any(v == 0 for v in self.std):
            raise ValueError("SAM 3.1 mean/std must contain three non-zero channels")

    @property
    def model_size(self) -> tuple[int, int]:
        return _as_hw(self.image_size)


@dataclass(frozen=True)
class SAM3FrameContext:
    frame_index: int
    image_size: tuple[int, int]
    model_size: tuple[int, int]
    transform: SpatialTransform
    source: str | None = None


@dataclass(frozen=True)
class SAM3ProcessorContext:
    frames: tuple[SAM3FrameContext, ...]
    model_size: tuple[int, int]

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def _frame_sources(inputs: Any) -> list[Any]:
    if isinstance(inputs, dict):
        inputs = inputs.get("frames", inputs.get("resource_path", inputs.get("video")))
    if inputs is None:
        raise ValueError("SAM 3.1 preprocessing requires an image or frames")
    if isinstance(inputs, (str, Path)) or hasattr(inputs, "__fspath__"):
        path = Path(inputs)
        if path.is_dir():
            frames = [p for p in sorted(path.iterdir()) if p.suffix.lower() in _IMAGE_SUFFIXES]
            if not frames:
                raise ValueError(f"SAM 3.1 frame directory contains no images: {path}")
            return frames
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            return [path]
        raise ValueError("SAM 3.1 accepts image files or frame directories")
    if isinstance(inputs, np.ndarray):
        return [inputs[index] for index in range(inputs.shape[0])] if inputs.ndim == 4 else [inputs]
    try:
        frames = list(inputs)
    except TypeError:
        return [inputs]
    if not frames:
        raise ValueError("SAM 3.1 requires at least one frame")
    return frames


class SAM3VideoProcessor:
    """Deterministic official 1008px bilinear frame preprocessing."""

    def __init__(self, config: SAM3VideoProcessorConfig | None = None) -> None:
        self.config = config or SAM3VideoProcessorConfig()

    def preprocess(self, inputs: Any) -> tuple[dict[str, np.ndarray], SAM3ProcessorContext]:
        mean = np.asarray(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.config.std, dtype=np.float32).reshape(1, 1, 3)
        out_h, out_w = self.config.model_size
        tensors: list[np.ndarray] = []
        contexts: list[SAM3FrameContext] = []
        for index, source in enumerate(_frame_sources(inputs)):
            image, image_size = load_image(source)
            height, width = image.shape[:2]
            resized = np.asarray(
                Image.fromarray(image).resize((out_w, out_h), Image.Resampling.BILINEAR)
            )
            normalized = (resized.astype(np.float32) / 255.0 - mean) / std
            tensors.append(np.transpose(normalized, (2, 0, 1)))
            contexts.append(
                SAM3FrameContext(
                    index,
                    image_size,
                    self.config.model_size,
                    SpatialTransform.resize((height, width), self.config.model_size),
                    str(source) if isinstance(source, (str, Path)) or hasattr(source, "__fspath__") else None,
                )
            )
        return {"pixel_values": np.ascontiguousarray(np.stack(tensors))}, SAM3ProcessorContext(
            tuple(contexts), self.config.model_size
        )
