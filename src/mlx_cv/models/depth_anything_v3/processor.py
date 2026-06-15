"""Depth Anything V3 monocular preprocessing and postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import mlx.core as mx

from ...core.base import Processor
from ...core.geometry import SpatialTransform
from ...core.image import load_image
from ...core.types import DepthMap, Result
from ...transforms.resize import Letterbox

__all__ = ["DA3ProcessorConfig", "DA3Processor"]


@dataclass(frozen=True)
class DA3ProcessorConfig:
    process_res: int = 518
    patch_size: int = 14
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def model_size(self) -> int:
        return max(self.patch_size, round(self.process_res / self.patch_size) * self.patch_size)


class DA3Processor(Processor):
    """One-image DA3 processor returning NCHW MLX tensors and `Result.depth`."""

    def __init__(self, config: DA3ProcessorConfig | None = None) -> None:
        self.config = config or DA3ProcessorConfig()

    def preprocess(self, inputs: Any) -> tuple[mx.array, SpatialTransform]:
        image, _ = load_image(inputs)
        resized, ctx = Letterbox(self.config.model_size, pad_value=0)(image)
        x = resized.astype(np.float32) / 255.0
        mean = np.asarray(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.config.std, dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None]
        return mx.array(np.ascontiguousarray(x)), ctx

    def postprocess(self, raw: Any, ctx: SpatialTransform) -> Result:
        data = raw.data if hasattr(raw, "data") else raw
        depth = np.asarray(data["depth"])
        if depth.ndim == 3:
            depth = depth[0]
        depth_conf = data.get("depth_conf") if hasattr(data, "get") else None
        if depth_conf is not None:
            depth_conf = np.asarray(depth_conf)
            if depth_conf.ndim == 3:
                depth_conf = depth_conf[0]
            depth_conf = ctx.invert_dense(depth_conf, mode="bilinear")
        depth_orig = ctx.invert_depth(depth)
        return Result(
            image_size=ctx.orig_size,
            depth=DepthMap(depth=depth_orig, depth_conf=depth_conf),
        )
