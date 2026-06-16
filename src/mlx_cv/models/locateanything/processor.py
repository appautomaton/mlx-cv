"""Runtime-light LocateAnything processor.

The processor owns image patchification, prompt image-token expansion, and
grounding-token postprocess. It accepts a tokenizer-like object but does not
import Transformers at runtime.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from ...core.base import Processor
from ...core.geometry import SpatialTransform
from ...core.types import Detections, Points, Result
from .config import LocateAnythingConfig
from .decode import TokenScheme, parse_grounding_tokens

__all__ = [
    "LocateAnythingProcessor",
    "LocateAnythingProcessorConfig",
    "LocateAnythingProcessorContext",
]


@dataclass
class LocateAnythingProcessorConfig:
    patch_size: int = 14
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    in_token_limit: int = 25600
    merge_kernel_size: tuple[int, int] = (2, 2)
    image_token: str = "<IMG_CONTEXT>"
    image_start_token: str = "<img>"
    image_end_token: str = "</img>"

    @classmethod
    def from_model_config(cls, config: LocateAnythingConfig) -> "LocateAnythingProcessorConfig":
        vision = config.vision_config
        return cls(
            patch_size=vision.patch_size,
            merge_kernel_size=tuple(vision.merge_kernel_size),
        )


@dataclass
class LocateAnythingProcessorContext:
    transform: SpatialTransform
    image_size: tuple[int, int]
    model_size: tuple[int, int]
    image_grid_hws: tuple[tuple[int, int], ...]
    expanded_text: list[str] | None = None


class LocateAnythingProcessor(Processor):
    def __init__(
        self,
        config: LocateAnythingProcessorConfig | LocateAnythingConfig | None = None,
        *,
        tokenizer: Any | None = None,
    ) -> None:
        if isinstance(config, LocateAnythingConfig):
            self.config = LocateAnythingProcessorConfig.from_model_config(config)
            self.model_config = config
        else:
            self.config = config or LocateAnythingProcessorConfig()
            self.model_config = LocateAnythingConfig()
        self.tokenizer = tokenizer

    def _to_pil(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, mx.array):
            arr = np.array(image)
        else:
            arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"image must be HWC or CHW, got shape {arr.shape}")
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr[..., :3], mode="RGB")

    def _rescale(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        patch = self.config.patch_size
        tokens = (width // patch) * (height // patch)
        if tokens > self.config.in_token_limit:
            scale = math.sqrt(self.config.in_token_limit / tokens)
            width = max(int(width * scale), patch)
            height = max(int(height * scale), patch)
            image = image.resize((width, height), Image.Resampling.BICUBIC)

        width, height = image.size
        merge_h, merge_w = self.config.merge_kernel_size
        step_w = merge_w * patch
        step_h = merge_h * patch
        target_w = math.ceil(width / step_w) * step_w
        target_h = math.ceil(height / step_h) * step_h
        if (target_w, target_h) != (width, height):
            image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

        grid_h = target_h // patch
        grid_w = target_w // patch
        if grid_h >= 512 or grid_w >= 512:
            raise ValueError("LocateAnything image grid exceeds MoonViT RoPE limits")
        return image

    def _image_to_tensor(self, image: Image.Image) -> mx.array:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = mx.array(arr).transpose(2, 0, 1)
        mean = mx.array(self.config.image_mean, dtype=mx.float32).reshape(3, 1, 1)
        std = mx.array(self.config.image_std, dtype=mx.float32).reshape(3, 1, 1)
        return (tensor - mean) / std

    def _patchify(self, image: mx.array) -> tuple[mx.array, tuple[int, int]]:
        patch = self.config.patch_size
        channels, height, width = image.shape
        patches = image.reshape(channels, height // patch, patch, width // patch, patch)
        patches = patches.transpose(1, 3, 0, 2, 4).reshape(-1, channels, patch, patch)
        return patches, (height // patch, width // patch)

    def _tokenize(self, texts: list[str]) -> dict[str, mx.array]:
        if self.tokenizer is None:
            return {}
        tokenized = self.tokenizer(texts, padding=True)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is None:
            attention_mask = [[1] * len(row) for row in input_ids]
        return {
            "input_ids": mx.array(input_ids, dtype=mx.int32),
            "attention_mask": mx.array(attention_mask, dtype=mx.int32),
        }

    def _image_token_id(self) -> int | None:
        if self.tokenizer is None or not hasattr(self.tokenizer, "convert_tokens_to_ids"):
            return None
        token_id = self.tokenizer.convert_tokens_to_ids(self.config.image_token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if token_id is None or token_id == unk_id:
            return None
        return int(token_id)

    def _expand_placeholders(self, texts: list[str], grid_hws: list[tuple[int, int]]) -> list[str]:
        merge_h, merge_w = self.config.merge_kernel_size
        merge_len = merge_h * merge_w
        pattern = re.compile(r"<image-\d+>")
        image_ix = 0

        def expand(_match):
            nonlocal image_ix
            if image_ix >= len(grid_hws):
                raise ValueError("found more image placeholders than images")
            h, w = grid_hws[image_ix]
            count = (h * w) // merge_len
            image_ix += 1
            return self.config.image_start_token + (self.config.image_token * count) + self.config.image_end_token

        out = [pattern.sub(expand, text) for text in texts]
        if image_ix != len(grid_hws):
            raise ValueError(f"number of image placeholders ({image_ix}) does not match images ({len(grid_hws)})")
        return out

    def preprocess(self, inputs: Any, prompt: str | list[str] | None = None):
        if isinstance(inputs, dict):
            images = inputs.get("images", inputs.get("image"))
            prompt = inputs.get("prompt", prompt)
        else:
            images = inputs
        if images is None:
            raise ValueError("LocateAnythingProcessor.preprocess requires an image")
        if isinstance(images, (Image.Image, np.ndarray, mx.array)):
            images = [images]

        pixel_values = []
        grid_hws: list[tuple[int, int]] = []
        transforms: list[SpatialTransform] = []
        image_sizes: list[tuple[int, int]] = []
        model_sizes: list[tuple[int, int]] = []
        for image in images:
            original = self._to_pil(image)
            orig_w, orig_h = original.size
            resized = self._rescale(original)
            model_w, model_h = resized.size
            tensor = self._image_to_tensor(resized)
            patches, grid_hw = self._patchify(tensor)
            pixel_values.append(patches)
            grid_hws.append(grid_hw)
            image_sizes.append((orig_h, orig_w))
            model_sizes.append((model_h, model_w))
            transforms.append(SpatialTransform.resize((orig_h, orig_w), (model_h, model_w)))

        if len(transforms) != 1:
            raise ValueError("LocateAnythingProcessor currently supports one image per Result")

        model_inputs: dict[str, Any] = {
            "pixel_values": mx.concatenate(pixel_values, axis=0),
            "image_grid_hws": mx.array(grid_hws, dtype=mx.int32),
        }
        expanded_text = None
        if prompt is not None:
            texts = [prompt] if isinstance(prompt, str) else list(prompt)
            expanded_text = self._expand_placeholders(texts, grid_hws)
            model_inputs.update(self._tokenize(expanded_text))
            image_token_id = self._image_token_id()
            if image_token_id is not None:
                model_inputs["image_token_id"] = image_token_id

        ctx = LocateAnythingProcessorContext(
            transform=transforms[0],
            image_size=image_sizes[0],
            model_size=model_sizes[0],
            image_grid_hws=tuple(grid_hws),
            expanded_text=expanded_text,
        )
        return model_inputs, ctx

    def _decode_label(self, label) -> str | None:
        if label is None:
            return None
        if isinstance(label, str):
            return label
        if self.tokenizer is None:
            return None
        if hasattr(self.tokenizer, "decode"):
            return str(self.tokenizer.decode(label, skip_special_tokens=True)).strip()
        if hasattr(self.tokenizer, "batch_decode"):
            return str(self.tokenizer.batch_decode([label], skip_special_tokens=True)[0]).strip()
        return None

    def _coords_to_model_box(self, coords: list[int], ctx: LocateAnythingProcessorContext) -> list[float]:
        model_h, model_w = ctx.model_size
        return [
            coords[0] / 1000.0 * model_w,
            coords[1] / 1000.0 * model_h,
            coords[2] / 1000.0 * model_w,
            coords[3] / 1000.0 * model_h,
        ]

    def _coords_to_model_point(self, coords: list[int], ctx: LocateAnythingProcessorContext) -> list[float]:
        model_h, model_w = ctx.model_size
        return [coords[0] / 1000.0 * model_w, coords[1] / 1000.0 * model_h]

    def postprocess(self, raw: Any, ctx: LocateAnythingProcessorContext) -> Result:
        token_ids = raw.get("token_ids", raw) if isinstance(raw, dict) else raw
        scheme = TokenScheme.from_config(self.model_config)
        items = parse_grounding_tokens(token_ids, scheme)

        boxes: list[list[float]] = []
        box_labels: list[str | None] = []
        points: list[list[float]] = []
        point_labels: list[str | None] = []
        for item in items:
            label = self._decode_label(item.label)
            if item.kind == "box":
                model_box = self._coords_to_model_box(item.coords, ctx)
                boxes.append(ctx.transform.invert_boxes([model_box], clip=True)[0].tolist())
                box_labels.append(label)
            elif item.kind == "point":
                model_point = self._coords_to_model_point(item.coords, ctx)
                points.append(ctx.transform.invert_points([model_point], clip=True)[0].tolist())
                point_labels.append(label)

        detections = Detections(np.asarray(boxes), labels=box_labels) if boxes else None
        sparse_points = Points(np.asarray(points), labels=point_labels) if points else None
        return Result(image_size=ctx.image_size, detections=detections, points=sparse_points)
