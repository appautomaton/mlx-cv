"""Dependency-light rendering for shared :class:`mlx_cv.Result` objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from .core.image import load_image

if TYPE_CHECKING:
    from .core.types import Result

__all__ = ["draw_result"]


_DEFAULT_PALETTE = (
    "#2F80ED",
    "#EB5757",
    "#27AE60",
    "#F2C94C",
    "#9B51E0",
    "#F2994A",
    "#56CCF2",
    "#6FCF97",
)


def _rgb(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    return ImageColor.getrgb(value) if isinstance(value, str) else tuple(int(v) for v in value)


def _palette(values: Sequence[str | tuple[int, int, int]] | None):
    colors = tuple(_rgb(value) for value in (values or _DEFAULT_PALETTE))
    if not colors:
        raise ValueError("palette must contain at least one color")
    return colors


def _base_image(result: "Result", image: Any) -> Image.Image:
    height, width = (int(result.image_size[0]), int(result.image_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Result.image_size must be positive, got {result.image_size}")
    if image is None:
        return Image.new("RGB", (width, height), "white")
    array, image_size = load_image(image)
    if tuple(image_size) != (height, width):
        raise ValueError(
            f"draw image has size {tuple(image_size)}, expected Result.image_size {(height, width)}"
        )
    return Image.fromarray(array, mode="RGB")


def _alpha_overlay(base: Image.Image, color: tuple[int, int, int], mask, alpha: float) -> None:
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != (base.height, base.width):
        raise ValueError(
            f"mask shape {mask_array.shape} does not match image size {(base.height, base.width)}"
        )
    layer = Image.new("RGBA", base.size, (*color, 0))
    layer.putalpha(Image.fromarray(mask_array.astype(np.uint8) * round(alpha * 255), mode="L"))
    base.alpha_composite(layer)


def _draw_masks(base: Image.Image, result: "Result", colors, alpha: float) -> None:
    masks = result.masks
    if masks is None:
        return
    data = np.asarray(masks.data)
    if data.ndim == 3:
        for index, mask in enumerate(data):
            _alpha_overlay(base, colors[index % len(colors)], mask, alpha)
        return
    if data.ndim != 2:
        raise ValueError(f"Result masks must have shape (N,H,W) or (H,W), got {data.shape}")
    if masks.kind == "semantic" or masks.kind == "panoptic":
        values = [value for value in np.unique(data) if value != 0]
        for index, value in enumerate(values):
            _alpha_overlay(base, colors[index % len(colors)], data == value, alpha)
    else:
        _alpha_overlay(base, colors[0], data.astype(bool), alpha)


def _depth_overlay(base: Image.Image, result: "Result", alpha: float) -> None:
    if result.depth is None:
        return
    depth = np.asarray(result.depth.depth, dtype=np.float64)
    if depth.shape != (base.height, base.width):
        raise ValueError(
            f"depth shape {depth.shape} does not match image size {(base.height, base.width)}"
        )
    valid = np.isfinite(depth)
    if not np.any(valid):
        return
    low, high = np.percentile(depth[valid], (2.0, 98.0))
    scale = max(float(high - low), np.finfo(np.float64).eps)
    normalized = np.clip((depth - low) / scale, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    rgba = np.concatenate(
        [rgb * 255.0, (valid.astype(np.float64) * alpha * 255.0)[..., None]],
        axis=-1,
    ).astype(np.uint8)
    base.alpha_composite(Image.fromarray(rgba, mode="RGBA"))


def _label_for(result: "Result", index: int, *, show_labels: bool, show_scores: bool) -> str:
    detections = result.detections
    if detections is None:
        return ""
    parts: list[str] = []
    if show_labels:
        if detections.labels is not None:
            parts.append(str(detections.labels[index]))
        elif detections.class_ids is not None:
            parts.append(str(int(detections.class_ids[index])))
        if detections.track_ids is not None:
            parts.append(f"track {int(detections.track_ids[index])}")
        elif result.tracks is not None and index < len(result.tracks):
            parts.append(f"track {int(result.tracks.ids[index])}")
    if show_scores and detections.scores is not None:
        parts.append(f"{float(detections.scores[index]):.2f}")
    return " ".join(parts)


def _draw_label(draw: ImageDraw.ImageDraw, xy, text: str, color) -> None:
    if not text:
        return
    x, y = (float(xy[0]), float(xy[1]))
    box = draw.textbbox((x, y), text)
    draw.rectangle(box, fill=(*color, 220))
    draw.text((x, y), text, fill=(255, 255, 255, 255))


def draw_result(
    result: "Result",
    image: Any = None,
    *,
    palette: Sequence[str | tuple[int, int, int]] | None = None,
    line_width: int = 3,
    point_radius: int = 4,
    mask_alpha: float = 0.35,
    depth_alpha: float = 0.65,
    show_labels: bool = True,
    show_scores: bool = True,
    show_depth: bool | None = None,
) -> Image.Image:
    """Render populated result modalities and return a new RGB Pillow image."""

    if line_width <= 0 or point_radius <= 0:
        raise ValueError("line_width and point_radius must be positive")
    if not 0.0 <= mask_alpha <= 1.0 or not 0.0 <= depth_alpha <= 1.0:
        raise ValueError("mask_alpha and depth_alpha must be between 0 and 1")
    colors = _palette(palette)
    base = _base_image(result, image).convert("RGBA")
    if show_depth is None:
        show_depth = result.depth is not None and result.masks is None
    if show_depth:
        _depth_overlay(base, result, depth_alpha)
    _draw_masks(base, result, colors, mask_alpha)

    draw = ImageDraw.Draw(base, mode="RGBA")
    if result.detections is not None:
        for index, box in enumerate(result.detections.boxes):
            color = colors[index % len(colors)]
            xy = tuple(float(value) for value in box)
            draw.rectangle(xy, outline=(*color, 255), width=line_width)
            _draw_label(
                draw,
                (xy[0], max(0.0, xy[1] - 12.0)),
                _label_for(
                    result,
                    index,
                    show_labels=show_labels,
                    show_scores=show_scores,
                ),
                color,
            )

    if result.points is not None:
        for index, point in enumerate(result.points.points):
            color = colors[index % len(colors)]
            x, y = (float(point[0]), float(point[1]))
            draw.ellipse(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill=(*color, 255),
            )

    if result.keypoints is not None:
        points = np.asarray(result.keypoints.keypoints)
        instances = points[None] if points.ndim == 2 else points
        for instance_index, instance in enumerate(instances):
            color = colors[instance_index % len(colors)]
            if result.keypoints.skeleton is not None:
                for start, end in result.keypoints.skeleton:
                    draw.line(
                        tuple(float(v) for v in (*instance[start, :2], *instance[end, :2])),
                        fill=(*color, 255),
                        width=line_width,
                    )
            for point in instance:
                x, y = (float(point[0]), float(point[1]))
                draw.ellipse(
                    (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                    fill=(*color, 255),
                )

    return base.convert("RGB")
