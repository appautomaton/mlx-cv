"""Invertible spatial coordinate transforms — the spine's coordinate discipline.

Every model runs on a preprocessed image (resized / letterboxed / cropped). A
``SpatialTransform`` records exactly how the original image was mapped into model
space, so any coordinate the model emits can be mapped *losslessly* back to the
original image. Preprocess always returns ``(tensor, ctx)``; postprocess always
consumes ``ctx``. See ARCHITECTURE.md §5.2 ("coordinates are sacred").

The mapping is a per-axis affine ``model = orig * scale + offset``:

* plain resize  -> scale = new/orig,           offset = 0
* letterbox     -> scale = min ratio (uniform), offset = pad (left, top)
* center crop   -> scale = 1,                    offset = -(crop origin)

That single form covers what real preprocessors do, and inverts exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SpatialTransform"]


def _sample(src: np.ndarray, ys: np.ndarray, xs: np.ndarray, mode: str, fill) -> np.ndarray:
    """Sample ``src`` at fractional ``(ys, xs)`` source coords (inverse warp).

    ``nearest`` preserves label/dtype (for masks); ``bilinear`` returns ``float64``
    (for depth / heatmaps). Out-of-domain samples are set to ``fill``.
    """
    src = np.asarray(src)
    H, W = src.shape[:2]
    extra = src.shape[2:]
    if mode == "nearest":
        xi = np.floor(xs + 0.5).astype(np.int64)
        yi = np.floor(ys + 0.5).astype(np.int64)
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out = np.empty(ys.shape + extra, dtype=src.dtype)
        out[...] = fill
        sampled = src[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)]
        out[inside] = sampled[inside]
        return out
    if mode == "bilinear":
        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        x1, y1 = x0 + 1, y0 + 1
        wx = (xs - x0).reshape(xs.shape + (1,) * len(extra))
        wy = (ys - y0).reshape(ys.shape + (1,) * len(extra))
        x0c, x1c = np.clip(x0, 0, W - 1), np.clip(x1, 0, W - 1)
        y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y1, 0, H - 1)
        f = src.astype(np.float64)
        top = f[y0c, x0c] * (1 - wx) + f[y0c, x1c] * wx
        bot = f[y1c, x0c] * (1 - wx) + f[y1c, x1c] * wx
        out = top * (1 - wy) + bot * wy
        inside = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
        out[~inside] = fill
        return out
    raise ValueError(f"unknown resampling mode {mode!r}; use 'nearest' or 'bilinear'")


@dataclass(frozen=True)
class SpatialTransform:
    """Records orig->model image geometry and inverts coordinates back.

    Points are ``(x, y)``; boxes are ``xyxy``. Inputs accept array-likes; outputs
    are ``float64`` numpy arrays with the same leading shape.

    Attributes:
        orig_size:  original image size ``(H, W)``.
        scale:      per-axis scale ``(sx, sy)`` mapping orig->model.
        offset:     per-axis offset ``(ox, oy)`` in model pixels.
        model_size: model-input size ``(H, W)`` (for clipping / record).
    """

    orig_size: tuple[int, int]
    scale: tuple[float, float] = (1.0, 1.0)
    offset: tuple[float, float] = (0.0, 0.0)
    model_size: tuple[int, int] | None = None

    # -- constructors -------------------------------------------------------
    @classmethod
    def identity(cls, orig_size: tuple[int, int]) -> "SpatialTransform":
        return cls(orig_size=orig_size, model_size=orig_size)

    @classmethod
    def resize(cls, orig_size: tuple[int, int], new_size: tuple[int, int]) -> "SpatialTransform":
        """Anisotropic resize: orig ``(H, W)`` -> new ``(H, W)``."""
        oh, ow = orig_size
        nh, nw = new_size
        return cls(orig_size=(oh, ow), scale=(nw / ow, nh / oh),
                   offset=(0.0, 0.0), model_size=(nh, nw))

    @classmethod
    def letterbox(cls, orig_size: tuple[int, int], new_size: tuple[int, int],
                  *, center: bool = True) -> "SpatialTransform":
        """Uniform resize to fit inside ``new`` ``(H, W)``, then pad to ``new``."""
        oh, ow = orig_size
        nh, nw = new_size
        s = min(nw / ow, nh / oh)
        if center:
            ox = (nw - ow * s) / 2.0
            oy = (nh - oh * s) / 2.0
        else:
            ox = oy = 0.0
        return cls(orig_size=(oh, ow), scale=(s, s), offset=(ox, oy), model_size=(nh, nw))

    @classmethod
    def crop(cls, orig_size: tuple[int, int], box_xyxy) -> "SpatialTransform":
        """Crop the original to ``(x0, y0, x1, y1)``; model space *is* the crop."""
        x0, y0, x1, y1 = (float(v) for v in box_xyxy)
        return cls(orig_size=orig_size, scale=(1.0, 1.0), offset=(-x0, -y0),
                   model_size=(int(round(y1 - y0)), int(round(x1 - x0))))

    # -- forward (orig -> model) -------------------------------------------
    def apply_points(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        sx, sy = self.scale
        ox, oy = self.offset
        out = pts.copy()
        out[..., 0] = pts[..., 0] * sx + ox
        out[..., 1] = pts[..., 1] * sy + oy
        return out

    def apply_boxes(self, boxes) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64)
        sx, sy = self.scale
        ox, oy = self.offset
        out = boxes.copy()
        out[..., 0::2] = boxes[..., 0::2] * sx + ox
        out[..., 1::2] = boxes[..., 1::2] * sy + oy
        return out

    # -- inverse (model -> orig) -------------------------------------------
    def invert_points(self, pts, *, clip: bool = False) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        sx, sy = self.scale
        ox, oy = self.offset
        out = pts.copy()
        out[..., 0] = (pts[..., 0] - ox) / sx
        out[..., 1] = (pts[..., 1] - oy) / sy
        return self._clip_points(out) if clip else out

    def invert_boxes(self, boxes, *, clip: bool = False) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64)
        sx, sy = self.scale
        ox, oy = self.offset
        out = boxes.copy()
        out[..., 0::2] = (boxes[..., 0::2] - ox) / sx
        out[..., 1::2] = (boxes[..., 1::2] - oy) / sy
        return self._clip_boxes(out) if clip else out

    # -- dense maps (mask / depth / heatmap) -------------------------------
    # Coordinates invert exactly (affine); dense maps invert via a *documented*
    # deterministic resampling policy (ARCHITECTURE §5.2, BUILDING-BLOCKS #9):
    #   masks            -> nearest  (labels must not be interpolated)
    #   depth / heatmap  -> bilinear (smooth fields; linear ramps round-trip exactly)
    # Both directions are inverse warps under the per-axis affine; out-of-domain
    # samples (e.g. letterbox padding) are filled with ``fill``.
    def _require_model_size(self) -> tuple[int, int]:
        if self.model_size is None:
            raise ValueError("dense resampling needs model_size; build via "
                             "resize()/letterbox()/crop()/identity(), not the raw ctor")
        return self.model_size

    def apply_dense(self, dense, *, mode: str = "bilinear", fill=0.0) -> np.ndarray:
        """Resample an original-space dense map into model space ``(H, W)``."""
        dense = np.asarray(dense)
        mh, mw = self._require_model_size()
        sx, sy = self.scale
        ox, oy = self.offset
        yy, xx = np.meshgrid(np.arange(mh, dtype=np.float64),
                             np.arange(mw, dtype=np.float64), indexing="ij")
        return _sample(dense, (yy - oy) / sy, (xx - ox) / sx, mode, fill)

    def invert_dense(self, dense, *, mode: str = "bilinear", fill=0.0) -> np.ndarray:
        """Resample a model-space dense map back to original space ``(H, W)``."""
        dense = np.asarray(dense)
        oh, ow = self.orig_size
        sx, sy = self.scale
        ox, oy = self.offset
        yy, xx = np.meshgrid(np.arange(oh, dtype=np.float64),
                             np.arange(ow, dtype=np.float64), indexing="ij")
        return _sample(dense, yy * sy + oy, xx * sx + ox, mode, fill)

    def invert_mask(self, mask, *, fill=0) -> np.ndarray:
        """Model-space mask -> original space, nearest (labels preserved)."""
        return self.invert_dense(mask, mode="nearest", fill=fill)

    def invert_depth(self, depth, *, fill=0.0) -> np.ndarray:
        """Model-space depth -> original space, bilinear."""
        return self.invert_dense(depth, mode="bilinear", fill=fill)

    def invert_heatmap(self, heatmap, *, fill=0.0) -> np.ndarray:
        """Model-space heatmap -> original space, bilinear."""
        return self.invert_dense(heatmap, mode="bilinear", fill=fill)

    # -- helpers ------------------------------------------------------------
    def _clip_points(self, out: np.ndarray) -> np.ndarray:
        h, w = self.orig_size
        out[..., 0] = np.clip(out[..., 0], 0, w)
        out[..., 1] = np.clip(out[..., 1], 0, h)
        return out

    def _clip_boxes(self, out: np.ndarray) -> np.ndarray:
        h, w = self.orig_size
        out[..., 0::2] = np.clip(out[..., 0::2], 0, w)
        out[..., 1::2] = np.clip(out[..., 1::2], 0, h)
        return out
