"""Depth Anything V3 preprocessing and postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import mlx.core as mx

from ...core.base import Processor
from ...core.geometry import SpatialTransform
from ...core.image import load_image
from ...core.types import CameraGeometry, DepthMap, Result
from ...transforms.resize import Letterbox

__all__ = ["DA3ProcessorConfig", "DA3MultiViewContext", "DA3Processor"]


@dataclass(frozen=True)
class DA3ProcessorConfig:
    process_res: int = 518
    patch_size: int = 14
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def model_size(self) -> int:
        return max(self.patch_size, round(self.process_res / self.patch_size) * self.patch_size)


@dataclass(frozen=True)
class DA3MultiViewContext:
    """Per-view inverse transforms and optional input camera geometry."""

    views: tuple[SpatialTransform, ...]
    camera_geometry: CameraGeometry | None = None

    def __post_init__(self) -> None:
        if len(self.views) < 1:
            raise ValueError("DA3MultiViewContext requires at least one view")
        if self.camera_geometry is not None and self.camera_geometry.view_count != len(self.views):
            raise ValueError(
                f"camera_geometry has {self.camera_geometry.view_count} views, "
                f"expected {len(self.views)}"
            )

    @property
    def view_count(self) -> int:
        return len(self.views)

    @property
    def orig_size(self) -> tuple[int, int]:
        return self.views[0].orig_size


class DA3Processor(Processor):
    """DA3 processor for one image or a still-image set.

    Single-image inputs preserve the original ``(1,3,H,W)`` tensor contract.
    Multi-view inputs use ``(1,V,3,H,W)`` so later DA3 any-view modules can keep
    a distinct image-set axis.
    """

    def __init__(self, config: DA3ProcessorConfig | None = None) -> None:
        self.config = config or DA3ProcessorConfig()

    def preprocess(self, inputs: Any) -> tuple[mx.array, SpatialTransform | DA3MultiViewContext]:
        if _is_multiview_input(inputs):
            images, camera_geometry = self._parse_multiview_inputs(inputs)
            tensors: list[np.ndarray] = []
            contexts: list[SpatialTransform] = []
            sizes: list[tuple[int, int]] = []
            for image_input in images:
                image, size = load_image(image_input)
                sizes.append(size)
                tensor, ctx = self._preprocess_image(image)
                tensors.append(tensor)
                contexts.append(ctx)
            if any(size != sizes[0] for size in sizes):
                raise ValueError(
                    f"DA3 multi-view preprocessing requires same-size still images; got {sizes}. "
                    "Use explicit per-view DepthMap entries for mixed-size outputs."
                )
            x = np.stack(tensors, axis=0)[None]
            ctx = DA3MultiViewContext(tuple(contexts), camera_geometry=camera_geometry)
            return mx.array(np.ascontiguousarray(x)), ctx

        image, _ = load_image(inputs)
        x, ctx = self._preprocess_image(image)
        return mx.array(np.ascontiguousarray(x[None])), ctx

    def _preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, SpatialTransform]:
        resized, ctx = Letterbox(self.config.model_size, pad_value=0)(image)
        x = resized.astype(np.float32) / 255.0
        mean = np.asarray(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.config.std, dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std
        return np.transpose(x, (2, 0, 1)), ctx

    def _parse_multiview_inputs(self, inputs: Any) -> tuple[tuple[Any, ...], CameraGeometry | None]:
        extrinsics = None
        intrinsics = None
        camera_geometry = None
        if isinstance(inputs, dict):
            if "images" not in inputs:
                raise ValueError("DA3 multi-view input dict requires an 'images' key")
            images = inputs["images"]
            extrinsics = inputs.get("extrinsics")
            intrinsics = inputs.get("intrinsics")
            camera_geometry = inputs.get("camera_geometry")
        else:
            images = inputs
        if not isinstance(images, (list, tuple)) or len(images) < 1:
            raise ValueError("DA3 multi-view preprocessing expects a non-empty list of still images")
        images_tuple = tuple(images)
        view_count = len(images_tuple)
        if camera_geometry is not None:
            if extrinsics is not None or intrinsics is not None:
                raise ValueError(
                    "Use either camera_geometry or extrinsics/intrinsics, not both"
                )
            if not isinstance(camera_geometry, CameraGeometry):
                raise TypeError("camera_geometry must be a CameraGeometry")
            if camera_geometry.view_count != view_count:
                raise ValueError(
                    f"camera_geometry has {camera_geometry.view_count} views, expected {view_count}"
                )
            return images_tuple, camera_geometry
        if extrinsics is None and intrinsics is None:
            return images_tuple, None
        return images_tuple, CameraGeometry(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            view_count=view_count,
        )

    def postprocess(self, raw: Any, ctx: SpatialTransform | DA3MultiViewContext) -> Result:
        data = raw.data if hasattr(raw, "data") else raw
        if isinstance(ctx, DA3MultiViewContext):
            return self._postprocess_multiview(data, ctx)

        depth = _single_dense_map(data["depth"], "depth")
        depth_conf = _optional_value(data, "depth_conf", "confidence")
        if depth_conf is not None:
            depth_conf = _single_dense_map(depth_conf, "depth_conf")
            depth_conf = ctx.invert_dense(depth_conf, mode="bilinear")
        depth_orig = ctx.invert_depth(depth)
        return Result(
            image_size=ctx.orig_size,
            depth=DepthMap(depth=depth_orig, depth_conf=depth_conf),
        )

    def _postprocess_multiview(self, data: Any, ctx: DA3MultiViewContext) -> Result:
        view_count = ctx.view_count
        depth = np.asarray(data["depth"])
        depth = _multiview_dense_maps(depth, view_count, "depth")
        depth_conf = _optional_value(data, "depth_conf", "confidence")
        if depth_conf is not None:
            depth_conf = _multiview_dense_maps(depth_conf, view_count, "depth_conf")

        depth_views = []
        for i, view_ctx in enumerate(ctx.views):
            view_conf = None
            if depth_conf is not None:
                view_conf = view_ctx.invert_dense(depth_conf[i], mode="bilinear")
            depth_views.append(
                DepthMap(depth=view_ctx.invert_depth(depth[i]), depth_conf=view_conf)
            )

        camera_geometry = _camera_geometry_from_output(data, view_count) or ctx.camera_geometry
        return Result(
            image_size=ctx.orig_size,
            depth=depth_views[0],
            depth_views=depth_views,
            camera_geometry=camera_geometry,
        )


def _is_multiview_input(inputs: Any) -> bool:
    if isinstance(inputs, dict):
        return "images" in inputs
    return isinstance(inputs, (list, tuple))


def _optional_value(data: Any, *keys: str) -> Any:
    if not hasattr(data, "get"):
        return None
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _single_dense_map(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[:2] == (1, 1):
        arr = arr[0, 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(
            f"DA3 monocular {name} must have shape (H, W), (1, H, W), "
            f"or (1, 1, H, W); got {arr.shape}"
        )
    return arr


def _multiview_dense_maps(value: Any, view_count: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 5:
        if arr.shape[0] != 1 or arr.shape[1] != view_count or arr.shape[2] != 1:
            raise ValueError(
                f"DA3 multi-view {name} must have shape (1, V, 1, H, W) "
                f"with V={view_count}; got {arr.shape}"
            )
        arr = arr[0, :, 0]
    elif arr.ndim == 4:
        if arr.shape[0] == 1 and arr.shape[1] == view_count:
            arr = arr[0]
        elif arr.shape[0] == view_count and arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            raise ValueError(
                f"DA3 multi-view {name} must have shape (1, V, H, W) "
                f"or (V, 1, H, W) with V={view_count}; got {arr.shape}"
            )
    elif arr.ndim == 3:
        if arr.shape[0] != view_count:
            raise ValueError(
                f"DA3 multi-view {name} has {arr.shape[0]} views, expected {view_count}; "
                "expected shape (V, H, W), (1, V, H, W), (V, 1, H, W), "
                "or (1, V, 1, H, W)"
            )
    else:
        raise ValueError(
            f"DA3 multi-view {name} must have a view axis; got shape {arr.shape}"
        )
    return arr


def _camera_geometry_from_output(data: Any, view_count: int) -> CameraGeometry | None:
    extrinsics = _optional_value(data, "extrinsics", "extrinsic")
    intrinsics = _optional_value(data, "intrinsics", "intrinsic")
    if extrinsics is None and intrinsics is None:
        return None
    if extrinsics is not None:
        extrinsics = _multiview_camera_array(extrinsics, view_count, "extrinsics")
    if intrinsics is not None:
        intrinsics = _multiview_camera_array(intrinsics, view_count, "intrinsics")
    return CameraGeometry(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        view_count=view_count,
        convention="w2c",
    )


def _multiview_camera_array(value: Any, view_count: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == view_count:
        return arr[0]
    if arr.ndim == 3 and arr.shape[0] == view_count:
        return arr
    got_views = None
    if arr.ndim == 4 and arr.shape[0] == 1:
        got_views = arr.shape[1]
    elif arr.ndim == 3:
        got_views = arr.shape[0]
    if got_views is not None:
        raise ValueError(f"DA3 multi-view {name} has {got_views} views, expected {view_count}")
    raise ValueError(
        f"DA3 multi-view {name} must have shape (V, ..., ...) or (1, V, ..., ...) "
        f"with V={view_count}; got {arr.shape}"
    )
