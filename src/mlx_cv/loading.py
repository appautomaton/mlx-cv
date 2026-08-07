"""Lazy public model loading without importing MLX at package import time."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from os import PathLike
from typing import Any

from .core.base import Task
from .core.registry import Registry

__all__ = [
    "MODEL_LOADERS",
    "ModelLoaderSpec",
    "available_models",
    "load",
    "register_model_loader",
]


@dataclass(frozen=True)
class ModelLoaderSpec:
    """One user-facing runtime and its lazily imported ``from_pretrained`` class."""

    name: str
    task: Task
    loader: str
    aliases: tuple[str, ...] = ()
    default_pretrained: str | None = None

    def resolve_loader(self):
        module_name, separator, attribute = self.loader.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError(
                f"model loader {self.name!r} must use a 'module:attribute' target"
            )
        module = importlib.import_module(module_name)
        loader = getattr(module, attribute)
        if not hasattr(loader, "from_pretrained"):
            raise TypeError(
                f"model loader {self.loader!r} does not expose from_pretrained()"
            )
        return loader


MODEL_LOADERS: Registry[ModelLoaderSpec] = Registry("model_loaders")
_MODEL_ALIASES: dict[str, str] = {}


def register_model_loader(spec: ModelLoaderSpec) -> ModelLoaderSpec:
    """Register one canonical public loader and its unambiguous aliases."""

    keys = (spec.name, *spec.aliases)
    normalized = tuple(key.strip().lower() for key in keys)
    if any(not key for key in normalized):
        raise ValueError("model loader names and aliases must be non-empty")
    conflicts = sorted(key for key in normalized if key in _MODEL_ALIASES)
    if conflicts:
        raise KeyError(f"model loader aliases already registered: {conflicts}")
    MODEL_LOADERS.register(spec.name, spec)
    _MODEL_ALIASES.update({key: spec.name for key in normalized})
    return spec


def available_models() -> tuple[str, ...]:
    """Return canonical model keys accepted by :func:`load`."""

    return tuple(MODEL_LOADERS.list())


def load(
    model: str,
    pretrained_model_name_or_path: str | PathLike[str] | None = None,
    **kwargs: Any,
):
    """Load a supported inference runtime from a local package or remote snapshot.

    ``model`` selects a canonical runtime such as ``"rfdetr-nano"`` or
    ``"sam3.1-video"``. The second argument is passed to that runtime's
    ``from_pretrained`` constructor. EoMT-DINOv3 has an official configured
    default; all other runtimes require an explicit package path or exact remote
    identifier.
    """

    if not isinstance(model, str) or not model.strip():
        raise TypeError("model must be a non-empty string")
    lookup = model.strip().lower()
    try:
        canonical = _MODEL_ALIASES[lookup]
    except KeyError:
        choices = ", ".join(available_models())
        raise KeyError(f"unknown model {model!r}; available models: {choices}") from None
    spec = MODEL_LOADERS.get(canonical)
    pretrained = pretrained_model_name_or_path or spec.default_pretrained
    if pretrained is None:
        raise ValueError(
            f"{spec.name!r} requires a local package path or exact remote identifier"
        )
    return spec.resolve_loader().from_pretrained(pretrained, **kwargs)


for _spec in (
    ModelLoaderSpec(
        name="eomt-dinov3-coco-panoptic-small-640",
        task=Task.SEGMENTATION,
        loader="mlx_cv.models.eomt_dinov3:EoMTDINOv3",
        aliases=("eomt-dinov3", "eomt-dinov3-small", "eomt-small"),
        default_pretrained="tue-mps/eomt-dinov3-coco-panoptic-small-640",
    ),
    ModelLoaderSpec(
        name="locateanything-3b",
        task=Task.GROUNDING,
        loader="mlx_cv.models.locateanything:LocateAnythingPipeline",
        aliases=("locateanything", "locate-anything-3b"),
    ),
    ModelLoaderSpec(
        name="rfdetr-nano",
        task=Task.DETECTION,
        loader="mlx_cv.models.rfdetr:RFDETRModel",
        aliases=("rf-detr-nano", "rfdetr"),
    ),
    ModelLoaderSpec(
        name="depth-anything-v3-monocular",
        task=Task.DEPTH,
        loader="mlx_cv.models.depth_anything_v3:DepthAnythingV3Monocular",
        aliases=("da3-monocular",),
    ),
    ModelLoaderSpec(
        name="depth-anything-v3-multiview",
        task=Task.DEPTH,
        loader="mlx_cv.models.depth_anything_v3:DepthAnythingV3MultiView",
        aliases=("depth-anything-v3-small", "da3-small", "da3-multiview"),
    ),
    ModelLoaderSpec(
        name="sam3.1-image",
        task=Task.SEGMENTATION,
        loader="mlx_cv.models.sam3:SAM3Processor",
        aliases=("sam3.1", "sam3-image"),
    ),
    ModelLoaderSpec(
        name="sam3.1-video",
        task=Task.TRACKING,
        loader="mlx_cv.models.sam3:SAM3VideoSession",
        aliases=("sam3-video", "sam3.1-multiplex"),
    ),
):
    register_model_loader(_spec)

del _spec
