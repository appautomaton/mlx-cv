"""mlx-cv: MLX-native computer vision for Apple Silicon.

This release (v0.0.3) pairs the task-agnostic spine — the unified ``Result`` type,
the invertible ``SpatialTransform`` coordinate context, the model/backbone/head
registries, pure box/coord ops, the prompt taxonomy, and the parity harness — with
the first models that plug into it: Depth Anything V3 (depth), RF-DETR (detection),
LocateAnything (grounding), and SAM3 (segmentation/tracking). The design is in
``docs/ARCHITECTURE.md`` (see §16 for the LocateAnything anchor plan).

The top-level package is numpy-backed and import-light; the MLX runtime the models
compute on is the optional ``mlx-cv[mlx]`` extra, and model weights are loaded from
externally supplied checkpoints rather than bundled with the package.
"""

from __future__ import annotations

from .core.base import (
    Head,
    LanguageBackbone,
    Module,
    Predictor,
    Processor,
    Task,
    Tracker,
    VisionBackbone,
)
from .core.features import (
    BackboneFeatures,
    FeatureMap,
    HeadInput,
    HeadOutput,
    Layout,
    TokenLayout,
)
from .core.geometry import SpatialTransform
from .core.image import load_image
from .core.registry import (
    BACKBONES,
    HEADS,
    MODELS,
    Registry,
    load_plugins,
    register_backbone,
    register_head,
    register_model,
)
from .core.types import (
    CameraGeometry,
    Detections,
    DepthMap,
    Embedding,
    Keypoints,
    Masks,
    Points,
    Result,
    Tracks,
    VideoResult,
)
from .core.tracking import (
    MultiplexBucket,
    ObjectMultiplexState,
    TrackMemoryRecord,
)

__version__ = "0.0.3"

__all__ = [
    "__version__",
    # output types
    "Result", "Detections", "Masks", "Keypoints", "Points", "DepthMap",
    "CameraGeometry", "Embedding", "Tracks", "VideoResult",
    "TrackMemoryRecord", "MultiplexBucket", "ObjectMultiplexState",
    # coordinate discipline + image I/O
    "SpatialTransform", "load_image",
    # registries
    "Registry", "MODELS", "BACKBONES", "HEADS",
    "register_model", "register_backbone", "register_head", "load_plugins",
    # contracts
    "Task", "Module", "VisionBackbone", "LanguageBackbone", "Head",
    "Processor", "Predictor", "Tracker",
    # feature + head I/O contracts
    "Layout", "TokenLayout", "FeatureMap", "BackboneFeatures", "HeadInput", "HeadOutput",
]
