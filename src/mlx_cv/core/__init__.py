"""Core spine: types, geometry, registry, image I/O, abstract contracts."""

from __future__ import annotations

from .base import (
    Head,
    LanguageBackbone,
    Module,
    Predictor,
    Processor,
    Task,
    Tracker,
    VisionBackbone,
)
from .features import (
    BackboneFeatures,
    FeatureMap,
    HeadInput,
    HeadOutput,
    Layout,
    TokenLayout,
)
from .geometry import SpatialTransform
from .image import load_image
from .registry import (
    BACKBONES,
    HEADS,
    MODELS,
    Registry,
    load_plugins,
    register_backbone,
    register_head,
    register_model,
)
from .types import (
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
from .tracking import (
    MultiplexBucket,
    ObjectMultiplexState,
    TrackMemoryRecord,
)

__all__ = [
    "Result", "Detections", "Masks", "Keypoints", "Points", "DepthMap",
    "CameraGeometry", "Embedding", "Tracks", "VideoResult",
    "TrackMemoryRecord", "MultiplexBucket", "ObjectMultiplexState",
    "SpatialTransform", "load_image",
    "Registry", "MODELS", "BACKBONES", "HEADS",
    "register_model", "register_backbone", "register_head", "load_plugins",
    "Task", "Module", "VisionBackbone", "LanguageBackbone", "Head",
    "Processor", "Predictor", "Tracker",
    "Layout", "TokenLayout", "FeatureMap", "BackboneFeatures", "HeadInput", "HeadOutput",
]
