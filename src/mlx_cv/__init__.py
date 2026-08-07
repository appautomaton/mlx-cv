"""MLX-native computer vision building blocks for Apple Silicon.

The import-light package root exposes NumPy-backed result types, spatial geometry,
registries, and model contracts. MLX-backed grounding, detection, depth,
segmentation, and tracking implementations live in their model subpackages and
load weights from external checkpoints or local model packages.

See ``docs/ARCHITECTURE.md`` for the implemented boundaries and current gaps.
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
from .loading import (
    MODEL_LOADERS,
    ModelLoaderSpec,
    available_models,
    load,
    register_model_loader,
)

__version__ = "0.0.4"

__all__ = [
    "__version__",
    # output types
    "Result", "Detections", "Masks", "Keypoints", "Points", "DepthMap",
    "CameraGeometry", "Embedding", "Tracks", "VideoResult",
    "TrackMemoryRecord", "MultiplexBucket", "ObjectMultiplexState",
    # public model loading
    "MODEL_LOADERS", "ModelLoaderSpec", "available_models", "load",
    "register_model_loader",
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
