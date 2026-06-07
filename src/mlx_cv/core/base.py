"""The spine's abstract contracts — shape, not implementation (§5.4, §5.5).

Every model plugs into these. No model code lives here; concrete models implement
``Processor`` / ``Predictor`` and register a ``Module`` + backbones/heads.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from .geometry import SpatialTransform
from .types import Result

__all__ = [
    "Task", "Module", "VisionBackbone", "LanguageBackbone", "Head",
    "Processor", "Predictor", "Tracker",
]


class Task(enum.Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    DEPTH = "depth"
    POSE = "pose"
    GROUNDING = "grounding"
    TRACKING = "tracking"
    EMBEDDING = "embedding"


@runtime_checkable
class Module(Protocol):
    """A pure compute graph (an ``mlx.nn.Module`` in practice). No I/O."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class VisionBackbone(Protocol):
    """``image -> list[FeatureMap]``. Registered with ``kind="vision"``."""

    def __call__(self, x: Any) -> list: ...


@runtime_checkable
class LanguageBackbone(Protocol):
    """``embeds -> hidden states`` (+ a decode loop). Registered with ``kind="llm"``."""

    def embed(self, token_ids: Any) -> Any: ...

    def __call__(self, embeds: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class Head(Protocol):
    """A task decoder over backbone features: ``features -> raw outputs``."""

    def __call__(self, feats: Any) -> Any: ...


class Processor(ABC):
    """Owns pre/post-processing (and prompt encoding). Testable in isolation."""

    @abstractmethod
    def preprocess(self, inputs: Any) -> tuple[Any, SpatialTransform]:
        """Return ``(model_input, ctx)``; ``ctx`` records the spatial mapping."""

    @abstractmethod
    def postprocess(self, raw: Any, ctx: SpatialTransform) -> Result:
        """Map raw model outputs back to original-image coords via ``ctx``."""


class Predictor(ABC):
    """Wires ``Processor -> Module -> Processor`` into ``predict()``. User-facing."""

    task: Task

    @abstractmethod
    def predict(self, inputs: Any, *, prompt: Any = None, **opts: Any) -> Result: ...


class Tracker(ABC):
    """Stateful temporal protocol for video / VOS (§5.5)."""

    @abstractmethod
    def init(self, frame: Any, prompt: Any) -> Result: ...

    @abstractmethod
    def step(self, frame: Any) -> Result:
        """Advance one frame, carrying memory."""
