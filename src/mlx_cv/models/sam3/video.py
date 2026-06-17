"""SAM 3.1 video frame processing and session state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ...core.geometry import SpatialTransform
from ...core.image import load_image
from ...core.types import VideoResult
from ...prompts import BoxPrompt, ExemplarPrompt, PointPrompt, TextPrompt
from ...transforms.resize import Resize

__all__ = [
    "SAM3VideoProcessorConfig",
    "SAM3VideoFrameContext",
    "SAM3VideoProcessorContext",
    "SAM3VideoProcessor",
    "SAM3VideoPrompt",
    "SAM3VideoSessionState",
    "SAM3VideoSessionManager",
]


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _as_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    return (int(size[0]), int(size[1]))


@dataclass(frozen=True)
class SAM3VideoProcessorConfig:
    image_size: int | tuple[int, int] = 1024
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if min(_as_hw(self.image_size)) <= 0:
            raise ValueError("SAM3 video image_size must be positive")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("SAM3 video mean/std must each contain 3 channels")
        if any(s == 0 for s in self.std):
            raise ValueError("SAM3 video std values must be non-zero")

    @property
    def model_size(self) -> tuple[int, int]:
        return _as_hw(self.image_size)


@dataclass(frozen=True)
class SAM3VideoFrameContext:
    frame_index: int
    image_size: tuple[int, int]
    model_size: tuple[int, int]
    transform: SpatialTransform
    source: str | None = None


@dataclass(frozen=True)
class SAM3VideoProcessorContext:
    frames: tuple[SAM3VideoFrameContext, ...]
    model_size: tuple[int, int]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def image_sizes(self) -> tuple[tuple[int, int], ...]:
        return tuple(frame.image_size for frame in self.frames)


def _frame_sources(inputs: Any) -> list[Any]:
    if isinstance(inputs, dict):
        if "frames" in inputs:
            frames = inputs["frames"]
        elif "resource_path" in inputs:
            frames = inputs["resource_path"]
        elif "video" in inputs:
            frames = inputs["video"]
        else:
            raise ValueError("SAM3VideoProcessor.preprocess requires frames, resource_path, or video")
    else:
        frames = inputs

    if isinstance(frames, (str, Path)) or hasattr(frames, "__fspath__"):
        path = Path(frames)
        if path.is_dir():
            out = [p for p in sorted(path.iterdir()) if p.suffix.lower() in _IMAGE_SUFFIXES]
            if not out:
                raise ValueError(f"SAM3 video frame directory contains no supported image files: {path}")
            return out
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            return [path]
        raise ValueError(
            "SAM3 video runtime accepts frame directories or image files; video-file decoding is optional"
        )

    if isinstance(frames, np.ndarray):
        if frames.ndim == 4:
            return [frames[i] for i in range(frames.shape[0])]
        return [frames]

    try:
        out = list(frames)
    except TypeError as exc:
        raise TypeError(f"unsupported SAM3 video input: {type(frames).__name__}") from exc
    if not out:
        raise ValueError("SAM3 video requires at least one frame")
    return out


class SAM3VideoProcessor:
    """Frame-sequence processor for SAM3 video sessions."""

    def __init__(self, config: SAM3VideoProcessorConfig | None = None) -> None:
        self.config = config or SAM3VideoProcessorConfig()

    def preprocess(self, inputs: Any) -> tuple[dict[str, Any], SAM3VideoProcessorContext]:
        sources = _frame_sources(inputs)
        resize = Resize(self.config.model_size)
        mean = np.asarray(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.config.std, dtype=np.float32).reshape(1, 1, 3)
        tensors = []
        contexts = []
        for frame_index, source in enumerate(sources):
            arr, image_size = load_image(source)
            resized, transform = resize(arr)
            x = resized.astype(np.float32) / 255.0
            x = (x - mean) / std
            tensors.append(np.transpose(x, (2, 0, 1)))
            contexts.append(
                SAM3VideoFrameContext(
                    frame_index=frame_index,
                    image_size=image_size,
                    model_size=self.config.model_size,
                    transform=transform,
                    source=str(source) if isinstance(source, (str, Path)) or hasattr(source, "__fspath__") else None,
                )
            )
        pixel_values = np.ascontiguousarray(np.stack(tensors, axis=0))
        context = SAM3VideoProcessorContext(tuple(contexts), self.config.model_size)
        return {"pixel_values": pixel_values}, context


@dataclass(frozen=True)
class SAM3VideoPrompt:
    mode: str
    frame_index: int
    prompt: Any
    texts: tuple[str, ...] = ()
    object_id: int | None = None
    label: str | None = None


@dataclass
class SAM3VideoSessionState:
    session_id: str
    context: SAM3VideoProcessorContext
    prompts: list[SAM3VideoPrompt] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _prompt_from_kwargs(prompt: Any = None, **kwargs) -> Any:
    if prompt is not None:
        return prompt
    if "text" in kwargs and kwargs["text"] is not None:
        return kwargs["text"]
    if "box" in kwargs or "boxes" in kwargs:
        return {"boxes": kwargs.get("boxes", kwargs.get("box"))}
    if "exemplar" in kwargs:
        return kwargs["exemplar"]
    if "points" in kwargs or "point" in kwargs:
        return {"points": kwargs.get("points", kwargs.get("point"))}
    if "mask" in kwargs or "mask_prompt" in kwargs:
        return {"mask": kwargs.get("mask", kwargs.get("mask_prompt"))}
    return None


def _classify_prompt(prompt: Any, frame_index: int, object_id: int | None, label: str | None) -> SAM3VideoPrompt:
    if prompt is None:
        raise ValueError("SAM3 video add_prompt requires a prompt")
    if isinstance(prompt, str):
        return SAM3VideoPrompt("sam3_video", frame_index, prompt, texts=(prompt,), object_id=object_id, label=label)
    if isinstance(prompt, TextPrompt):
        return SAM3VideoPrompt("sam3_video", frame_index, prompt, texts=(prompt.text,), object_id=object_id, label=label)
    if isinstance(prompt, PointPrompt):
        raise NotImplementedError("SAM3 video point prompts are deferred until tracker propagation support")
    if isinstance(prompt, (BoxPrompt, ExemplarPrompt)):
        return SAM3VideoPrompt("sam3_tracker", frame_index, prompt, object_id=object_id, label=label)
    if isinstance(prompt, dict):
        unsupported = {"point", "points", "mask", "masks", "mask_prompt", "mask_prompts"} & set(prompt)
        if unsupported:
            key = sorted(unsupported)[0]
            raise NotImplementedError(f"SAM3 video prompt state does not support {key!r} yet")
        text = prompt.get("text", prompt.get("prompt"))
        if text is not None:
            texts = (text,) if isinstance(text, str) else tuple(str(t) for t in text)
            return SAM3VideoPrompt("sam3_video", frame_index, prompt, texts=texts, object_id=object_id, label=label)
        if "box" in prompt or "boxes" in prompt or "exemplar" in prompt or "exemplar_image" in prompt:
            return SAM3VideoPrompt("sam3_tracker", frame_index, prompt, object_id=object_id, label=label)
    raise TypeError(f"unsupported SAM3 video prompt type: {type(prompt).__name__}")


class SAM3VideoSessionManager:
    """Small local session manager mirroring the upstream SAM3 video request names."""

    def __init__(self, processor: SAM3VideoProcessor | None = None) -> None:
        self.processor = processor or SAM3VideoProcessor()
        self.sessions: dict[str, SAM3VideoSessionState] = {}

    def start_session(
        self,
        *,
        frames: Any = None,
        resource_path: Any = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SAM3VideoSessionState:
        source = frames if frames is not None else resource_path
        if source is None:
            raise ValueError("SAM3 video start_session requires frames or resource_path")
        _, context = self.processor.preprocess(source)
        sid = session_id or uuid4().hex
        if sid in self.sessions:
            raise ValueError(f"SAM3 video session already exists: {sid}")
        state = SAM3VideoSessionState(sid, context, metadata=metadata or {})
        self.sessions[sid] = state
        return state

    def add_prompt(
        self,
        session_id: str,
        *,
        frame_index: int = 0,
        prompt: Any = None,
        object_id: int | None = None,
        label: str | None = None,
        **kwargs,
    ) -> SAM3VideoPrompt:
        state = self._session(session_id)
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= state.context.frame_count:
            raise ValueError(f"SAM3 video frame_index {frame_index} is outside the session")
        prepared = _classify_prompt(_prompt_from_kwargs(prompt, **kwargs), frame_index, object_id, label)
        state.prompts.append(prepared)
        return prepared

    def propagate_in_video(self, session_id: str, **kwargs) -> VideoResult:
        self._session(session_id)
        raise NotImplementedError("SAM3 video propagation requires tracker memory; this is implemented in Slice 4")

    def handle_request(self, request: dict[str, Any]) -> Any:
        request_type = request.get("type")
        if request_type == "start_session":
            return self.start_session(
                frames=request.get("frames"),
                resource_path=request.get("resource_path"),
                session_id=request.get("session_id"),
                metadata=request.get("metadata"),
            )
        if request_type == "add_prompt":
            return self.add_prompt(
                request["session_id"],
                frame_index=request.get("frame_index", 0),
                prompt=request.get("prompt"),
                text=request.get("text"),
                box=request.get("box"),
                boxes=request.get("boxes"),
                exemplar=request.get("exemplar"),
                point=request.get("point"),
                points=request.get("points"),
                mask=request.get("mask"),
                mask_prompt=request.get("mask_prompt"),
                object_id=request.get("object_id"),
                label=request.get("label"),
            )
        if request_type == "propagate_in_video":
            return self.propagate_in_video(request["session_id"])
        raise ValueError(f"unsupported SAM3 video request type: {request_type!r}")

    def _session(self, session_id: str) -> SAM3VideoSessionState:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown SAM3 video session: {session_id}") from exc
