"""SAM 3.1 video frame processing and session state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlx.core as mx
import numpy as np

from ...core.base import Task, Tracker
from ...core.geometry import SpatialTransform
from ...core.image import load_image
from ...core.tracking import ObjectMultiplexState, TrackMemoryRecord
from ...core.types import Detections, Masks, Result, Tracks, VideoResult
from ...prompts import BoxPrompt, ExemplarPrompt, PointPrompt, TextPrompt
from ...transforms.resize import Resize
from .config import SAM3MultiplexDecoderConfig, SAM3VideoConfig, SAM3VideoMemoryConfig, SAM3VideoTrackerConfig
from .multiplex_state import SAM3MultiplexController, SAM3MultiplexState
from .video_model import SAM3VideoModel

__all__ = [
    "SAM3VideoProcessorConfig",
    "SAM3VideoFrameContext",
    "SAM3VideoProcessorContext",
    "SAM3VideoProcessor",
    "SAM3VideoPrompt",
    "SAM3VideoSessionState",
    "SAM3VideoSessionManager",
    "SAM3VideoTracker",
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
    multiplex_state: ObjectMultiplexState
    prompts: list[SAM3VideoPrompt] = field(default_factory=list)
    memory: list[TrackMemoryRecord] = field(default_factory=list)
    pixel_values: np.ndarray | None = None
    neural_output_dict: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _prompt_from_kwargs(prompt: Any = None, **kwargs) -> Any:
    if prompt is not None:
        return prompt
    if "text" in kwargs and kwargs["text"] is not None:
        return kwargs["text"]
    boxes = kwargs.get("boxes")
    if boxes is None:
        boxes = kwargs.get("box")
    if boxes is not None:
        return {"boxes": boxes}
    if kwargs.get("exemplar") is not None:
        return kwargs["exemplar"]
    points = kwargs.get("points")
    if points is None:
        points = kwargs.get("point")
    if points is not None:
        return {"points": points}
    mask = kwargs.get("mask")
    if mask is None:
        mask = kwargs.get("mask_prompt")
    if mask is not None:
        return {"mask": mask}
    return None


def _classify_prompt(prompt: Any, frame_index: int, object_id: int | None, label: str | None) -> SAM3VideoPrompt:
    if prompt is None:
        raise ValueError("SAM3 video add_prompt requires a prompt")
    if isinstance(prompt, str):
        raise NotImplementedError("SAM3 video text prompts require the detector/text path, which is not ported yet")
    if isinstance(prompt, TextPrompt):
        raise NotImplementedError("SAM3 video text prompts require the detector/text path, which is not ported yet")
    if isinstance(prompt, PointPrompt):
        raise NotImplementedError("SAM3 video point prompts are deferred until tracker propagation support")
    if isinstance(prompt, ExemplarPrompt):
        raise NotImplementedError("SAM3 video exemplar prompts require the exemplar path, which is not ported yet")
    if isinstance(prompt, BoxPrompt):
        return SAM3VideoPrompt("sam3_tracker", frame_index, prompt, object_id=object_id, label=label)
    if isinstance(prompt, dict):
        unsupported = {
            "point",
            "points",
            "mask",
            "masks",
            "mask_prompt",
            "mask_prompts",
            "exemplar",
            "exemplar_image",
        } & set(prompt)
        if unsupported:
            key = sorted(unsupported)[0]
            raise NotImplementedError(f"SAM3 video prompt state does not support {key!r} yet")
        text = prompt.get("text", prompt.get("prompt"))
        if text is not None:
            raise NotImplementedError("SAM3 video text prompts require the detector/text path, which is not ported yet")
        if "box" in prompt or "boxes" in prompt:
            return SAM3VideoPrompt("sam3_tracker", frame_index, prompt, object_id=object_id, label=label)
    raise TypeError(f"unsupported SAM3 video prompt type: {type(prompt).__name__}")


def _default_video_config(multiplex_count: int) -> SAM3VideoConfig:
    multiplex_count = max(1, int(multiplex_count))
    tracker = SAM3VideoTrackerConfig(
        hidden_dim=16,
        image_size=(32, 32),
        feature_grid=(2, 2),
        multiplex_count=multiplex_count,
        num_maskmem=3,
        max_obj_ptrs_in_encoder=4,
        condition_as_mask_input=True,
        save_image_features=True,
    )
    memory = SAM3VideoMemoryConfig(
        hidden_dim=tracker.hidden_dim,
        image_size=tracker.image_size,
        feature_grid=tracker.feature_grid,
        multiplex_count=tracker.multiplex_count,
        condition_as_mask_input=tracker.condition_as_mask_input,
    )
    decoder = SAM3MultiplexDecoderConfig(
        hidden_dim=tracker.hidden_dim,
        multiplex_count=tracker.multiplex_count,
        low_res_mask_size=(8, 8),
        high_res_mask_size=tracker.image_size,
        pred_obj_scores=True,
    )
    return SAM3VideoConfig(tracker=tracker, memory=memory, decoder=decoder)


class SAM3VideoSessionManager:
    """Small local session manager mirroring the upstream SAM3 video request names."""

    def __init__(
        self,
        processor: SAM3VideoProcessor | None = None,
        *,
        multiplex_bucket_capacity: int = 16,
        model: SAM3VideoModel | None = None,
        video_config: SAM3VideoConfig | None = None,
    ) -> None:
        self.multiplex_bucket_capacity = int(multiplex_bucket_capacity)
        if self.multiplex_bucket_capacity < 1:
            raise ValueError("SAM3 video multiplex_bucket_capacity must be positive")
        if model is not None and video_config is not None:
            raise ValueError("Specify either SAM3 video model or video_config, not both")
        self.model = model or SAM3VideoModel(video_config or _default_video_config(self.multiplex_bucket_capacity))
        if self.multiplex_bucket_capacity > self.model.cfg.tracker.multiplex_count:
            raise ValueError(
                "SAM3 video multiplex_bucket_capacity cannot exceed model multiplex_count "
                f"({self.model.cfg.tracker.multiplex_count})"
            )
        self.processor = processor or SAM3VideoProcessor(
            SAM3VideoProcessorConfig(image_size=self.model.cfg.tracker.image_size)
        )
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
        processed, context = self.processor.preprocess(source)
        sid = session_id or uuid4().hex
        if sid in self.sessions:
            raise ValueError(f"SAM3 video session already exists: {sid}")
        state = SAM3VideoSessionState(
            sid,
            context,
            ObjectMultiplexState(bucket_capacity=self.multiplex_bucket_capacity),
            pixel_values=processed["pixel_values"],
            metadata=metadata or {},
        )
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
        if object_id is None:
            object_id = self._next_object_id(state)
        prepared = _classify_prompt(_prompt_from_kwargs(prompt, **kwargs), frame_index, object_id, label)
        state.multiplex_state.assign_object(object_id)
        state.prompts.append(prepared)
        return prepared

    def remove_object(self, session_id: str, object_id: int) -> None:
        state = self._session(session_id)
        object_id = int(object_id)
        state.prompts = [prompt for prompt in state.prompts if prompt.object_id != object_id]
        state.memory = [record for record in state.memory if record.object_id != object_id]
        state.multiplex_state.remove_object(object_id)

    def propagate_in_video(
        self,
        session_id: str,
        *,
        start_frame_index: int = 0,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> VideoResult:
        state = self._session(session_id)
        start_frame_index = int(start_frame_index)
        if reverse:
            start = min(start_frame_index, state.context.frame_count - 1)
            indices = [] if start < 0 else list(range(start, -1, -1))
        else:
            start = max(0, start_frame_index)
            indices = list(range(start, state.context.frame_count))
        if max_frame_num_to_track is not None:
            indices = indices[: max(0, int(max_frame_num_to_track))]

        frame_results: list[Result] = []
        state.memory.clear()
        state.multiplex_state.memory.clear()
        state.neural_output_dict.clear()
        for frame_index in indices:
            frame_ctx = state.context.frames[frame_index]
            pixel_value = None if state.pixel_values is None else state.pixel_values[frame_index]
            result = self._propagate_frame(state, frame_ctx, pixel_value=pixel_value)
            frame_results.append(result)
        return VideoResult(
            frame_results,
            frame_indices=[state.context.frames[idx].frame_index for idx in indices],
            session_id=session_id,
            metadata={
                "claim_level": "mlx_neural_forward",
                "tracker": "mlx_neural",
                "multiplex": state.multiplex_state.to_dict(),
            },
        )

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
            return self.propagate_in_video(
                request["session_id"],
                start_frame_index=request.get("start_frame_index", 0),
                max_frame_num_to_track=request.get("max_frame_num_to_track"),
                reverse=request.get("reverse", False),
            )
        if request_type == "remove_object":
            return self.remove_object(request["session_id"], request["object_id"])
        raise ValueError(f"unsupported SAM3 video request type: {request_type!r}")

    def _session(self, session_id: str) -> SAM3VideoSessionState:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown SAM3 video session: {session_id}") from exc

    def _next_object_id(self, state: SAM3VideoSessionState) -> int:
        used = [prompt.object_id for prompt in state.prompts if prompt.object_id is not None]
        return 1 if not used else max(used) + 1

    def _propagate_frame(
        self,
        state: SAM3VideoSessionState,
        frame_ctx: SAM3VideoFrameContext,
        *,
        pixel_value: np.ndarray | None = None,
    ) -> Result:
        h, w = frame_ctx.image_size
        if not state.prompts:
            return Result(image_size=frame_ctx.image_size)

        object_ids = tuple(state.multiplex_state.active_object_ids)
        model_multiplex = self._model_multiplex_state(state)
        if pixel_value is None and state.pixel_values is not None:
            frame_index = frame_ctx.frame_index
            if 0 <= frame_index < int(state.pixel_values.shape[0]):
                pixel_value = state.pixel_values[frame_index]
        image_features, image_pos_enc, high_res_features = _frame_neural_features(
            pixel_value,
            self.model.cfg,
            model_multiplex.num_buckets,
            frame_ctx.frame_index,
        )
        conditioning_prompts = [prompt for prompt in state.prompts if prompt.frame_index == frame_ctx.frame_index]
        is_init_cond_frame = bool(conditioning_prompts)
        mask_inputs = None
        if is_init_cond_frame:
            mask_inputs = _prompt_seed_masks(conditioning_prompts, object_ids, self.model.cfg.tracker.image_size)

        model_out = self.model.track_step(
            frame_index=frame_ctx.frame_index,
            image_features=image_features,
            image_pos_enc=image_pos_enc,
            high_res_features=high_res_features,
            mask_inputs=mask_inputs,
            is_init_cond_frame=is_init_cond_frame,
            output_dict=state.neural_output_dict,
            multiplex_state=model_multiplex,
        )

        masks_np = _model_masks_to_public(model_out.stage.high_res_masks, frame_ctx.image_size)
        score_logits = np.asarray(model_out.stage.object_score_logits).reshape(-1)
        scores_np = 1.0 / (1.0 + np.exp(-score_logits))

        boxes = []
        labels = []
        metadata = []
        for object_idx, object_id in enumerate(object_ids):
            prompt = _latest_prompt_for_object(state.prompts, object_id)
            label = prompt.label or (prompt.texts[0] if prompt.texts else f"object_{object_id}")
            bucket_index = state.multiplex_state.object_to_bucket[object_id]
            boxes.append(_mask_box(masks_np[object_idx]))
            labels.append(label)
            metadata.append({"prompt_mode": prompt.mode, "multiplex_bucket": bucket_index})
            score = float(scores_np[object_idx]) if object_idx < len(scores_np) else 1.0
            record = TrackMemoryRecord(
                object_id,
                frame_ctx.frame_index,
                mask_shape=tuple(masks_np[object_idx].shape),
                score=score,
                metadata={
                    "prompt_mode": prompt.mode,
                    "label": label,
                    "multiplex_bucket": bucket_index,
                    "tracker": "mlx_neural",
                },
            )
            state.memory.append(record)
            state.multiplex_state.add_memory(record)

        track_arr = np.asarray(object_ids, dtype=np.int64)
        return Result(
            image_size=(h, w),
            masks=Masks(masks_np, kind="instance", labels=labels),
            detections=Detections(
                np.asarray(boxes, dtype=np.float64),
                scores=np.asarray(scores_np[: len(object_ids)], dtype=np.float64),
                labels=labels,
                track_ids=track_arr,
            ),
            tracks=Tracks(
                track_arr,
                frame_index=frame_ctx.frame_index,
                scores=np.asarray(scores_np[: len(object_ids)], dtype=np.float64),
                labels=labels,
                metadata=metadata,
            ),
        )

    def _model_multiplex_state(self, state: SAM3VideoSessionState) -> SAM3MultiplexState:
        controller = SAM3MultiplexController(
            self.model.cfg.tracker.multiplex_count,
            eval_multiplex_count=state.multiplex_state.bucket_capacity,
        )
        return controller.get_state(
            len(state.multiplex_state.active_object_ids),
            object_ids=state.multiplex_state.active_object_ids,
        )


class SAM3VideoTracker(Tracker):
    """Streaming ``init``/``step`` adapter over :class:`SAM3VideoSessionManager`.

    Implements the spine temporal contract (``core.base.Tracker``, ARCHITECTURE §5.5;
    BUILDING-BLOCKS maps it to SAM video) for SAM 3.1 video. ``init(frame, prompt)``
    seeds a session from the first frame, registers the prompt, and returns that
    frame's :class:`~mlx_cv.core.types.Result`; each ``step(frame)`` advances one frame
    and carries tracker/multiplex memory forward. Unlike the batch
    :meth:`SAM3VideoSessionManager.propagate_in_video` (which resets memory and needs
    the whole clip up front), the tracker consumes frames one at a time for online use.
    """

    task = Task.TRACKING

    def __init__(
        self,
        *,
        manager: SAM3VideoSessionManager | None = None,
        processor: SAM3VideoProcessor | None = None,
        session_id: str | None = None,
    ) -> None:
        if manager is not None and processor is not None:
            raise ValueError("Specify either manager or processor, not both")
        self.manager = manager or SAM3VideoSessionManager(processor=processor)
        self._session_id = session_id
        self._state: SAM3VideoSessionState | None = None
        self._cursor = 0

    @property
    def session_id(self) -> str | None:
        """Underlying session id; populated once :meth:`init` runs."""
        return self._session_id

    def init(self, frame: Any, prompt: Any = None) -> Result:
        if self._state is not None:
            raise RuntimeError(
                "SAM3VideoTracker.init was already called; create a new tracker per sequence"
            )
        previous_session_id = self._session_id
        state = self.manager.start_session(frames=[frame], session_id=self._session_id)
        try:
            if prompt is not None:
                self.manager.add_prompt(state.session_id, frame_index=0, prompt=prompt)
            pixel_value = None if state.pixel_values is None else state.pixel_values[0]
            result = self.manager._propagate_frame(state, state.context.frames[0], pixel_value=pixel_value)
        except Exception:
            self.manager.sessions.pop(state.session_id, None)
            self._state = None
            self._session_id = previous_session_id
            raise
        self._state = state
        self._session_id = state.session_id
        self._cursor = 1
        return result

    def step(self, frame: Any) -> Result:
        if self._state is None:
            raise RuntimeError("SAM3VideoTracker.step called before init")
        processed, context = self.manager.processor.preprocess([frame])
        frame_ctx = replace(context.frames[0], frame_index=self._cursor)
        result = self.manager._propagate_frame(self._state, frame_ctx, pixel_value=processed["pixel_values"][0])
        self._cursor += 1
        return result


def _first_box(prompt: Any) -> np.ndarray | None:
    if isinstance(prompt, BoxPrompt):
        return prompt.boxes[0]
    if isinstance(prompt, dict):
        boxes = prompt.get("boxes", prompt.get("box"))
        if boxes is not None:
            return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[0]
    return None


def _latest_prompt_for_object(prompts: list[SAM3VideoPrompt], object_id: int) -> SAM3VideoPrompt:
    for prompt in reversed(prompts):
        if prompt.object_id == object_id:
            return prompt
    raise ValueError(f"SAM3 video object_id {object_id} has no active prompt")


def _prompt_seed_box(prompt: SAM3VideoPrompt, image_size: tuple[int, int]) -> np.ndarray:
    h, w = image_size
    box = _first_box(prompt.prompt)
    if box is None:
        object_id = int(prompt.object_id if prompt.object_id is not None else 0)
        span_w = max(1, w // 3)
        span_h = max(1, h // 3)
        x0 = (object_id * 3 + int(prompt.frame_index)) % max(1, w - span_w + 1)
        y0 = (object_id * 5 + int(prompt.frame_index)) % max(1, h - span_h + 1)
        box = np.asarray([x0, y0, x0 + span_w, y0 + span_h], dtype=np.float64)
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float64)
    return np.asarray(
        [
            np.clip(x0, 0, max(0, w - 1)),
            np.clip(y0, 0, max(0, h - 1)),
            np.clip(max(x1, x0 + 1), 1, w),
            np.clip(max(y1, y0 + 1), 1, h),
        ],
        dtype=np.float64,
    )


def _prompt_seed_mask(image_size: tuple[int, int], box: np.ndarray) -> np.ndarray:
    h, w = image_size
    x0, y0, x1, y1 = box
    ix0 = int(np.floor(np.clip(x0, 0, w)))
    iy0 = int(np.floor(np.clip(y0, 0, h)))
    ix1 = int(np.ceil(np.clip(x1, ix0 + 1, w)))
    iy1 = int(np.ceil(np.clip(y1, iy0 + 1, h)))
    mask = np.zeros((h, w), dtype=np.float32)
    mask[iy0:iy1, ix0:ix1] = True
    return mask


def _prompt_seed_masks(
    prompts: list[SAM3VideoPrompt],
    object_ids: tuple[int, ...],
    model_size: tuple[int, int],
) -> mx.array:
    masks = np.zeros((len(object_ids), 1, model_size[0], model_size[1]), dtype=np.float32)
    prompts_by_object = {int(prompt.object_id): prompt for prompt in prompts if prompt.object_id is not None}
    for object_idx, object_id in enumerate(object_ids):
        prompt = prompts_by_object.get(int(object_id))
        if prompt is not None:
            masks[object_idx, 0] = _prompt_seed_mask(model_size, _prompt_seed_box(prompt, model_size))
    return mx.array(masks)


def _resize_indices(in_size: int, out_size: int) -> np.ndarray:
    if out_size <= 0:
        raise ValueError("resize output size must be positive")
    if in_size == out_size:
        return np.arange(in_size, dtype=np.int64)
    coords = np.arange(out_size, dtype=np.float64) * (in_size / out_size)
    return np.minimum(np.floor(coords).astype(np.int64), in_size - 1)


def _resize_2d_nearest(x: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    y_idx = _resize_indices(int(x.shape[0]), int(size[0]))
    x_idx = _resize_indices(int(x.shape[1]), int(size[1]))
    return np.asarray(x)[y_idx][:, x_idx]


def _resize_chw_nearest(x: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    y_idx = _resize_indices(int(x.shape[1]), int(size[0]))
    x_idx = _resize_indices(int(x.shape[2]), int(size[1]))
    return np.asarray(x)[:, y_idx][:, :, x_idx]


def _as_chw(pixel_value: np.ndarray | None, fallback_size: tuple[int, int]) -> np.ndarray:
    if pixel_value is None:
        return np.zeros((3, fallback_size[0], fallback_size[1]), dtype=np.float32)
    arr = np.asarray(pixel_value, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.broadcast_to(arr[None, :, :], (3, arr.shape[0], arr.shape[1]))
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr[:3]
        if arr.shape[0] == 1:
            arr = np.broadcast_to(arr, (3, arr.shape[1], arr.shape[2]))
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = np.transpose(arr[..., :3], (2, 0, 1))
        if arr.shape[0] == 1:
            arr = np.broadcast_to(arr, (3, arr.shape[1], arr.shape[2]))
    else:
        raise ValueError(f"SAM3 video pixel_value must be CHW, HWC, or HW; got {arr.shape}")
    return np.nan_to_num(arr.astype(np.float32), copy=False)


def _channel_bank(pixel_chw: np.ndarray, size: tuple[int, int], channels: int, frame_index: int) -> np.ndarray:
    resized = _resize_chw_nearest(pixel_chw, size)
    out = np.empty((channels, int(size[0]), int(size[1])), dtype=np.float32)
    for channel_idx in range(channels):
        source = resized[channel_idx % resized.shape[0]]
        scale = 1.0 + (channel_idx / max(1, channels))
        out[channel_idx] = source * scale + (frame_index + 1) * 0.01 + (channel_idx % 7) * 0.001
    return out


def _position_bank(size: tuple[int, int], channels: int) -> np.ndarray:
    h, w = int(size[0]), int(size[1])
    y = (np.arange(h, dtype=np.float32) + 0.5) / h
    x = (np.arange(w, dtype=np.float32) + 0.5) / w
    yy = np.broadcast_to(y[:, None], (h, w))
    xx = np.broadcast_to(x[None, :], (h, w))
    base = np.stack([xx, yy], axis=0)
    repeats = int(np.ceil(channels / 2))
    return np.concatenate([base] * repeats, axis=0)[:channels].astype(np.float32)


def _frame_neural_features(
    pixel_value: np.ndarray | None,
    cfg: SAM3VideoConfig,
    num_buckets: int,
    frame_index: int,
) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array]]:
    pixel_chw = _as_chw(pixel_value, cfg.tracker.image_size)
    channels = cfg.tracker.hidden_dim
    image_features = _channel_bank(pixel_chw, cfg.tracker.feature_grid, channels, frame_index)
    image_pos = _position_bank(cfg.tracker.feature_grid, channels)
    low_res = _channel_bank(pixel_chw, cfg.decoder.low_res_mask_size, channels, frame_index)
    half_res_size = (
        max(1, cfg.decoder.low_res_mask_size[0] // 2),
        max(1, cfg.decoder.low_res_mask_size[1] // 2),
    )
    half_res = _channel_bank(pixel_chw, half_res_size, channels, frame_index)
    low_res = np.broadcast_to(low_res[None, ...], (num_buckets, channels, *cfg.decoder.low_res_mask_size)).copy()
    half_res = np.broadcast_to(half_res[None, ...], (num_buckets, channels, *half_res_size)).copy()
    return (
        mx.array(image_features[None, ...]),
        mx.array(image_pos[None, ...]),
        (mx.array(low_res), mx.array(half_res)),
    )


def _model_masks_to_public(mask_logits: mx.array, image_size: tuple[int, int]) -> np.ndarray:
    logits = np.asarray(mask_logits)
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"SAM3 video high-res masks must have shape (O,1,H,W), got {logits.shape}")
    masks = []
    for object_logits in logits[:, 0]:
        resized = _resize_2d_nearest(np.nan_to_num(object_logits.astype(np.float32)), image_size)
        threshold = float(np.mean(resized))
        mask = resized > threshold
        if not mask.any():
            y, x = np.unravel_index(int(np.argmax(resized)), resized.shape)
            mask[y, x] = True
        masks.append(mask)
    return np.stack(masks, axis=0)


def _mask_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return np.asarray([x0, y0, x1, y1], dtype=np.float64)
