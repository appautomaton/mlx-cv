"""Canonical SAM 3.1 multiplex-video session API on MLX Metal."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlx.core as mx
import numpy as np

from .multiplex_state import SAM3MultiplexController, SAM3MultiplexState
from .sam31_processor import SAM3VideoProcessor
from .sam31_video import SAM3VideoModel

__all__ = [
    "SAM3VideoFrameResult",
    "SAM3VideoSession",
    "SAM3VideoSessionState",
]


def _sigmoid(x: mx.array) -> mx.array:
    return 1.0 / (1.0 + mx.exp(-x))


def _resize_axis_half_pixel(x: mx.array, out_size: int, axis: int) -> mx.array:
    in_size = int(x.shape[axis])
    if in_size == out_size:
        return x
    coords = (mx.arange(out_size, dtype=mx.float32) + 0.5) * (
        in_size / out_size
    ) - 0.5
    lower_raw = mx.floor(coords).astype(mx.int32)
    upper_raw = lower_raw + 1
    lower = mx.clip(lower_raw, 0, in_size - 1)
    upper = mx.clip(upper_raw, 0, in_size - 1)
    weight = coords - lower_raw.astype(mx.float32)
    left = mx.take(x, lower, axis=axis)
    right = mx.take(x, upper, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = out_size
    weight = weight.reshape(shape)
    return left * (1.0 - weight) + right * weight


def _resize_bilinear_nhwc(x: mx.array, size: tuple[int, int]) -> mx.array:
    x = _resize_axis_half_pixel(x, int(size[0]), 1)
    return _resize_axis_half_pixel(x, int(size[1]), 2)


@dataclass
class _Prompt:
    frame_index: int
    object_id: int
    coords: np.ndarray | None = None
    labels: np.ndarray | None = None
    mask: np.ndarray | None = None


@dataclass
class _Memory:
    frame_index: int
    features: mx.array
    position: mx.array
    image: mx.array
    image_position: mx.array
    object_pointers: mx.array


@dataclass
class SAM3VideoFrameResult:
    frame_index: int
    object_ids: tuple[int, ...]
    masks: np.ndarray
    scores: np.ndarray
    bucket_assignments: tuple[tuple[int, ...], ...]


@dataclass
class SAM3VideoSessionState:
    session_id: str
    pixel_values: np.ndarray
    context: Any
    prompts: dict[int, _Prompt] = field(default_factory=dict)
    active_object_ids: list[int] = field(default_factory=list)
    memories: list[_Memory] = field(default_factory=list)


class SAM3VideoSession:
    """Official request names backed by the SAM 3.1 TriHead + multiplex tracker."""

    def __init__(
        self,
        model: SAM3VideoModel | None = None,
        processor: SAM3VideoProcessor | None = None,
        *,
        bucket_capacity: int = 16,
    ):
        self.model = model or SAM3VideoModel()
        self.processor = processor or SAM3VideoProcessor()
        self.controller = SAM3MultiplexController(
            16, eval_multiplex_count=bucket_capacity
        )
        self.sessions: dict[str, SAM3VideoSessionState] = {}

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool | None = None,
        token: str | bool | None = None,
        bucket_capacity: int = 16,
    ) -> "SAM3VideoSession":
        from ...hub import resolve_pretrained
        from .sam31_checkpoint import load_sam3_video_weights

        resolved = resolve_pretrained(
            checkpoint,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        if resolved.is_dir():
            resolved = resolved / "model.safetensors"
        model = load_sam3_video_weights(SAM3VideoModel(), resolved)
        return cls(model=model, bucket_capacity=bucket_capacity)

    def start_session(
        self, *, frames: Any, session_id: str | None = None
    ) -> SAM3VideoSessionState:
        processed, context = self.processor.preprocess(frames)
        identifier = session_id or uuid4().hex
        if identifier in self.sessions:
            raise ValueError(f"SAM 3.1 video session already exists: {identifier}")
        state = SAM3VideoSessionState(
            identifier, processed["pixel_values"], context
        )
        self.sessions[identifier] = state
        return state

    def add_prompt(
        self,
        session_id: str,
        *,
        frame_index: int,
        object_id: int | None = None,
        box: Any = None,
        points: Any = None,
        labels: Any = None,
        mask: Any = None,
    ) -> int:
        state = self._session(session_id)
        if not 0 <= int(frame_index) < len(state.pixel_values):
            raise ValueError("SAM 3.1 prompt frame is outside the session")
        if sum(value is not None for value in (box, points, mask)) != 1:
            raise ValueError("SAM 3.1 add_prompt requires exactly one of box/points/mask")
        if object_id is None:
            object_id = max(state.active_object_ids, default=0) + 1
        object_id = int(object_id)
        transform = state.context.frames[int(frame_index)].transform
        prompt = _Prompt(int(frame_index), object_id)
        if box is not None:
            x0, y0, x1, y1 = transform.apply_boxes(np.asarray(box)).reshape(4)
            prompt.coords = np.asarray([[x0, y0], [x1, y1]], dtype=np.float32)
            prompt.labels = np.asarray([2, 3], dtype=np.int32)
        elif points is not None:
            prompt.coords = transform.apply_points(points).astype(np.float32)
            prompt.labels = np.asarray(
                np.ones((len(prompt.coords),), dtype=np.int32)
                if labels is None
                else labels,
                dtype=np.int32,
            )
        else:
            prompt.mask = transform.apply_dense(mask, mode="bilinear").astype(np.float32)
        state.prompts[object_id] = prompt
        if object_id not in state.active_object_ids:
            state.active_object_ids.append(object_id)
            state.active_object_ids.sort()
            # Bucket topology changed; old bucket-space memory is no longer valid.
            state.memories.clear()
        return object_id

    def remove_object(self, session_id: str, object_id: int) -> None:
        state = self._session(session_id)
        object_id = int(object_id)
        if object_id not in state.active_object_ids:
            raise KeyError(f"SAM 3.1 object is not active: {object_id}")
        state.active_object_ids.remove(object_id)
        state.prompts.pop(object_id, None)
        state.memories.clear()

    def reset_session(self, session_id: str) -> None:
        state = self._session(session_id)
        state.prompts.clear()
        state.active_object_ids.clear()
        state.memories.clear()

    def propagate_in_video(
        self,
        session_id: str,
        *,
        start_frame_index: int = 0,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> list[SAM3VideoFrameResult]:
        state = self._session(session_id)
        if not state.active_object_ids:
            raise ValueError("SAM 3.1 video propagation requires at least one prompt")
        indices = (
            list(range(int(start_frame_index), -1, -1))
            if reverse
            else list(range(int(start_frame_index), len(state.pixel_values)))
        )
        if max_frame_num_to_track is not None:
            indices = indices[: max(0, int(max_frame_num_to_track))]
        state.memories.clear()
        results = []
        for frame_index in indices:
            results.append(self._run_frame(state, frame_index))
        return results

    def _session(self, session_id: str) -> SAM3VideoSessionState:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown SAM 3.1 video session: {session_id}") from exc

    def _state(self, state: SAM3VideoSessionState) -> SAM3MultiplexState:
        return self.controller.get_state(
            len(state.active_object_ids), object_ids=state.active_object_ids
        )

    def _interactive_outputs(self, vision, prompts: list[_Prompt]):
        tracker = self.model.tracker
        count = len(prompts)
        low = mx.repeat(vision.interactive_hidden_states[-1], count, axis=0)
        high = [
            mx.repeat(
                tracker.interactive_sam_mask_decoder.conv_s0(
                    vision.interactive_hidden_states[0]
                ),
                count,
                axis=0,
            ),
            mx.repeat(
                tracker.interactive_sam_mask_decoder.conv_s1(
                    vision.interactive_hidden_states[1]
                ),
                count,
                axis=0,
            ),
        ]
        max_points = max(
            (len(prompt.coords) if prompt.coords is not None else 1)
            for prompt in prompts
        )
        coord_rows, label_rows, dense_rows = [], [], []
        for prompt in prompts:
            coords = (
                np.zeros((1, 2), dtype=np.float32)
                if prompt.coords is None
                else prompt.coords
            )
            labels = (
                np.full((1,), -1, dtype=np.int32)
                if prompt.labels is None
                else prompt.labels
            )
            pad = max_points - len(coords)
            coord_rows.append(np.pad(coords, ((0, pad), (0, 0))))
            label_rows.append(np.pad(labels, (0, pad), constant_values=-1))
            if prompt.mask is None:
                dense_rows.append(None)
            else:
                mask = mx.array(prompt.mask)[None, :, :, None]
                dense_rows.append(_resize_bilinear_nhwc(mask, (288, 288))[0])
        sparse = tracker.interactive_sam_prompt_encoder.encode_sparse(
            mx.array(np.stack(coord_rows)), mx.array(np.stack(label_rows))
        )[:, None]
        dense = []
        for row in dense_rows:
            dense.append(
                tracker.interactive_sam_prompt_encoder.encode_dense(
                    None if row is None else row[None], 1
                )[0]
            )
        multimask_output = all(
            prompt.coords is None or len(prompt.coords) <= 1 for prompt in prompts
        )
        decoded = tracker.interactive_sam_mask_decoder(
            low,
            mx.repeat(
                tracker.interactive_sam_prompt_encoder.get_dense_pe(), count, axis=0
            ),
            sparse,
            mx.stack(dense),
            high,
            multimask_output=multimask_output,
        )
        masks = decoded.masks[:, 0]
        ious = decoded.iou_pred[:, 0]
        tokens = decoded.sam_tokens_out[:, 0]
        candidates = int(ious.shape[-1])
        choice = mx.argmax(ious, axis=-1)
        one_hot = mx.arange(candidates)[None] == choice[:, None]
        selected_masks = mx.sum(masks * one_hot[:, :, None, None], axis=1)
        selected_tokens = mx.sum(tokens * one_hot[:, :, None], axis=1)
        selected_ious = mx.sum(ious * one_hot, axis=1)
        object_logits = decoded.object_score_logits[:, 0, 0]
        pointers = tracker.interactive_obj_ptr_proj(selected_tokens)
        appearing = (object_logits > 0).astype(pointers.dtype)[:, None]
        pointers = appearing * pointers + (1 - appearing) * tracker.no_obj_ptr_linear(
            pointers
        )
        return selected_masks, selected_ious, object_logits, pointers

    def _propagation_outputs(
        self, vision, mux: SAM3MultiplexState, memories, frame_index: int | None = None
    ):
        tracker = self.model.tracker
        buckets = mux.num_buckets
        current = mx.repeat(vision.propagation_hidden_states[-1], buckets, axis=0)
        current_pos = mx.repeat(
            vision.propagation_position_encoding[-1], buckets, axis=0
        )
        query = current.reshape(buckets, -1, 256)
        query_pos = current_pos.reshape(buckets, -1, 256)

        spatial_memory, spatial_image, spatial_pos = [], [], []
        for distance, memory in enumerate(reversed(memories[-6:]), start=1):
            index = max(0, 6 - distance)
            temporal = tracker.maskmem_tpos_enc[index].reshape(1, 1, 256)
            spatial_memory.append(memory.features.reshape(buckets, -1, 256))
            spatial_image.append(memory.image.reshape(buckets, -1, 256))
            spatial_pos.append(
                memory.image_position.reshape(buckets, -1, 256) + temporal
            )
        memory = mx.concatenate(spatial_memory, axis=1)
        memory_image = mx.concatenate(spatial_image, axis=1)
        memory_image_pos = mx.concatenate(spatial_pos, axis=1)

        pointer_memories = memories[-16:]
        pointer_rows = [row.object_pointers for row in pointer_memories]
        pointers = mx.concatenate(pointer_rows, axis=1) if pointer_rows else None
        pointer_count = 0 if pointers is None else int(pointers.shape[1])
        if pointers is not None:
            relative = mx.array(
                [
                    0.0
                    if frame_index is None
                    else (frame_index - row.frame_index) / 15.0
                    for row in pointer_memories
                ],
                dtype=mx.float32,
            )
            half = 128
            dimension = mx.arange(half, dtype=mx.float32)
            scale = 10000.0 ** (2.0 * mx.floor(dimension / 2.0) / half)
            angles = relative[:, None] / scale[None]
            temporal = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)
            temporal = tracker.obj_ptr_tpos_proj(temporal)
            temporal = mx.repeat(temporal, 16, axis=0)[None]
            temporal = mx.repeat(temporal, buckets, axis=0)
            memory = mx.concatenate([memory, pointers], axis=1)
            memory_image = mx.concatenate(
                [memory_image, mx.zeros_like(pointers)], axis=1
            )
            memory_image_pos = mx.concatenate(
                [memory_image_pos, temporal], axis=1
            )
        fused = tracker.transformer.encoder(
            query,
            query,
            memory_image,
            memory,
            object_pos=query_pos,
            memory_image_pos=memory_image_pos,
            num_object_pointer_tokens=pointer_count,
        ).reshape(current.shape)

        high = [
            mx.repeat(
                tracker.sam_mask_decoder.conv_s0(
                    vision.propagation_hidden_states[0]
                ),
                buckets,
                axis=0,
            ),
            mx.repeat(
                tracker.sam_mask_decoder.conv_s1(
                    vision.propagation_hidden_states[1]
                ),
                buckets,
                axis=0,
            ),
        ]
        valid = mux.valid_mask.astype(fused.dtype)[:, :, None]
        suppression = (
            valid * tracker.output_valid_embed[None]
            + (1 - valid) * tracker.output_invalid_embed[None]
        )
        decoded = tracker.sam_mask_decoder(
            fused,
            tracker.image_pe_layer.dense(fused.shape[1], fused.shape[2]),
            high,
            suppression,
            multimask_output=True,
        )
        masks = mux.demux(decoded["masks"])
        ious = mux.demux(decoded["iou_pred"])
        tokens = mux.demux(decoded["sam_tokens_out"])
        object_logits = mux.demux(decoded["object_score_logits"])[..., 0]
        choice = mx.argmax(ious, axis=-1)
        one_hot = mx.arange(3)[None] == choice[:, None]
        selected_masks = mx.sum(masks * one_hot[:, :, None, None], axis=1)
        selected_tokens = mx.sum(tokens * one_hot[:, :, None], axis=1)
        selected_ious = mx.sum(ious * one_hot, axis=1)
        pointers = tracker.obj_ptr_proj(selected_tokens)
        appearing = (object_logits > 0).astype(pointers.dtype)[:, None]
        pointers = appearing * pointers + (1 - appearing) * tracker.no_obj_ptr_linear(
            pointers
        )
        return selected_masks, selected_ious, object_logits, pointers

    def _encode_memory(
        self,
        frame_index: int,
        vision,
        mux: SAM3MultiplexState,
        low_masks: mx.array,
        object_logits: mx.array,
        pointers: mx.array,
        conditioning: bool,
    ) -> _Memory:
        tracker = self.model.tracker
        high_masks = _resize_bilinear_nhwc(low_masks[..., None], (1008, 1008))
        probabilities = _sigmoid(high_masks) * 2.0 - 1.0
        mux_masks = mux.mux(probabilities)[..., 0].transpose(0, 2, 3, 1)
        condition_value = 1.0 if conditioning else 0.0
        conditions = mx.ones_like(probabilities) * condition_value
        mux_conditions = mux.mux(conditions)[..., 0].transpose(0, 2, 3, 1)
        memory_masks = mx.concatenate([mux_masks, mux_conditions], axis=-1)
        buckets = mux.num_buckets
        image = mx.repeat(vision.propagation_hidden_states[-1], buckets, axis=0)
        position = mx.repeat(
            vision.propagation_position_encoding[-1], buckets, axis=0
        )
        features, memory_position = tracker.maskmem_backbone(image, memory_masks)
        appearing = mux.mux((object_logits > 0).astype(features.dtype)[:, None])
        empty_embed = mx.sum(
            (1 - appearing) * tracker.no_obj_embed_spatial[None], axis=1
        )
        features = features + empty_embed[:, None, None]
        return _Memory(
            frame_index,
            features,
            memory_position,
            image,
            position,
            mux.mux(pointers).reshape(buckets, 16, 256),
        )

    def _run_frame(
        self, state: SAM3VideoSessionState, frame_index: int
    ) -> SAM3VideoFrameResult:
        mux = self._state(state)
        pixels = mx.array(state.pixel_values[frame_index : frame_index + 1])
        vision = self.model.detector.vision_encoder(pixels)
        frame_prompts = [
            state.prompts[object_id]
            for object_id in state.active_object_ids
            if state.prompts[object_id].frame_index == frame_index
        ]
        if frame_prompts:
            if len(frame_prompts) != len(state.active_object_ids):
                raise ValueError(
                    "SAM 3.1 currently requires all active objects to share a correction frame"
                )
            low_masks, ious, logits, pointers = self._interactive_outputs(
                vision, frame_prompts
            )
            conditioning = True
        elif state.memories:
            low_masks, ious, logits, pointers = self._propagation_outputs(
                vision, mux, state.memories, frame_index
            )
            conditioning = False
        else:
            raise ValueError(
                "SAM 3.1 propagation reached a frame before its conditioning prompt"
            )
        low_masks = mx.where(logits[:, None, None] > 0, low_masks, -1024.0)
        state.memories.append(
            self._encode_memory(
                frame_index,
                vision,
                mux,
                low_masks,
                logits,
                pointers,
                conditioning,
            )
        )
        state.memories[:] = state.memories[-16:]
        model_masks = _resize_bilinear_nhwc(
            low_masks[..., None], (1008, 1008)
        )[..., 0]
        mx.eval(model_masks, ious, logits)
        context = state.context.frames[frame_index]
        masks = np.stack(
            [context.transform.invert_mask(np.asarray(mask) > 0) for mask in model_masks]
        )
        scores = np.asarray(_sigmoid(logits), dtype=np.float32)
        return SAM3VideoFrameResult(
            frame_index,
            tuple(state.active_object_ids),
            masks,
            scores,
            mux.assignments_tuple,
        )
