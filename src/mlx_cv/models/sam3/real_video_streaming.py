"""Faithful SAM3 video streaming session (slice 13).

A single-object inference session over the faithful :class:`Sam3VideoModel`: seed a box
prompt on one frame, then propagate the mask across the clip via the SAM2-style memory
bank (:meth:`Sam3TrackerVideoModel.track_step`). Each frame's encoded memory + object
pointer is appended to the bank; the next frame attends to the most recent
``num_maskmem`` entries through the 4-layer memory-attention transformer.

Object-Multiplex multi-object association is a later slice; this is the per-object
streaming forward. No torch/transformers imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .real_video_model import Sam3TrackerStageOutput, Sam3TrackerVideoModel, Sam3VideoModel

__all__ = ["Sam3VideoFrameResult", "Sam3VideoSession"]


@dataclass
class Sam3VideoFrameResult:
    """One frame of a streaming result (single object per ``object_ids`` entry)."""

    frame_index: int
    object_ids: list[int]
    masks: mx.array  # [num_obj, h, w] bool (low-res tracker mask)
    object_score_logits: mx.array  # [num_obj]


def _box_corner_points(box: tuple[float, float, float, float]) -> mx.array:
    """xyxy box (prompt-encoder input-image coords) -> 2 SAM corner points ``[2, 2]``."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return mx.array([[x0, y0], [x1, y1]])


_BOX_CORNER_LABELS = (2.0, 3.0)  # SAM box corners: top-left=2, bottom-right=3


@dataclass
class _Prompt:
    frame_index: int
    box: tuple[float, float, float, float]
    object_id: int


class Sam3VideoSession:
    """Single-object streaming session over the faithful tracker."""

    def __init__(self, model: Sam3VideoModel, *, num_maskmem: int | None = None):
        self.model = model
        self.tracker = model.tracker_model
        self.num_maskmem = num_maskmem if num_maskmem is not None else self.tracker.num_maskmem
        self._prompts: list[_Prompt] = []

    @classmethod
    def from_tracker(cls, tracker: Sam3TrackerVideoModel, *, num_maskmem: int | None = None) -> "Sam3VideoSession":
        """Build a session around a tracker only (no detector) — for feature-injection use."""
        session = cls.__new__(cls)
        session.model = None
        session.tracker = tracker
        session.num_maskmem = num_maskmem if num_maskmem is not None else tracker.num_maskmem
        session._prompts = []
        return session

    def add_box_prompt(self, frame_index: int, box, object_id: int) -> None:
        """Seed an object with a box prompt. Multiple objects may be added (multiplex batch)."""
        self._prompts.append(_Prompt(frame_index=int(frame_index), box=tuple(box), object_id=int(object_id)))

    def propagate(self, pixel_values_per_frame) -> list[Sam3VideoFrameResult]:
        """Extract per-frame tracker features via the detector, then run the streaming loop."""
        if self.model is None:
            raise ValueError("Sam3VideoSession.propagate needs a full Sam3VideoModel; use run_from_features otherwise")
        features = [self.model.extract_tracker_features(pv) for pv in pixel_values_per_frame]
        return self.run_from_features(features)

    @staticmethod
    def _to_object_batch(x: mx.array, num_objects: int) -> mx.array:
        """Broadcast a single shared-frame tensor ``[1, ...]`` to the object batch ``[B, ...]``."""
        if x.shape[0] == num_objects:
            return x
        if x.shape[0] != 1:
            raise ValueError(f"expected frame tensor with leading dim 1 or {num_objects}, got {x.shape[0]}")
        return mx.repeat(x, num_objects, axis=0)

    def _seed_point_inputs(self, prompts: list[_Prompt]) -> tuple[mx.array, mx.array]:
        """Batched box-corner prompts for the seed frame -> ``(coords [B,2,2], labels [B,2])``."""
        coords = mx.stack([_box_corner_points(p.box) for p in prompts], axis=0)
        labels = mx.broadcast_to(mx.array([_BOX_CORNER_LABELS]), (len(prompts), len(_BOX_CORNER_LABELS)))
        return coords, labels

    def run_from_features(self, per_frame_features) -> list[Sam3VideoFrameResult]:
        """Run the memory-propagation loop over pre-extracted per-frame features.

        Each entry is ``(image_embeddings, image_positional_embeddings, high_res_features)`` for a
        single frame (shared across objects); ``high_res_features`` is ``[4g-res, 2g-res]`` raw FPN
        levels (NHWC). All seeded objects are tracked together as the batch dimension ``B`` (Object
        Multiplex): each object carries its own memory + box prompt; per frame they run through one
        ``track_step`` over the shared frame features.
        """
        if not self._prompts:
            raise ValueError("Sam3VideoSession requires a prompt; call add_box_prompt first")
        seed_frames = {p.frame_index for p in self._prompts}
        if len(seed_frames) != 1:
            raise ValueError(
                "Sam3VideoSession batches objects that share one seed frame; per-object seed frames / "
                "mid-clip spawning is handled by the association layer (later slice)"
            )
        prompts = self._prompts
        seed_frame = prompts[0].frame_index
        object_ids = [p.object_id for p in prompts]
        num_objects = len(prompts)
        bank: list[Sam3TrackerStageOutput] = []
        results: list[Sam3VideoFrameResult] = []
        for frame_index, (vision_features, vision_pos, high_res_features) in enumerate(per_frame_features):
            is_init = frame_index == seed_frame
            out = self.tracker.track_step(
                vision_features=self._to_object_batch(vision_features, num_objects),
                vision_pos=self._to_object_batch(vision_pos, num_objects),
                high_res_features=[self._to_object_batch(h, num_objects) for h in high_res_features],
                is_init_cond_frame=is_init,
                point_inputs=self._seed_point_inputs(prompts) if is_init else None,
                previous_frames=bank if bank else None,
            )
            bank.append(out)
            results.append(
                Sam3VideoFrameResult(
                    frame_index=frame_index,
                    object_ids=list(object_ids),
                    masks=out.low_res_masks[:, :, :, 0] > 0,  # [B, h, w] (one row per object)
                    object_score_logits=out.object_score_logits.reshape(-1),
                )
            )
        return results
