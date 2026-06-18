"""Model-side SAM3 video module assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn

from .config import SAM3VideoConfig
from .multiplex_state import SAM3MultiplexController, SAM3MultiplexState
from .video_tracking import SAM3VideoMultiplexTrackerCore, SAM3VideoStageOutput

__all__ = ["SAM3VideoFrameOutput", "SAM3VideoModel"]


@dataclass
class SAM3VideoFrameOutput:
    frame_index: int
    track_ids: mx.array
    masks_bool: mx.array
    stage: SAM3VideoStageOutput
    multiplex_state: SAM3MultiplexState
    taps: dict[str, Any]


class SAM3VideoModel(nn.Module):
    """Thin MLX video engine wrapper for isolated SAM3 module tests."""

    def __init__(
        self,
        cfg: SAM3VideoConfig | None = None,
        *,
        tracker: SAM3VideoMultiplexTrackerCore | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SAM3VideoConfig()
        self.multiplex_controller = SAM3MultiplexController(self.cfg.tracker.multiplex_count)
        self.tracker = tracker or SAM3VideoMultiplexTrackerCore(self.cfg)

    def init_multiplex_state(self, object_ids: Sequence[int]) -> SAM3MultiplexState:
        return self.multiplex_controller.get_state(len(object_ids), object_ids=tuple(int(v) for v in object_ids))

    def track_step(
        self,
        *,
        frame_index: int,
        image_features: mx.array,
        image_pos_enc: mx.array | None = None,
        high_res_features: tuple[mx.array, ...] | list[mx.array] | None = None,
        mask_inputs: mx.array | None = None,
        is_init_cond_frame: bool = False,
        output_dict: dict[str, Any] | None = None,
        multiplex_state: SAM3MultiplexState,
        capture_taps: bool = False,
    ) -> SAM3VideoFrameOutput:
        stage = self.tracker.track_step(
            frame_index=frame_index,
            image_features=image_features,
            image_pos_enc=image_pos_enc,
            high_res_features=high_res_features,
            mask_inputs=mask_inputs,
            is_init_cond_frame=is_init_cond_frame,
            output_dict=output_dict,
            multiplex_state=multiplex_state,
            capture_taps=capture_taps,
        )
        track_ids = multiplex_state.object_ids_array
        masks_bool = stage.high_res_masks[:, 0, :, :] > 0
        taps = dict(stage.taps)
        if capture_taps:
            taps["output.track_ids"] = track_ids
            taps["output.masks_bool"] = masks_bool
            taps["output.multiplex"] = {
                "assignments": [bucket.copy() for bucket in multiplex_state.assignments],
                "object_ids": list(multiplex_state.object_ids),
                "multiplex_count": multiplex_state.multiplex_count,
            }
        return SAM3VideoFrameOutput(
            frame_index=int(frame_index),
            track_ids=track_ids,
            masks_bool=masks_bool,
            stage=stage,
            multiplex_state=multiplex_state,
            taps=taps,
        )
