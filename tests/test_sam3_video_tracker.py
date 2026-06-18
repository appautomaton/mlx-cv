import numpy as np
import pytest

from mlx_cv.core.base import Task, Tracker
from mlx_cv.core.types import Result
from mlx_cv.models.sam3 import SAM3VideoTracker
from mlx_cv.prompts import BoxPrompt, PointPrompt


def _frame():
    return np.zeros((6, 8, 3), dtype=np.uint8)


def test_sam3_video_tracker_satisfies_spine_tracker_contract():
    assert issubclass(SAM3VideoTracker, Tracker)
    assert SAM3VideoTracker.task is Task.TRACKING


def test_sam3_video_tracker_init_then_step_streams_per_frame_results():
    tracker = SAM3VideoTracker()

    first = tracker.init(_frame(), BoxPrompt([[1, 1, 4, 4]]))
    assert isinstance(first, Result)
    assert first.tracks.frame_index == 0
    assert first.tracks.ids.tolist() == [1]  # object id auto-assigned
    assert first.masks.data.shape == (1, 6, 8)
    assert first.detections.track_ids.tolist() == first.tracks.ids.tolist()

    second = tracker.step(_frame())
    third = tracker.step(_frame())

    # frame indices advance with the stream; identity is carried forward
    assert [second.tracks.frame_index, third.tracks.frame_index] == [1, 2]
    assert second.tracks.ids.tolist() == [1]
    assert third.tracks.ids.tolist() == [1]
    assert third.masks.data.shape == (1, 6, 8)


def test_sam3_video_tracker_accumulates_memory_across_steps():
    tracker = SAM3VideoTracker()
    tracker.init(_frame(), BoxPrompt([[1, 1, 4, 4]]))
    tracker.step(_frame())
    tracker.step(_frame())

    # Streaming accumulates memory (batch propagate_in_video resets it each call).
    memory = tracker._state.memory
    assert [(rec.object_id, rec.frame_index) for rec in memory] == [(1, 0), (1, 1), (1, 2)]


def test_sam3_video_tracker_step_before_init_raises():
    with pytest.raises(RuntimeError, match="before init"):
        SAM3VideoTracker().step(_frame())


def test_sam3_video_tracker_double_init_raises():
    tracker = SAM3VideoTracker()
    tracker.init(_frame(), BoxPrompt([[1, 1, 4, 4]]))
    with pytest.raises(RuntimeError, match="already called"):
        tracker.init(_frame(), BoxPrompt([[1, 1, 4, 4]]))


def test_sam3_video_tracker_failed_init_rolls_back_session_state():
    tracker = SAM3VideoTracker()

    with pytest.raises(NotImplementedError, match="text prompts require"):
        tracker.init(_frame(), "person")

    assert tracker._state is None
    assert tracker.session_id is None
    assert tracker.manager.sessions == {}

    recovered = tracker.init(_frame(), BoxPrompt([[1, 1, 4, 4]]))
    assert recovered.tracks.ids.tolist() == [1]


def test_sam3_video_tracker_preserves_point_prompt_rejection():
    # PCS scope boundary is preserved end-to-end through the streaming surface.
    with pytest.raises(NotImplementedError, match="point prompts"):
        SAM3VideoTracker().init(_frame(), PointPrompt([[3, 3]]))
