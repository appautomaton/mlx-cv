import numpy as np
import pytest

from mlx_cv.models.sam3 import SAM3VideoProcessorConfig, SAM3VideoSessionManager
from mlx_cv.prompts import BoxPrompt, PointPrompt, TextPrompt


def _manager():
    return SAM3VideoSessionManager()


def _frames():
    return [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]


def test_sam3_video_session_start_add_text_and_request_flow():
    manager = _manager()
    state = manager.handle_request({"type": "start_session", "session_id": "s1", "frames": _frames()})
    prompt = manager.handle_request(
        {
            "type": "add_prompt",
            "session_id": state.session_id,
            "frame_index": 0,
            "text": "person",
            "object_id": 4,
        }
    )

    assert state.session_id == "s1"
    assert state.context.frame_count == 2
    assert prompt.mode == "sam3_video"
    assert prompt.texts == ("person",)
    assert prompt.object_id == 4
    assert state.prompts == [prompt]


def test_sam3_video_session_admits_visual_prompt_for_tracker_boundary():
    manager = _manager()
    state = manager.start_session(frames=_frames())
    prompt = manager.add_prompt(state.session_id, frame_index=1, prompt=BoxPrompt([[0, 0, 2, 2]]))

    assert prompt.mode == "sam3_tracker"
    assert prompt.frame_index == 1


def test_sam3_video_session_rejects_deferred_prompt_state():
    manager = _manager()
    state = manager.start_session(frames=_frames())

    with pytest.raises(NotImplementedError, match="point"):
        manager.add_prompt(state.session_id, prompt=PointPrompt([[1, 1]]))
    with pytest.raises(NotImplementedError, match="mask"):
        manager.add_prompt(state.session_id, prompt={"mask_prompt": np.zeros((2, 2))})


def test_sam3_video_session_validates_frame_and_propagation_boundary():
    manager = _manager()
    state = manager.start_session(frames=_frames())

    with pytest.raises(ValueError, match="outside the session"):
        manager.add_prompt(state.session_id, frame_index=3, prompt=TextPrompt("person"))
    with pytest.raises(NotImplementedError, match="tracker memory"):
        manager.propagate_in_video(state.session_id)
    with pytest.raises(KeyError, match="unknown SAM3 video session"):
        manager.add_prompt("missing", prompt="person")


def test_sam3_video_session_accepts_custom_processor_config():
    manager = SAM3VideoSessionManager()
    manager.processor.config = SAM3VideoProcessorConfig(image_size=6)
    state = manager.start_session(frames=_frames())

    assert state.context.model_size == (6, 6)
