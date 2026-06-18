import numpy as np
import pytest

from mlx_cv.models.sam3 import SAM3VideoProcessorConfig, SAM3VideoSessionManager
from mlx_cv.prompts import BoxPrompt, ExemplarPrompt, PointPrompt, TextPrompt


def _manager():
    return SAM3VideoSessionManager()


def _frames():
    return [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]


def test_sam3_video_session_rejects_text_prompt_until_detector_path_is_ported():
    manager = _manager()
    state = manager.handle_request({"type": "start_session", "session_id": "s1", "frames": _frames()})

    with pytest.raises(NotImplementedError, match="text prompts require"):
        manager.handle_request(
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
    assert state.prompts == []
    assert state.multiplex_state.active_object_ids == ()


def test_sam3_video_session_admits_visual_prompt_for_tracker_boundary():
    manager = _manager()
    state = manager.start_session(frames=_frames())
    prompt = manager.add_prompt(state.session_id, frame_index=1, prompt=BoxPrompt([[0, 0, 2, 2]]))

    assert prompt.mode == "sam3_tracker"
    assert prompt.frame_index == 1


def test_sam3_video_handle_request_preserves_singular_box_prompt():
    manager = _manager()
    state = manager.start_session(frames=_frames(), session_id="box")

    prompt = manager.handle_request(
        {
            "type": "add_prompt",
            "session_id": state.session_id,
            "frame_index": 0,
            "box": [0, 0, 2, 2],
            "object_id": 8,
        }
    )

    assert prompt.mode == "sam3_tracker"
    assert prompt.object_id == 8
    np.testing.assert_array_equal(np.asarray(prompt.prompt["boxes"], dtype=np.float64), [0.0, 0.0, 2.0, 2.0])
    assert state.multiplex_state.active_object_ids == (8,)


def test_sam3_video_session_rejects_deferred_prompt_state():
    manager = _manager()
    state = manager.start_session(frames=_frames())

    with pytest.raises(NotImplementedError, match="point"):
        manager.add_prompt(state.session_id, prompt=PointPrompt([[1, 1]]))
    with pytest.raises(NotImplementedError, match="mask"):
        manager.add_prompt(state.session_id, prompt={"mask_prompt": np.zeros((2, 2))})
    with pytest.raises(NotImplementedError, match="exemplar"):
        manager.add_prompt(
            state.session_id,
            prompt=ExemplarPrompt(image=np.zeros((2, 2, 3), dtype=np.uint8), boxes=[[0, 0, 1, 1]]),
        )


def test_sam3_video_handle_request_rejects_unsupported_prompt_fields_without_mutation():
    manager = _manager()
    state = manager.start_session(frames=_frames(), session_id="req")

    bad_requests = [
        ({"type": "add_prompt", "session_id": state.session_id}, ValueError, "requires a prompt"),
        ({"type": "add_prompt", "session_id": state.session_id, "point": [[1, 1]]}, NotImplementedError, "points"),
        (
            {"type": "add_prompt", "session_id": state.session_id, "mask_prompt": np.zeros((2, 2))},
            NotImplementedError,
            "mask",
        ),
        (
            {
                "type": "add_prompt",
                "session_id": state.session_id,
                "exemplar": ExemplarPrompt(image=np.zeros((2, 2, 3), dtype=np.uint8), boxes=[[0, 0, 1, 1]]),
            },
            NotImplementedError,
            "exemplar",
        ),
    ]
    for request, exc_type, match in bad_requests:
        with pytest.raises(exc_type, match=match):
            manager.handle_request(request)
        assert state.prompts == []
        assert state.multiplex_state.active_object_ids == ()


def test_sam3_video_session_validates_frame_boundary_and_unknown_session():
    manager = _manager()
    state = manager.start_session(frames=_frames())

    with pytest.raises(ValueError, match="outside the session"):
        manager.add_prompt(state.session_id, frame_index=3, prompt=TextPrompt("person"))
    with pytest.raises(KeyError, match="unknown SAM3 video session"):
        manager.add_prompt("missing", prompt=BoxPrompt([[0, 0, 1, 1]]))


def test_sam3_video_session_accepts_custom_processor_config():
    manager = SAM3VideoSessionManager()
    manager.processor.config = SAM3VideoProcessorConfig(image_size=6)
    state = manager.start_session(frames=_frames())

    assert state.context.model_size == (6, 6)
