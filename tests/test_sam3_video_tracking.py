import numpy as np

from mlx_cv.models.sam3 import SAM3VideoSessionManager
from mlx_cv.prompts import BoxPrompt


def _frames(count=3):
    return [np.zeros((6, 8, 3), dtype=np.uint8) for _ in range(count)]


def test_sam3_video_propagation_returns_stable_text_track_ids_and_masks():
    manager = SAM3VideoSessionManager()
    state = manager.start_session(frames=_frames(), session_id="clip")
    manager.add_prompt(state.session_id, frame_index=0, text="person", object_id=9)

    video = manager.propagate_in_video(state.session_id)

    assert len(video) == 3
    assert video.to_dict()["metadata"]["claim_level"] == "local_contract_fixture"
    assert video.frame_indices.tolist() == [0, 1, 2]
    assert [frame.tracks.ids.tolist() for frame in video.frames] == [[9], [9], [9]]
    assert [frame.tracks.frame_index for frame in video.frames] == [0, 1, 2]
    assert all(frame.masks.data.shape == (1, 6, 8) for frame in video.frames)
    assert all(frame.detections.track_ids.tolist() == frame.tracks.ids.tolist() for frame in video.frames)
    assert len(state.memory) == 3
    assert [(record.object_id, record.frame_index) for record in state.memory] == [(9, 0), (9, 1), (9, 2)]


def test_sam3_video_propagation_supports_visual_tracker_prompt():
    manager = SAM3VideoSessionManager()
    state = manager.start_session(frames=_frames(), session_id="visual")
    manager.add_prompt(state.session_id, frame_index=0, prompt=BoxPrompt([[1, 1, 4, 4]]), object_id=3)

    video = manager.handle_request({"type": "propagate_in_video", "session_id": state.session_id})

    assert [frame.tracks.ids.tolist() for frame in video.frames] == [[3], [3], [3]]
    assert all(frame.tracks.metadata == [{"prompt_mode": "sam3_tracker"}] for frame in video.frames)
    assert all(frame.masks.data.any() for frame in video.frames)


def test_sam3_video_propagation_keeps_masks_detections_and_tracks_aligned():
    manager = SAM3VideoSessionManager()
    state = manager.start_session(frames=_frames(4))
    manager.add_prompt(state.session_id, prompt="person", object_id=1)
    manager.add_prompt(state.session_id, prompt="car", object_id=2)

    video = manager.propagate_in_video(state.session_id, start_frame_index=1, max_frame_num_to_track=2)

    assert video.frame_indices.tolist() == [1, 2]
    for frame in video.frames:
        assert frame.masks.data.shape[0] == len(frame.tracks)
        assert len(frame.detections) == len(frame.tracks)
        assert frame.detections.track_ids.tolist() == frame.tracks.ids.tolist()
        assert frame.tracks.ids.tolist() == [1, 2]
