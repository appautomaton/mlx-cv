import numpy as np
import mlx.core as mx
from types import SimpleNamespace

from mlx_cv.models.sam3 import SAM3VideoConfig, SAM3VideoSessionManager
from mlx_cv.prompts import BoxPrompt


def _frames(count=3):
    return [np.zeros((6, 8, 3), dtype=np.uint8) for _ in range(count)]


class _FakeVideoModel:
    def __init__(self):
        self.cfg = SAM3VideoConfig.tiny_fixture()
        self.calls = []

    def track_step(self, **kwargs):
        multiplex_state = kwargs["multiplex_state"]
        object_count = multiplex_state.total_valid_entries
        logits = np.zeros((object_count, 1, 32, 32), dtype=np.float32)
        for object_idx in range(object_count):
            logits[object_idx, 0, 16:32, 16:32] = 10.0 + object_idx
        score_logits = np.arange(object_count, dtype=np.float32).reshape(object_count, 1) + 2.0
        self.calls.append(
            {
                "frame_index": kwargs["frame_index"],
                "is_init_cond_frame": kwargs["is_init_cond_frame"],
                "mask_inputs_shape": None if kwargs["mask_inputs"] is None else tuple(kwargs["mask_inputs"].shape),
                "object_ids": tuple(multiplex_state.object_ids),
            }
        )
        return SimpleNamespace(
            stage=SimpleNamespace(
                high_res_masks=mx.array(logits),
                object_score_logits=mx.array(score_logits),
            )
        )


def test_sam3_video_propagation_returns_stable_tracker_ids_and_masks():
    manager = SAM3VideoSessionManager()
    state = manager.start_session(frames=_frames(), session_id="clip")
    manager.add_prompt(state.session_id, frame_index=0, prompt=BoxPrompt([[1, 1, 4, 4]]), object_id=9)

    video = manager.propagate_in_video(state.session_id)

    assert len(video) == 3
    assert video.to_dict()["metadata"]["claim_level"] == "mlx_neural_forward"
    assert video.to_dict()["metadata"]["tracker"] == "mlx_neural"
    assert video.frame_indices.tolist() == [0, 1, 2]
    assert [frame.tracks.ids.tolist() for frame in video.frames] == [[9], [9], [9]]
    assert [frame.tracks.frame_index for frame in video.frames] == [0, 1, 2]
    assert all(frame.masks.data.shape == (1, 6, 8) for frame in video.frames)
    assert all(frame.detections.track_ids.tolist() == frame.tracks.ids.tolist() for frame in video.frames)
    assert len(state.memory) == 3
    assert [(record.object_id, record.frame_index) for record in state.memory] == [(9, 0), (9, 1), (9, 2)]


def test_sam3_video_public_outputs_come_from_model_track_step_and_prompt_frame():
    fake = _FakeVideoModel()
    manager = SAM3VideoSessionManager(model=fake, multiplex_bucket_capacity=2)
    state = manager.start_session(frames=_frames(), session_id="fake")
    manager.add_prompt(state.session_id, frame_index=1, prompt=BoxPrompt([[0, 0, 2, 2]]), object_id=7)

    video = manager.propagate_in_video(state.session_id)

    assert [call["frame_index"] for call in fake.calls] == [0, 1, 2]
    assert fake.calls[0]["is_init_cond_frame"] is False
    assert fake.calls[0]["mask_inputs_shape"] is None
    assert fake.calls[1]["is_init_cond_frame"] is True
    assert fake.calls[1]["mask_inputs_shape"] == (1, 1, 32, 32)
    assert all(call["object_ids"] == (7,) for call in fake.calls)

    frame = video.frames[0]
    assert frame.masks.data.shape == (1, 6, 8)
    assert frame.masks.data[0, 3:6, 4:8].all()
    assert not frame.masks.data[0, :3, :4].any()
    np.testing.assert_array_equal(frame.detections.boxes, np.asarray([[4.0, 3.0, 8.0, 6.0]]))
    np.testing.assert_allclose(frame.detections.scores, [1.0 / (1.0 + np.exp(-2.0))])


def test_sam3_video_reverse_propagation_starts_at_requested_frame():
    fake = _FakeVideoModel()
    manager = SAM3VideoSessionManager(model=fake, multiplex_bucket_capacity=2)
    state = manager.start_session(frames=_frames(4), session_id="rev")
    manager.add_prompt(state.session_id, frame_index=2, prompt=BoxPrompt([[1, 1, 4, 4]]), object_id=5)

    video = manager.propagate_in_video(state.session_id, reverse=True, start_frame_index=2, max_frame_num_to_track=3)

    assert video.frame_indices.tolist() == [2, 1, 0]
    assert [call["frame_index"] for call in fake.calls] == [2, 1, 0]
    assert fake.calls[0]["is_init_cond_frame"] is True


def test_sam3_video_propagation_supports_visual_tracker_prompt():
    manager = SAM3VideoSessionManager()
    state = manager.start_session(frames=_frames(), session_id="visual")
    manager.add_prompt(state.session_id, frame_index=0, prompt=BoxPrompt([[1, 1, 4, 4]]), object_id=3)

    video = manager.handle_request({"type": "propagate_in_video", "session_id": state.session_id})

    assert [frame.tracks.ids.tolist() for frame in video.frames] == [[3], [3], [3]]
    assert all(
        frame.tracks.metadata == [{"prompt_mode": "sam3_tracker", "multiplex_bucket": 0}]
        for frame in video.frames
    )
    assert all(frame.masks.data.any() for frame in video.frames)


def test_sam3_video_propagation_keeps_masks_detections_and_tracks_aligned():
    manager = SAM3VideoSessionManager()
    state = manager.start_session(frames=_frames(4))
    manager.add_prompt(state.session_id, prompt=BoxPrompt([[1, 1, 4, 4]]), object_id=1)
    manager.add_prompt(state.session_id, prompt=BoxPrompt([[2, 2, 5, 5]]), object_id=2)

    video = manager.propagate_in_video(state.session_id, start_frame_index=1, max_frame_num_to_track=2)

    assert video.frame_indices.tolist() == [1, 2]
    for frame in video.frames:
        assert frame.masks.data.shape[0] == len(frame.tracks)
        assert len(frame.detections) == len(frame.tracks)
        assert frame.detections.track_ids.tolist() == frame.tracks.ids.tolist()
        assert frame.tracks.ids.tolist() == [1, 2]
