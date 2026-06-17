import numpy as np

from mlx_cv.models.sam3 import SAM3VideoSessionManager


def _frames(count=3):
    return [np.zeros((6, 8, 3), dtype=np.uint8) for _ in range(count)]


def test_sam3_object_multiplex_assigns_fixed_capacity_buckets():
    manager = SAM3VideoSessionManager(multiplex_bucket_capacity=1)
    state = manager.start_session(frames=_frames())
    manager.add_prompt(state.session_id, prompt="person", object_id=1)
    manager.add_prompt(state.session_id, prompt="car", object_id=2)

    assert state.multiplex_state.object_to_bucket == {1: 0, 2: 1}
    assert [bucket.object_ids for bucket in state.multiplex_state.buckets] == [(1,), (2,)]

    video = manager.propagate_in_video(state.session_id)
    multiplex = video.to_dict()["metadata"]["multiplex"]

    assert multiplex["bucket_capacity"] == 1
    assert multiplex["object_to_bucket"] == {"1": 0, "2": 1}
    assert multiplex["active_object_ids"] == [1, 2]


def test_sam3_object_multiplex_track_metadata_matches_bucket_state():
    manager = SAM3VideoSessionManager(multiplex_bucket_capacity=2)
    state = manager.start_session(frames=_frames())
    manager.add_prompt(state.session_id, prompt="person", object_id=10)
    manager.add_prompt(state.session_id, prompt="bag", object_id=11)

    video = manager.propagate_in_video(state.session_id)

    assert state.multiplex_state.object_to_bucket == {10: 0, 11: 0}
    for frame in video.frames:
        assert frame.tracks.ids.tolist() == [10, 11]
        assert frame.tracks.metadata == [
            {"prompt_mode": "sam3_video", "multiplex_bucket": 0},
            {"prompt_mode": "sam3_video", "multiplex_bucket": 0},
        ]
    assert len(state.multiplex_state.memory) == 6


def test_sam3_object_multiplex_updates_when_objects_are_removed_and_added():
    manager = SAM3VideoSessionManager(multiplex_bucket_capacity=2)
    state = manager.start_session(frames=_frames())
    manager.add_prompt(state.session_id, prompt="person", object_id=1)
    manager.add_prompt(state.session_id, prompt="car", object_id=2)
    manager.handle_request({"type": "remove_object", "session_id": state.session_id, "object_id": 1})
    manager.add_prompt(state.session_id, prompt="dog", object_id=3)

    assert state.multiplex_state.object_to_bucket == {2: 0, 3: 0}
    assert state.multiplex_state.active_object_ids == (2, 3)

    video = manager.propagate_in_video(state.session_id, max_frame_num_to_track=1)
    frame = video.frames[0]
    assert frame.tracks.ids.tolist() == [2, 3]
    assert frame.detections.track_ids.tolist() == [2, 3]
