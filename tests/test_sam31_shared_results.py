import numpy as np
import pytest


pytest.importorskip("mlx")

from mlx_cv import Masks, Result, Tracks, VideoResult
from mlx_cv.models.sam3 import SAM3ImagePrediction, SAM3VideoFrameResult
from mlx_cv.models.sam3.sam31_session import SAM3VideoSession, SAM3VideoSessionState


def test_sam_result_names_are_shared_contract_aliases():
    assert SAM3ImagePrediction is Result
    assert SAM3VideoFrameResult is Result


def test_sam_video_propagation_returns_video_result():
    session = object.__new__(SAM3VideoSession)
    state = SAM3VideoSessionState(
        session_id="session",
        pixel_values=np.zeros((2, 3, 4, 4), dtype=np.float32),
        context=None,
        active_object_ids=[7],
    )
    session.sessions = {"session": state}

    def fake_frame(_state, frame_index):
        return Result(
            image_size=(4, 4),
            masks=Masks(np.ones((1, 4, 4), dtype=bool)),
            tracks=Tracks([7], frame_index=frame_index, scores=[0.9]),
        )

    session._run_frame = fake_frame

    result = session.propagate_in_video("session")

    assert isinstance(result, VideoResult)
    assert result.session_id == "session"
    assert result.frame_indices.tolist() == [0, 1]
    assert result[0].tracks.ids.tolist() == [7]
    assert [frame.tracks.frame_index for frame in result] == [0, 1]
