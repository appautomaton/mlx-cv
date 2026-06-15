"""Unit tests for the shared weight-convert rule engine (`hub/convert.py`)."""

import mlx.core as mx
import numpy as np

from mlx_cv.hub.convert import Drop, Rename, Transpose, convert_state_dict


def test_rename_transpose_drop_passthrough_round_trip():
    state = {
        "keep.weight": np.ones((2, 3), dtype=np.float32),
        "old.name": np.zeros((4,), dtype=np.float32),
        "conv.weight": np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5),
        "junk": np.ones((1,), dtype=np.float32),
    }
    rules = [
        Drop("junk"),
        Rename("old.name", "new.name"),
        Transpose("conv.weight", (0, 2, 3, 1)),
    ]
    out = dict(convert_state_dict(state, rules))

    assert "junk" not in out                                  # drop
    assert "new.name" in out and "old.name" not in out        # rename
    assert out["conv.weight"].shape == (2, 4, 5, 3)           # transpose (O,in,kH,kW)->(O,kH,kW,in)
    assert "keep.weight" in out and out["keep.weight"].shape == (2, 3)   # passthrough
    assert all(isinstance(v, mx.array) for v in out.values())  # → mlx arrays


def test_transpose_preserves_values_under_axis_move():
    state = {"w": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}
    out = dict(convert_state_dict(state, [Transpose("w", (2, 0, 1))]))
    assert out["w"].shape == (4, 2, 3)
    assert np.array_equal(np.array(out["w"]), np.transpose(state["w"], (2, 0, 1)))


def test_no_rules_is_identity_paths():
    state = {"a.b": np.ones((2,), dtype=np.float32), "c": np.zeros((3,), dtype=np.float32)}
    out = dict(convert_state_dict(state, []))
    assert set(out) == {"a.b", "c"}
