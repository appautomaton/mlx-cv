"""Unit tests for the shared weight-convert rule engine (`hub/convert.py`)."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_cv.hub.convert import (
    Drop,
    PrefixRename,
    Rename,
    Transpose,
    TransposePattern,
    convert_state_dict,
)


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


def test_prefix_rename_then_exact_rename_and_pattern_transpose():
    state = {
        "pretrained.pos_embed": np.ones((1, 5, 8), dtype=np.float32),
        "pretrained.patch_embed.proj.weight": np.arange(
            2 * 3 * 4 * 5, dtype=np.float32
        ).reshape(2, 3, 4, 5),
    }
    out = dict(convert_state_dict(state, [
        PrefixRename("pretrained.", ""),
        Rename("pos_embed", "pos_embed.table"),
        TransposePattern("patch_embed.proj.weight", (0, 2, 3, 1), ndim=4),
    ]))

    assert "pos_embed.table" in out and "pretrained.pos_embed" not in out
    assert out["patch_embed.proj.weight"].shape == (2, 4, 5, 3)
    assert np.array_equal(
        np.array(out["patch_embed.proj.weight"]),
        np.transpose(state["pretrained.patch_embed.proj.weight"], (0, 2, 3, 1)),
    )


def test_path_patterns_lock_conv_and_conv_transpose_layouts():
    conv = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    deconv = np.arange(3 * 2 * 4 * 5, dtype=np.float32).reshape(3, 2, 4, 5)
    out = dict(convert_state_dict(
        {
            "projects.0.weight": conv,
            "resize_layers.0.weight": deconv,
        },
        [
            TransposePattern("projects.*.weight", (0, 2, 3, 1), ndim=4),
            TransposePattern("resize_layers.[01].weight", (1, 2, 3, 0), ndim=4),
        ],
    ))

    assert out["projects.0.weight"].shape == (2, 4, 5, 3)
    assert out["resize_layers.0.weight"].shape == (2, 4, 5, 3)
    assert np.array_equal(np.array(out["projects.0.weight"]), np.transpose(conv, (0, 2, 3, 1)))
    assert np.array_equal(
        np.array(out["resize_layers.0.weight"]),
        np.transpose(deconv, (1, 2, 3, 0)),
    )


def test_transpose_pattern_rejects_wrong_rank():
    with pytest.raises(ValueError, match="expected 4"):
        convert_state_dict(
            {"projects.0.weight": np.ones((2, 3), dtype=np.float32)},
            [TransposePattern("projects.*.weight", (0, 2, 3, 1), ndim=4)],
        )
