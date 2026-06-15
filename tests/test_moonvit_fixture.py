import pathlib

import numpy as np

from mlx_cv.parity import (
    MOONVIT_FIXTURE_CONFIG,
    load_case,
    moonvit_fixed_inputs,
    moonvit_tap_order,
)

_FIX = pathlib.Path(__file__).parent / "fixtures"


def test_moonvit_fixed_inputs_cover_same_grid_and_interpolated_grid():
    cfg = MOONVIT_FIXTURE_CONFIG
    inputs = moonvit_fixed_inputs()
    grid_hws = inputs["grid_hws"]
    pixel_values = inputs["pixel_values"]

    assert cfg["hidden_size"] // cfg["num_attention_heads"] % 4 == 0
    assert tuple(grid_hws[0]) == (cfg["init_pos_emb_height"], cfg["init_pos_emb_width"])
    assert tuple(grid_hws[1]) != (cfg["init_pos_emb_height"], cfg["init_pos_emb_width"])
    assert np.all(grid_hws % np.array(cfg["merge_kernel_size"], dtype=np.int32) == 0)
    assert pixel_values.shape == (
        int(np.sum(grid_hws[:, 0] * grid_hws[:, 1])),
        cfg["num_channels"],
        cfg["patch_size"],
        cfg["patch_size"],
    )


def test_moonvit_committed_fixture_schema_and_taps_match_config():
    cfg = MOONVIT_FIXTURE_CONFIG
    case = load_case(_FIX / "moonvit_tiny_fixture.npz")
    inputs = moonvit_fixed_inputs()

    assert case.name == cfg["name"]
    assert set(case.inputs) == {"pixel_values", "grid_hws"}
    assert np.array_equal(case.inputs["pixel_values"], inputs["pixel_values"])
    assert np.array_equal(case.inputs["grid_hws"], inputs["grid_hws"])
    assert list(case.taps) == moonvit_tap_order()
    assert set(case.expected) == {"norm", "merged_00", "merged_01"}
    assert case.expected["merged_00"].shape == (1, cfg["hidden_size"] * 4)
    assert case.expected["merged_01"].shape == (2, cfg["hidden_size"] * 4)
    assert case.taps["attention_mask_visible"].shape == (12, 12)
    assert not case.taps["attention_mask_visible"][:4, 4:].any()
    assert not case.taps["attention_mask_visible"][4:, :4].any()
