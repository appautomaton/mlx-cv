import numpy as np
import pytest

from mlx_cv.parity import (
    DINOV3_VARIANT,
    ParityCase,
    allclose_tree,
    assert_parity,
    bisect,
    dinov3_fixed_input,
    dinov3_tap_order,
    load_case,
    save_case,
)


def test_allclose_tree():
    a = {"x": np.zeros(3), "y": [1, 2, 3]}
    assert allclose_tree(a, {"x": np.zeros(3), "y": [1, 2, 3]})
    assert not allclose_tree(a, {"x": np.ones(3), "y": [1, 2, 3]})


def test_assert_parity_raises():
    with pytest.raises(AssertionError):
        assert_parity({"x": np.ones(2)}, {"x": np.zeros(2)}, name="x")


def test_bisect_finds_first_divergence():
    ref = {"a": np.zeros(2), "b": np.zeros(2), "c": np.zeros(2)}
    got = {"a": np.zeros(2), "b": np.ones(2), "c": np.zeros(2)}
    assert bisect(ref, got) == "b"
    assert bisect(ref, ref) is None


def test_paritycase():
    pc = ParityCase(name="t", inputs=1, expected=2)
    assert pc.name == "t" and pc.taps is None


# -- fixture schema + fixed input (Slice 3) --------------------------------

def _synthetic_case() -> ParityCase:
    rng = np.random.default_rng(0)
    return ParityCase(
        name="dinov3_vit_small",
        inputs={"x": rng.standard_normal((1, 3, 8, 8)).astype(np.float32)},
        expected={
            "x_norm_clstoken": rng.standard_normal((1, 4)).astype(np.float32),
            "x_storage_tokens": rng.standard_normal((1, 2, 4)).astype(np.float32),
            "x_norm_patchtokens": rng.standard_normal((1, 4, 4)).astype(np.float32),
        },
        taps={k: rng.standard_normal((1, 4)).astype(np.float32)
              for k in ["patch_embed", "block_00", "block_01", "norm"]},
    )


def test_save_load_case_roundtrip(tmp_path):
    case = _synthetic_case()
    p = tmp_path / "case.npz"
    save_case(case, p)
    loaded = load_case(p)
    assert loaded.name == case.name
    assert allclose_tree(loaded.inputs, case.inputs)
    assert allclose_tree(loaded.expected, case.expected)
    assert allclose_tree(loaded.taps, case.taps)
    assert list(loaded.taps.keys()) == list(case.taps.keys())   # forward order preserved


def test_loaded_fixture_runs_through_assert_parity_and_bisect(tmp_path):
    case = _synthetic_case()
    p = tmp_path / "case.npz"
    save_case(case, p)
    loaded = load_case(p)
    # identical reload -> parity holds, bisect finds no drift
    assert_parity(loaded.expected, case.expected, name=loaded.name)
    assert bisect(loaded.taps, case.taps) is None
    # perturb one deep tap -> bisect localizes it
    drifted = dict(loaded.taps)
    drifted["block_01"] = drifted["block_01"] + 1.0
    assert bisect(case.taps, drifted) == "block_01"


def test_save_case_rejects_non_dict(tmp_path):
    bad = ParityCase(name="t", inputs=1, expected=2)
    with pytest.raises(TypeError):
        save_case(bad, tmp_path / "x.npz")


def test_dinov3_fixed_input_deterministic():
    a = dinov3_fixed_input()
    b = dinov3_fixed_input()
    assert a.shape == (1, 3, 64, 64) and a.dtype == np.float32
    assert np.array_equal(a, b)                                  # reproducible
    assert not np.array_equal(a, dinov3_fixed_input(seed=1))     # seed matters
    # grid is 64/16 = 4 -> 16 patch tokens
    assert DINOV3_VARIANT["img_size"] // DINOV3_VARIANT["patch_size"] == 4


def test_dinov3_tap_order_schema():
    taps = dinov3_tap_order()
    assert taps[:2] == ["patch_embed", "rope_sincos"]
    assert taps[-4:] == ["norm", "cls", "storage", "patch"]
    block_taps = [t for t in taps if t.startswith("block_")]
    assert len(block_taps) == DINOV3_VARIANT["depth"] == 12
    assert block_taps[0] == "block_00" and block_taps[-1] == "block_11"
