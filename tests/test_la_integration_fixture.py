import numpy as np

from mlx_cv.parity import (
    LOCATEANYTHING_FIXTURE_CONFIG,
    load_case,
    locateanything_fixed_inputs,
    locateanything_tap_order,
)

FIXTURE = "tests/fixtures/locateanything_tiny_fixture.npz"
WEIGHTS = "tests/fixtures/locateanything_tiny_fixture_weights.npz"


def test_locateanything_fixture_schema_and_fixed_inputs():
    inputs = locateanything_fixed_inputs()
    assert inputs["input_ids"].shape == (1, 3)
    assert inputs["cached_image_features"].shape == (1, 32)
    assert inputs["pbd_block_logits"].shape[0] == 6
    assert inputs["generated_ids"].ndim == 1


def test_locateanything_fixture_round_trips_schema():
    case = load_case(FIXTURE)
    assert case.name == LOCATEANYTHING_FIXTURE_CONFIG["name"]
    assert list(case.taps or {}) == locateanything_tap_order()
    assert case.expected["inputs_embeds"].shape == (1, 3, 8)
    assert case.expected["boxes"].shape == (1, 4)
    assert case.expected["points"].shape == (1, 2)
    weights = np.load(WEIGHTS, allow_pickle=False)
    assert "__config_json__" in weights.files
