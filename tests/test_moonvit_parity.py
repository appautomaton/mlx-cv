import pathlib

import numpy as np
import pytest

from mlx_cv.parity import (
    MOONVIT_FIXTURE_CONFIG,
    assert_parity,
    bisect,
    load_case,
    moonvit_tap_order,
)

mx = pytest.importorskip("mlx.core")

from mlx_cv.backbones.vision.moonvit.config import MoonViTConfig  # noqa: E402
from mlx_cv.backbones.vision.moonvit.convert import load_moonvit_weights  # noqa: E402
from mlx_cv.backbones.vision.moonvit.modeling import MoonViTBackbone  # noqa: E402

_FIX = pathlib.Path(__file__).parent / "fixtures"
_ATOL = 1e-4


def _run_parity():
    with mx.stream(mx.cpu):
        case = load_case(_FIX / "moonvit_tiny_fixture.npz")
        model = MoonViTBackbone(MoonViTConfig.from_dict(MOONVIT_FIXTURE_CONFIG))
        load_moonvit_weights(model, _FIX / "moonvit_tiny_fixture_weights.npz")
        merged, taps = model(
            mx.array(case.inputs["pixel_values"]),
            mx.array(case.inputs["grid_hws"].astype(np.int32)),
            capture_taps=True,
        )
        mx.eval(model.parameters(), *merged, *taps.values())

    got = {"norm": np.array(taps["norm"])}
    for i, item in enumerate(merged):
        got[f"merged_{i:02d}"] = np.array(item)
    got_taps = {key: np.array(value) for key, value in taps.items()}
    return case, got, got_taps


def test_moonvit_loaded_forward_parity_headline():
    case, got, _ = _run_parity()
    assert_parity(got, case.expected, atol=_ATOL, rtol=_ATOL, name=case.name)


def test_moonvit_taps_match_schema_and_bisect_clean():
    case, _, taps = _run_parity()
    assert list(taps) == moonvit_tap_order()
    assert bisect(case.taps, taps, atol=_ATOL, rtol=_ATOL) is None


def test_moonvit_bisect_localizes_injected_drift():
    case, _, taps = _run_parity()
    corrupted = dict(taps)
    corrupted["block_01"] = corrupted["block_01"] + 1.0
    assert bisect(case.taps, corrupted, atol=_ATOL, rtol=_ATOL) == "block_01"
