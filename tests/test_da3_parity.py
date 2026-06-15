import pathlib
import sys

import numpy as np
import pytest

from mlx_cv.parity import assert_parity, bisect, da3_monocular_tap_order, load_case

mx = pytest.importorskip("mlx.core")

from mlx_cv.models.depth_anything_v3 import (  # noqa: E402
    DA3MonocularConfig,
    DA3Processor,
    DepthAnythingV3Monocular,
    load_da3_monocular_weights,
)
from mlx_cv.core import SpatialTransform  # noqa: E402

_FIX = pathlib.Path(__file__).parent / "fixtures"
_ATOL = 1e-4


def _run_parity():
    with mx.stream(mx.cpu):
        case = load_case(_FIX / "da3_monocular_tiny_fixture.npz")
        model = DepthAnythingV3Monocular(DA3MonocularConfig.tiny_fixture())
        load_da3_monocular_weights(model, _FIX / "da3_monocular_tiny_fixture_weights.npz")
        raw = model(mx.array(case.inputs["x"]), capture_taps=True)
        result = DA3Processor().postprocess(raw, SpatialTransform.identity((28, 28)))
        got = {
            "depth": result.depth.depth,
            "depth_conf": result.depth.depth_conf,
        }
        taps = {k: np.array(v) for k, v in raw["taps"].items()}
    return case, got, taps


def test_da3_monocular_depth_and_confidence_parity():
    case, got, _ = _run_parity()
    assert_parity(got, case.expected, atol=_ATOL, name=case.name)


def test_da3_monocular_taps_match_schema_and_bisect_clean():
    case, _, taps = _run_parity()
    assert list(taps.keys()) == da3_monocular_tap_order()
    assert bisect(case.taps, taps, atol=_ATOL) is None


def test_da3_monocular_bisect_localizes_injected_drift():
    case, _, taps = _run_parity()
    corrupted = dict(taps)
    corrupted["dpt.output_logits"] = corrupted["dpt.output_logits"] + 1.0
    assert bisect(case.taps, corrupted, atol=_ATOL) == "dpt.output_logits"


def test_da3_dependency_guards():
    pyproject = pathlib.Path("pyproject.toml").read_text()
    assert "torch" not in pyproject
    assert "transformers" not in pyproject
    code = ("import sys, mlx_cv.core; "
            "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)")
    import subprocess

    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
