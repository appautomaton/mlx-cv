"""Slice 6 — headline: MLX DINOv3 `forward_features` matches the committed
official-PyTorch golden fixture within tolerance, and `bisect` localizes drift.

The fixture was minted from the official `references/dinov3` on torch-CPU
(`tools/mint_dinov3_fixture.py`). We run the MLX port on the CPU stream so the
comparison is against the same fp32 accumulation order as the oracle; on the GPU
(Metal) stream the math is identical but fp32 reductions drift ~1e-4, which is a
device-accumulation artifact, not a port error.
"""

import pathlib

import numpy as np
import pytest

from mlx_cv.parity import (
    DINOV3_FIXTURE_CONFIG,
    assert_parity,
    bisect,
    dinov3_tap_order,
    load_case,
)

mx = pytest.importorskip("mlx.core")
import mlx_cv.backbones.vision.dinov3 as d3   # noqa: E402  (import self-registers)

_FIX = pathlib.Path(__file__).parent / "fixtures"
_ATOL = 1e-4


def _run_parity():
    with mx.stream(mx.cpu):                    # match the torch-CPU oracle
        case = load_case(_FIX / "dinov3_tiny_fixture.npz")
        model = d3.build_dinov3(DINOV3_FIXTURE_CONFIG)
        d3.load_dinov3_weights(model, _FIX / "dinov3_tiny_fixture_weights.npz")
        feats = model.forward_features(mx.array(case.inputs["x"]), capture_taps=True)
        got = {
            "x_norm_clstoken": np.array(feats.cls_token),
            "x_storage_tokens": np.array(feats.storage_tokens),
            "x_norm_patchtokens": np.array(feats.patch_tokens.data),
        }
        taps = {k: np.array(v) for k, v in feats.extras["taps"].items()}
    return case, got, taps


def test_dinov3_forward_parity_headline():
    case, got, _ = _run_parity()
    assert_parity(got, case.expected, atol=_ATOL, name=case.name)


def test_dinov3_taps_match_schema_and_bisect_clean():
    case, _, taps = _run_parity()
    assert list(taps.keys()) == dinov3_tap_order(depth=DINOV3_FIXTURE_CONFIG["depth"])
    assert bisect(case.taps, taps, atol=_ATOL) is None        # no drift, any tap


def test_dinov3_bisect_localizes_injected_drift():
    case, _, taps = _run_parity()
    corrupted = dict(taps)
    corrupted["block_01"] = corrupted["block_01"] + 1.0       # perturb one deep tap
    assert bisect(case.taps, corrupted, atol=_ATOL) == "block_01"
