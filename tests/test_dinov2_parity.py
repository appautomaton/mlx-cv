"""DA3-style DINOv2 tiny fixture parity."""

import pathlib

import numpy as np
import pytest

from mlx_cv.parity import (
    DINOV2_DA3_FIXTURE_CONFIG,
    assert_parity,
    bisect,
    dinov2_da3_tap_order,
    load_case,
)

mx = pytest.importorskip("mlx.core")
import mlx_cv.backbones.vision.dinov2 as d2  # noqa: E402

_FIX = pathlib.Path(__file__).parent / "fixtures"
_ATOL = 1e-4


def _run_parity():
    with mx.stream(mx.cpu):
        case = load_case(_FIX / "dinov2_da3_tiny_fixture.npz")
        cfg = d2.DINOv2Config(
            embed_dim=DINOV2_DA3_FIXTURE_CONFIG["embed_dim"],
            depth=DINOV2_DA3_FIXTURE_CONFIG["depth"],
            num_heads=DINOV2_DA3_FIXTURE_CONFIG["num_heads"],
            patch_size=DINOV2_DA3_FIXTURE_CONFIG["patch_size"],
            n_register_tokens=DINOV2_DA3_FIXTURE_CONFIG["n_register_tokens"],
            pretrain_grid=DINOV2_DA3_FIXTURE_CONFIG["pretrain_grid"],
            ffn_ratio=DINOV2_DA3_FIXTURE_CONFIG["ffn_ratio"],
            layer_norm_eps=DINOV2_DA3_FIXTURE_CONFIG["layer_norm_eps"],
            final_norm_eps=DINOV2_DA3_FIXTURE_CONFIG["final_norm_eps"],
        )
        model = d2.DINOv2ViT(cfg)
        d2.load_dinov2_weights(model, _FIX / "dinov2_da3_tiny_fixture_weights.npz")
        feats = model.forward_features(
            mx.array(case.inputs["x"]),
            intermediate_layers=DINOV2_DA3_FIXTURE_CONFIG["intermediate_layers"],
            capture_taps=True,
        )
        got = {
            "x_norm_clstoken": np.array(feats.cls_token),
            "x_norm_patchtokens": np.array(feats.patch_tokens.data),
            "intermediates": np.stack([np.array(fm.data) for fm in feats.intermediates]),
        }
        taps = {k: np.array(v) for k, v in feats.extras["taps"].items()}
    return case, got, taps


def test_dinov2_da3_forward_parity_headline():
    case, got, _ = _run_parity()
    assert_parity(got, case.expected, atol=_ATOL, name=case.name)


def test_dinov2_da3_taps_match_schema_and_bisect_clean():
    case, _, taps = _run_parity()
    assert list(taps.keys()) == dinov2_da3_tap_order()
    assert bisect(case.taps, taps, atol=_ATOL) is None


def test_dinov2_da3_bisect_localizes_injected_drift():
    case, _, taps = _run_parity()
    corrupted = dict(taps)
    corrupted["intermediate_02"] = corrupted["intermediate_02"] + 1.0
    assert bisect(case.taps, corrupted, atol=_ATOL) == "intermediate_02"
