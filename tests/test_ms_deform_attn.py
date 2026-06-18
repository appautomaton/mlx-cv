import numpy as np
import pytest
import mlx.core as mx

from mlx_cv.ops import ms_deform_attn_core
from mlx_cv.parity.fixtures import rfdetr_ms_deform_attn_fixed_inputs


def test_ms_deform_attn_core_matches_tiny_reference_fixture():
    case = rfdetr_ms_deform_attn_fixed_inputs()
    got = ms_deform_attn_core(
        mx.array(case["value"]),
        case["value_spatial_shapes"],
        mx.array(case["sampling_locations"]),
        mx.array(case["attention_weights"]),
    )
    assert got.shape == case["expected"].shape
    assert np.allclose(np.array(got), case["expected"], rtol=1e-6, atol=1e-6)


def test_ms_deform_attn_core_rejects_shape_mismatches():
    case = rfdetr_ms_deform_attn_fixed_inputs()
    with pytest.raises(ValueError, match="value spatial size"):
        ms_deform_attn_core(
            mx.array(case["value"][..., :-1]),
            case["value_spatial_shapes"],
            mx.array(case["sampling_locations"]),
            mx.array(case["attention_weights"]),
        )
    with pytest.raises(ValueError, match="attention_weights"):
        ms_deform_attn_core(
            mx.array(case["value"]),
            case["value_spatial_shapes"],
            mx.array(case["sampling_locations"]),
            mx.zeros((1, 2, 2, 2)),
        )
