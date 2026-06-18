import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv import MODELS
from mlx_cv.models.depth_anything_v3 import DA3MonocularConfig, DepthAnythingV3Monocular
import mlx_cv.models.depth_anything_v3 as _da3  # noqa: F401  (import self-registers)


def test_da3_monocular_registered():
    assert "depth-anything-v3-monocular" in MODELS


def test_da3_monocular_tiny_forward_returns_depth_and_confidence():
    model = DepthAnythingV3Monocular(DA3MonocularConfig.tiny_fixture())
    x = mx.array(np.zeros((1, 3, 28, 28), dtype=np.float32))
    out = model(x, capture_taps=True)
    mx.eval(out["depth"], out["depth_conf"])

    assert out["depth"].shape == (1, 28, 28)
    assert out["depth_conf"].shape == (1, 28, 28)
    assert "dinov2.patch_embed" in out["taps"]
    assert "dpt.output_logits" in out["taps"]


def test_da3_monocular_rejects_non_nchw_input():
    model = DepthAnythingV3Monocular(DA3MonocularConfig.tiny_fixture())
    with pytest.raises(ValueError, match="NCHW"):
        model(mx.zeros((1, 28, 28, 3)))
