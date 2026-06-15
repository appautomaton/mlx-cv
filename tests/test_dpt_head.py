import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv.core import BackboneFeatures, FeatureMap, HeadInput, Layout
from mlx_cv.heads.dense import DPTConfig, DPTHead
from mlx_cv.heads.dense.dpt import resize_bilinear_align_corners


def _features(dim: int = 32, grid: tuple[int, int] = (2, 2)) -> BackboneFeatures:
    rng = np.random.default_rng(0)
    inter = [
        FeatureMap(
            mx.array(rng.standard_normal((1, grid[0] * grid[1], dim)).astype(np.float32)),
            layout=Layout.BNC,
            grid=grid,
            stride=14,
        )
        for _ in range(4)
    ]
    return BackboneFeatures(patch_tokens=inter[-1], intermediates=inter)


def test_dpt_head_output_dim_two_returns_depth_and_confidence():
    cfg = DPTConfig(
        dim_in=32,
        output_dim=2,
        features=16,
        out_channels=(8, 8, 8, 8),
        use_sky_head=False,
        pos_embed=False,
        down_ratio=1,
        norm_type="idt",
    )
    head = DPTHead(cfg)
    out = head(HeadInput(features=_features(), image_size=(28, 28)), capture_taps=True)
    mx.eval(out["depth"], out["depth_conf"])

    assert out["depth"].shape == (1, 28, 28)
    assert out["depth_conf"].shape == (1, 28, 28)
    assert "output_logits" in out["taps"]


def test_dpt_head_output_dim_one_omits_confidence():
    head = DPTHead(DPTConfig(dim_in=32, output_dim=1, features=16, out_channels=(8, 8, 8, 8)))
    out = head(HeadInput(features=_features(), image_size=(28, 28)))
    assert "depth" in out
    assert "depth_conf" not in out


def test_dpt_requires_four_intermediates():
    head = DPTHead(DPTConfig(dim_in=32, output_dim=1, features=16, out_channels=(8, 8, 8, 8)))
    feats = _features()
    feats.intermediates = feats.intermediates[:3]
    with pytest.raises(ValueError, match="four intermediates"):
        head(HeadInput(features=feats, image_size=(28, 28)))


def test_dpt_rejects_unimplemented_sky_and_pos_embed_branches():
    with pytest.raises(NotImplementedError, match="use_sky_head=False"):
        DPTHead(DPTConfig(dim_in=32, use_sky_head=True))
    with pytest.raises(NotImplementedError, match="pos_embed=False"):
        DPTHead(DPTConfig(dim_in=32, pos_embed=True))


def test_bilinear_resize_align_corners_matches_hand_computed_grid():
    x = mx.array(np.array([[[[0.0], [2.0]], [[4.0], [6.0]]]], dtype=np.float32))
    y = resize_bilinear_align_corners(x, (3, 3))
    expected = np.array([[[[0.0], [1.0], [2.0]],
                          [[2.0], [3.0], [4.0]],
                          [[4.0], [5.0], [6.0]]]], dtype=np.float32)
    assert np.allclose(np.array(y), expected)
