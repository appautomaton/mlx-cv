import numpy as np

from mlx.utils import tree_flatten

from mlx_cv.heads.dense import DPTConfig, DPTHead
from mlx_cv.heads.dense.convert import convert_dpt_state_dict


def test_dpt_convert_distinguishes_conv_and_conv_transpose_layouts():
    conv = np.arange(8 * 32 * 1 * 1, dtype=np.float32).reshape(8, 32, 1, 1)
    deconv = np.arange(8 * 8 * 4 * 4, dtype=np.float32).reshape(8, 8, 4, 4)
    out = dict(convert_dpt_state_dict({
        "projects.0.weight": conv,
        "resize_layers.0.weight": deconv,
    }))

    assert out["projects.0.weight"].shape == (8, 1, 1, 32)
    assert out["resize_layers.0.weight"].shape == (8, 4, 4, 8)
    assert np.array_equal(np.array(out["projects.0.weight"]), np.transpose(conv, (0, 2, 3, 1)))
    assert np.array_equal(
        np.array(out["resize_layers.0.weight"]),
        np.transpose(deconv, (1, 2, 3, 0)),
    )


def test_dpt_conv_transpose_converted_shape_matches_mlx_module():
    cfg = DPTConfig(dim_in=32, output_dim=2, features=16, out_channels=(8, 8, 8, 8))
    model = DPTHead(cfg)
    params = dict(tree_flatten(model.parameters()))
    source = {
        "resize_layers.0.weight": np.zeros((8, 8, 4, 4), dtype=np.float32),
        "resize_layers.1.weight": np.zeros((8, 8, 2, 2), dtype=np.float32),
    }
    converted = dict(convert_dpt_state_dict(source))

    assert converted["resize_layers.0.weight"].shape == params["resize_layers.0.weight"].shape
    assert converted["resize_layers.1.weight"].shape == params["resize_layers.1.weight"].shape
