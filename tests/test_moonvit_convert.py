import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_cv.backbones.vision.moonvit.config import MoonViTConfig
from mlx_cv.backbones.vision.moonvit.convert import (
    convert_moonvit_state_dict,
    load_moonvit_weights,
)
from mlx_cv.backbones.vision.moonvit.modeling import MoonViTBackbone
from mlx_cv.parity import MOONVIT_FIXTURE_CONFIG

WEIGHTS = "tests/fixtures/moonvit_tiny_fixture_weights.npz"


def _tiny_config() -> MoonViTConfig:
    return MoonViTConfig.from_dict(MOONVIT_FIXTURE_CONFIG)


def test_moonvit_convert_rules_transpose_conv_and_strip_encoder_namespace():
    state = {
        "patch_embed.proj.weight": np.ones((8, 3, 2, 2), dtype=np.float32),
        "patch_embed.proj.bias": np.ones((8,), dtype=np.float32),
        "encoder.blocks.0.wqkv.weight": np.ones((24, 8), dtype=np.float32),
        "encoder.blocks.0.wo.bias": np.ones((8,), dtype=np.float32),
        "encoder.final_layernorm.weight": np.ones((8,), dtype=np.float32),
        "__versions_json__": np.asarray("{}"),
    }

    out = dict(convert_moonvit_state_dict(state))

    assert out["patch_embed.proj.weight"].shape == (8, 2, 2, 3)
    assert "patch_embed.proj.bias" in out
    assert "blocks.0.wqkv.weight" in out
    assert "blocks.0.wo.bias" in out
    assert "final_layernorm.weight" in out
    assert "__versions_json__" not in out
    assert "encoder.blocks.0.wqkv.weight" not in out


def test_load_moonvit_weights_populates_expected_parameter_tree():
    model = load_moonvit_weights(MoonViTBackbone(_tiny_config()), WEIGHTS)
    mx.eval(model.parameters())
    params = dict(tree_flatten(model.parameters()))

    assert params["patch_embed.proj.weight"].shape == (8, 2, 2, 3)
    assert params["blocks.0.wqkv.weight"].shape == (24, 8)
    assert params["blocks.1.wo.bias"].shape == (8,)
    assert params["final_layernorm.weight"].shape == (8,)
