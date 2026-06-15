import numpy as np

from mlx_cv.models.depth_anything_v3.convert import convert_da3_monocular_state_dict


def test_da3_convert_delegates_backbone_and_head_rules():
    state = {
        "backbone.pretrained.pos_embed": np.ones((1, 5, 32), dtype=np.float32),
        "backbone.pretrained.patch_embed.proj.weight": np.ones((32, 3, 14, 14), dtype=np.float32),
        "head.projects.0.weight": np.ones((8, 32, 1, 1), dtype=np.float32),
        "head.resize_layers.0.weight": np.ones((8, 8, 4, 4), dtype=np.float32),
    }
    out = dict(convert_da3_monocular_state_dict(state))

    assert "backbone.pos_embed.table" in out
    assert out["backbone.patch_embed.proj.weight"].shape == (32, 14, 14, 3)
    assert out["head.projects.0.weight"].shape == (8, 1, 1, 32)
    assert out["head.resize_layers.0.weight"].shape == (8, 4, 4, 8)
