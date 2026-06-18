import numpy as np

from mlx_cv.models.depth_anything_v3.convert import (
    DA3_MULTIVIEW_DEFAULT_AUX_LAYERNORM_KEYS,
    convert_da3_monocular_state_dict,
    convert_da3_multiview_state_dict,
)


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


def test_da3_multiview_convert_maps_dualdpt_aux_and_camera_groups():
    state = {
        "backbone.pretrained.camera_token": np.ones((1, 2, 8), dtype=np.float32),
        "head.scratch.output_conv1_aux.0.0.weight": np.ones((8, 16, 3, 3), dtype=np.float32),
        "head.scratch.output_conv2_aux.3.5.weight": np.ones((7, 32, 1, 1), dtype=np.float32),
        "cam_enc.pose_branch.fc1.weight": np.ones((4, 9), dtype=np.float32),
        "cam_dec.fc_t.weight": np.ones((3, 16), dtype=np.float32),
    }
    out = dict(convert_da3_multiview_state_dict(state))

    assert out["backbone.camera_token"].shape == (1, 2, 8)
    assert out["head.scratch.output_conv1_aux.0.0.weight"].shape == (8, 3, 3, 16)
    assert out["head.scratch.output_conv2_aux.3.5.weight"].shape == (7, 1, 1, 32)
    assert out["cam_enc.pose_branch.fc1.weight"].shape == (4, 9)
    assert out["cam_dec.fc_t.weight"].shape == (3, 16)
    assert set(DA3_MULTIVIEW_DEFAULT_AUX_LAYERNORM_KEYS).issubset(out)
    assert np.array_equal(np.array(out["head.scratch.output_conv2_aux.1.2.weight"]), np.ones((32,), dtype=np.float32))
    assert np.array_equal(np.array(out["head.scratch.output_conv2_aux.3.2.bias"]), np.zeros((32,), dtype=np.float32))
