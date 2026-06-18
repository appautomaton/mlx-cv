import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from mlx_cv import MODELS  # noqa: E402
from mlx_cv.models.depth_anything_v3 import (  # noqa: E402
    DA3MultiViewConfig,
    DepthAnythingV3MultiView,
    affine_inverse,
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)
import mlx_cv.models.depth_anything_v3 as _da3  # noqa: F401,E402  (import self-registers)


def _tiny_model() -> DepthAnythingV3MultiView:
    return DepthAnythingV3MultiView(DA3MultiViewConfig.tiny_fixture())


def test_da3_multiview_registered():
    assert "depth-anything-v3-multiview" in MODELS


def test_da3_multiview_config_matches_small_contract_dimensions():
    cfg = DA3MultiViewConfig.small()

    assert cfg.backbone.embed_dim == 384
    assert cfg.backbone.head_input_dim == 768
    assert cfg.head.dim_in == 768
    assert cfg.head.features == 64
    assert cfg.head.out_channels == (48, 96, 192, 384)
    assert cfg.cam_enc.dim_out == 384
    assert cfg.cam_dec.dim_in == 768
    assert cfg.extrinsics_convention == "w2c"


def test_da3_multiview_config_matches_base_dualdpt_checkpoint_dimensions():
    cfg = DA3MultiViewConfig.base()

    assert cfg.backbone.embed_dim == 768
    assert cfg.backbone.head_input_dim == 1536
    assert cfg.head.dim_in == 1536
    assert cfg.head.features == 128
    assert cfg.head.out_channels == (96, 192, 384, 768)
    assert cfg.cam_enc.dim_out == 768
    assert cfg.cam_dec.dim_in == 1536


def test_da3_camera_encoder_trunk_uses_upstream_layernorm_eps():
    model = _tiny_model()

    assert model.cam_enc.trunk[0].norm1.eps == 1e-5
    assert model.cam_enc.trunk[0].norm2.eps == 1e-5
    assert model.backbone.blocks[0].norm1.eps == 1e-6


def test_da3_multiview_parameter_tree_exposes_dualdpt_and_camera_groups():
    params = dict(tree_flatten(_tiny_model().parameters()))

    assert "head.scratch.output_conv2_aux.3.5.weight" in params
    assert params["head.scratch.output_conv2_aux.3.5.weight"].shape == (7, 1, 1, 32)
    assert "head.scratch.output_conv2_aux.0.2.weight" in params
    assert params["head.scratch.output_conv2_aux.3.2.weight"].shape == (32,)
    assert params["head.scratch.output_conv2_aux.3.2.bias"].shape == (32,)
    assert "cam_enc.pose_branch.fc1.weight" in params
    assert "cam_enc.trunk.0.attn.qkv.weight" in params
    assert "cam_dec.fc_fov.0.weight" in params


def test_da3_multiview_tiny_forward_returns_depth_ray_and_camera_w2c():
    model = _tiny_model()
    x = mx.array(np.zeros((1, 3, 3, 4, 4), dtype=np.float32))
    out = model(x, capture_taps=True, reference_view_strategy="middle")
    mx.eval(
        out["depth"],
        out["depth_conf"],
        out["ray"],
        out["ray_conf"],
        out["pose_encoding"],
        out["extrinsics"],
        out["intrinsics"],
    )

    assert out["depth"].shape == (1, 3, 4, 4)
    assert out["depth_conf"].shape == (1, 3, 4, 4)
    assert out["ray"].shape[:2] == (1, 3)
    assert out["ray"].shape[-1] == 6
    assert out["ray_conf"].shape[:2] == (1, 3)
    assert out["pose_encoding"].shape == (1, 3, 9)
    assert out["extrinsics"].shape == (1, 3, 3, 4)
    assert out["intrinsics"].shape == (1, 3, 3, 3)
    assert out["extrinsics_convention"] == "w2c"
    assert "dualdpt.main_logits" in out["taps"]
    assert "camera_dec.pose_encoding" in out["taps"]
    assert "camera_dec.extrinsics_w2c" in out["taps"]


def test_da3_pose_utilities_use_scalar_last_quaternion_and_affine_inverse():
    c2w_np = np.eye(4, dtype=np.float32).reshape(1, 1, 4, 4)
    c2w_np[0, 0, :3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    intr_np = np.array(
        [[[[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]]],
        dtype=np.float32,
    )
    pose = extri_intri_to_pose_encoding(mx.array(c2w_np), mx.array(intr_np), (4, 4))
    decoded_c2w, decoded_intr = pose_encoding_to_extri_intri(pose, (4, 4))
    w2c = affine_inverse(decoded_c2w)
    mx.eval(pose, decoded_intr, w2c)

    assert np.allclose(np.array(pose)[0, 0, 3:7], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(np.array(decoded_intr), intr_np, atol=1e-5)
    assert np.allclose(np.array(w2c)[0, 0, :3, 3], [-1.0, -2.0, -3.0], atol=1e-5)


def test_da3_pose_conditioned_input_validates_shapes_and_emits_camera_tokens():
    model = _tiny_model()
    x = mx.array(np.zeros((1, 2, 3, 4, 4), dtype=np.float32))
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, None], 2, axis=1)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, None], 2, axis=1)

    out = model(x, extrinsics=mx.array(extrinsics), intrinsics=mx.array(intrinsics), capture_taps=True)
    mx.eval(out["taps"]["camera_enc.tokens"], out["taps"]["camera_enc.pose_encoding"])

    assert out["taps"]["camera_enc.tokens"].shape == (1, 2, 8)
    assert out["taps"]["camera_enc.pose_encoding"].shape == (1, 2, 9)
    with pytest.raises(ValueError, match="requires both extrinsics and intrinsics"):
        model(x, extrinsics=mx.array(extrinsics))

    bad_intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, None], 3, axis=1)
    with pytest.raises(ValueError, match="intrinsics must share input B,V axes"):
        model(x, extrinsics=mx.array(extrinsics), intrinsics=mx.array(bad_intrinsics))
