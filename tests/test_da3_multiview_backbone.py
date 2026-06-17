import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from mlx_cv.backbones.vision.dinov2 import (  # noqa: E402
    DA3AnyViewDINOv2,
    DA3AnyViewDINOv2Config,
    DA3Attention,
)
from mlx_cv.core.features import Layout  # noqa: E402


def _tiny_cfg(**overrides) -> DA3AnyViewDINOv2Config:
    values = dict(
        embed_dim=8,
        depth=4,
        num_heads=2,
        patch_size=2,
        pretrain_grid=2,
        out_layers=(1, 3),
        alt_start=2,
        qknorm_start=2,
        rope_start=2,
        cat_token=True,
    )
    values.update(overrides)
    return DA3AnyViewDINOv2Config(**values)


def test_da3_attention_qk_norm_is_opt_in_per_head():
    plain = DA3Attention(8, 2, qk_norm=False)
    normed = DA3Attention(8, 2, qk_norm=True)

    plain_params = dict(tree_flatten(plain.parameters()))
    normed_params = dict(tree_flatten(normed.parameters()))

    assert not any(key.startswith("q_norm.") or key.startswith("k_norm.") for key in plain_params)
    assert normed_params["q_norm.weight"].shape == (4,)
    assert normed_params["q_norm.bias"].shape == (4,)
    assert normed_params["k_norm.weight"].shape == (4,)
    assert normed_params["k_norm.bias"].shape == (4,)


def test_da3_anyview_config_accepts_real_net_contract_shape():
    cfg = DA3AnyViewDINOv2Config.from_dict(
        {
            "name": "vits",
            "out_layers": [5, 7, 9, 11],
            "alt_start": 4,
            "qknorm_start": 4,
            "rope_start": 4,
            "cat_token": True,
        }
    )

    assert cfg.embed_dim == 384
    assert cfg.depth == 12
    assert cfg.num_heads == 6
    assert cfg.head_input_dim == 768
    assert cfg.out_layers == (5, 7, 9, 11)


def test_da3_anyview_block_admission_is_layer_conditioned():
    model = DA3AnyViewDINOv2(_tiny_cfg())
    params = dict(tree_flatten(model.parameters()))

    assert "blocks.1.attn.q_norm.weight" not in params
    assert params["blocks.2.attn.q_norm.weight"].shape == (4,)
    assert model.blocks[1].attn.rope_frequency is None
    assert model.blocks[2].attn.rope_frequency == 100.0


def test_da3_qk_norm_eps_matches_upstream_default_separate_from_block_norm():
    model = DA3AnyViewDINOv2(_tiny_cfg(layer_norm_eps=1e-6, qknorm_start=2))
    block = model.blocks[2]

    assert block.norm1.eps == 1e-6
    assert block.norm2.eps == 1e-6
    assert block.attn.q_norm.eps == 1e-5
    assert block.attn.k_norm.eps == 1e-5


def test_da3_anyview_accepts_bvchw_and_preserves_bvnc_layout():
    model = DA3AnyViewDINOv2(_tiny_cfg())
    x = mx.zeros((1, 2, 3, 4, 4))

    feats = model.forward_features(x)
    mx.eval(feats.patch_tokens.data)

    assert feats.patch_tokens.data.shape == (1, 2, 4, 16)
    assert feats.patch_tokens.layout == Layout.BSNC
    assert feats.patch_tokens.view_axis == 1
    assert feats.patch_tokens.grid == (2, 2)
    assert [fm.data.shape for fm in feats.intermediates] == [(1, 2, 4, 16), (1, 2, 4, 16)]
    assert feats.extras["anyview_dinov2"]["cat_token"] is True


def test_da3_anyview_alternates_local_and_global_attention_from_alt_start():
    model = DA3AnyViewDINOv2(_tiny_cfg(depth=5, out_layers=(4,)))
    feats = model.forward_features(mx.zeros((1, 2, 3, 4, 4)), capture_taps=True)

    assert feats.extras["attention_modes"] == ("local", "local", "local", "global", "local")
    assert feats.extras["attention_input_shapes"] == (
        (2, 5, 8),
        (2, 5, 8),
        (2, 5, 8),
        (1, 10, 8),
        (2, 5, 8),
    )
    assert feats.extras["taps"]["attention_modes"] == feats.extras["attention_modes"]


def test_da3_anyview_selects_reference_reorders_restores_and_injects_camera_tokens():
    model = DA3AnyViewDINOv2(_tiny_cfg(depth=3, out_layers=(2,), alt_start=1, qknorm_start=1, rope_start=1))
    model.camera_token = mx.array(
        np.array(
            [
                [
                    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                ]
            ],
            dtype=np.float32,
        )
    )

    feats = model.forward_features(
        mx.zeros((1, 3, 3, 4, 4)),
        capture_taps=True,
        reference_view_strategy="middle",
    )
    taps = feats.extras["taps"]
    mx.eval(
        taps["reference_indices"],
        taps["view_reorder"],
        taps["view_restore"],
        taps["tokens_after_camera_token"],
    )

    assert np.array_equal(np.array(taps["reference_indices"]), np.array([1], dtype=np.int32))
    assert np.array_equal(np.array(taps["view_reorder"]), np.array([[1, 0, 2]], dtype=np.int32))
    assert np.array_equal(np.array(taps["view_restore"]), np.array([[1, 0, 2]], dtype=np.int32))
    camera_slots = np.array(taps["tokens_after_camera_token"])[0, :, 0]
    assert np.array_equal(camera_slots[0], np.full((8,), 10.0, dtype=np.float32))
    assert np.array_equal(camera_slots[1], np.full((8,), 20.0, dtype=np.float32))
    assert np.array_equal(camera_slots[2], np.full((8,), 20.0, dtype=np.float32))


def test_da3_cat_token_split_normalization_matches_upstream_contract():
    model = DA3AnyViewDINOv2(_tiny_cfg(out_layers=(3,)))
    feats = model.forward_features(mx.zeros((1, 2, 3, 4, 4)), capture_taps=True)

    prenorm = feats.extras["taps"]["selected_prenorm_03"]
    expected = mx.concatenate(
        [
            prenorm[:, :, 1:, : model.embed_dim],
            model.norm(prenorm[..., model.embed_dim :])[:, :, 1:],
        ],
        axis=-1,
    )
    mx.eval(feats.patch_tokens.data, expected)

    assert feats.patch_tokens.data.shape == (1, 2, 4, model.embed_dim * 2)
    assert np.allclose(np.array(feats.patch_tokens.data), np.array(expected), rtol=1e-6, atol=1e-6)


def test_da3_anyview_rejects_non_bvchw_input():
    model = DA3AnyViewDINOv2(_tiny_cfg())
    with pytest.raises(ValueError, match="BVCHW"):
        model.forward_features(mx.zeros((1, 3, 4, 4)))
