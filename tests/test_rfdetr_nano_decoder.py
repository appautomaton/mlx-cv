import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import mlx.core as mx

from mlx_cv.backbones.vision.necks import RFDETRFeaturePyramid, RFDETRPyramidLevel
from mlx_cv.core.features import FeatureMap, Layout
from mlx_cv.heads.detection import RFDETRDecoderConfig, RFDETRQueryDecoder
from mlx_cv.heads.detection.rfdetr import (
    RFDETRDecoderLayer,
    _bbox_reparametrize,
    _generate_encoder_output_proposals,
    _topk_indices,
    slice_grouped_queries_for_inference,
)


def _level(data: mx.array) -> RFDETRPyramidLevel:
    _, height, width, _ = data.shape
    return RFDETRPyramidLevel(
        data=data,
        feature=FeatureMap(data, layout=Layout.BHWC, grid=(height, width), stride=16),
        mask=mx.zeros((1, height, width), dtype=mx.bool_),
        position=mx.zeros((1, height, width, 2)),
        stride=16,
    )


def _pyramid(hidden_dim: int = 8) -> RFDETRFeaturePyramid:
    data = mx.array(np.arange(1 * 2 * 2 * hidden_dim, dtype=np.float32).reshape(1, 2, 2, hidden_dim) / 100.0)
    return RFDETRFeaturePyramid([_level(data)])


def _tiny_nano_cfg() -> RFDETRDecoderConfig:
    return RFDETRDecoderConfig(
        hidden_dim=8,
        num_queries=2,
        num_heads=2,
        self_attn_heads=2,
        num_layers=2,
        num_points=2,
        num_classes=3,
        ffn_hidden_dim=16,
        group_detr=3,
        query_dim=4,
        use_self_attention=True,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        decoder_final_norm=True,
    )


def test_nano_decoder_parameter_shapes_match_real_checkpoint_contract():
    cfg = RFDETRDecoderConfig(
        hidden_dim=256,
        num_queries=300,
        num_heads=16,
        self_attn_heads=8,
        num_layers=2,
        num_points=2,
        num_classes=91,
        ffn_hidden_dim=2048,
        group_detr=13,
        query_dim=4,
        use_self_attention=True,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        decoder_final_norm=True,
    )
    decoder = RFDETRQueryDecoder(cfg, num_levels=1)
    params = dict(tree_flatten(decoder.parameters()))

    assert tuple(params["query_embed"].shape) == (3900, 256)
    assert tuple(params["reference_embed"].shape) == (3900, 4)
    assert tuple(params["layers.0.self_attn.query_proj.weight"].shape) == (256, 256)
    assert tuple(params["layers.0.self_attn.out_proj.bias"].shape) == (256,)
    assert tuple(params["layers.1.sampling_offsets.weight"].shape) == (64, 256)
    assert tuple(params["layers.1.attention_weights.weight"].shape) == (32, 256)
    assert tuple(params["layers.0.ffn1.weight"].shape) == (2048, 256)
    assert tuple(params["layers.0.ffn2.weight"].shape) == (256, 2048)
    assert tuple(params["layers.0.norm3.weight"].shape) == (256,)
    assert tuple(params["norm.weight"].shape) == (256,)
    assert tuple(params["ref_point_head.layers.0.weight"].shape) == (256, 512)
    assert tuple(params["ref_point_head.layers.1.weight"].shape) == (256, 256)
    assert tuple(params["enc_output.12.weight"].shape) == (256, 256)
    assert tuple(params["enc_out_class_embed.12.weight"].shape) == (91, 256)
    assert tuple(params["enc_out_bbox_embed.12.layers.2.bias"].shape) == (4,)


def test_encoder_output_proposals_are_deterministic_for_tiny_grid():
    memory = mx.ones((1, 4, 2))
    output_memory, proposals = _generate_encoder_output_proposals(
        memory,
        np.array([[2, 2]], dtype=np.int32),
        unsigmoid=False,
    )
    mx.eval(output_memory, proposals)

    expected = np.array(
        [
            [
                [0.25, 0.25, 0.05, 0.05],
                [0.75, 0.25, 0.05, 0.05],
                [0.25, 0.75, 0.05, 0.05],
                [0.75, 0.75, 0.05, 0.05],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.array(output_memory), np.ones((1, 4, 2)), atol=1e-6)
    np.testing.assert_allclose(np.array(proposals), expected, atol=1e-6)


def test_grouped_query_inference_slice_uses_group_zero_rows():
    labelled = mx.array(
        np.array([[group, query] for group in range(4) for query in range(3)], dtype=np.float32)
    )

    out = slice_grouped_queries_for_inference(labelled, num_queries=3, group_detr=4)

    np.testing.assert_array_equal(np.array(out), np.array([[0, 0], [0, 1], [0, 2]], dtype=np.float32))


def test_two_stage_topk_uses_stable_index_tiebreak_for_near_ties():
    scores = mx.array([[0.1, 0.100006, 0.100007, 0.10005]], dtype=mx.float32)

    out = _topk_indices(scores, 3)
    mx.eval(out)

    np.testing.assert_array_equal(np.array(out), np.array([[3, 2, 1]], dtype=np.int32))


def test_bbox_reparameterization_uses_reference_box_scale():
    delta = mx.array([[[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, np.log(2.0), np.log(0.5)]]])
    reference = mx.array([[[0.5, 0.5, 0.2, 0.4], [0.4, 0.6, 0.1, 0.2]]])

    out = _bbox_reparametrize(delta, reference)
    mx.eval(out)

    expected = np.array([[[0.5, 0.5, 0.2, 0.4], [0.5, 0.4, 0.2, 0.1]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(out), expected, atol=1e-6)


def test_cross_attention_sampling_uses_query_pos(monkeypatch):
    import mlx_cv.heads.detection.rfdetr as module

    captured = []

    def fake_core(value, spatial_shapes, sampling_locations, attention_weights):
        captured.append(sampling_locations)
        return mx.zeros((value.shape[0], sampling_locations.shape[1], value.shape[1] * value.shape[2]))

    monkeypatch.setattr(module, "ms_deform_attn_core", fake_core)
    cfg = RFDETRDecoderConfig(
        hidden_dim=4,
        num_queries=1,
        num_heads=1,
        num_layers=1,
        num_points=1,
        num_classes=2,
    )
    layer = RFDETRDecoderLayer(cfg, num_levels=1)
    layer.update(
        tree_unflatten(
            [
                ("sampling_offsets.weight", mx.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.float32)),
                ("sampling_offsets.bias", mx.zeros((2,), dtype=mx.float32)),
                ("attention_weights.weight", mx.zeros((1, 4), dtype=mx.float32)),
                ("attention_weights.bias", mx.zeros((1,), dtype=mx.float32)),
            ]
        )
    )

    out = layer(
        mx.zeros((1, 1, 4)),
        mx.zeros((1, 1, 4)),
        np.array([[1, 1]], dtype=np.int32),
        mx.zeros((1, 1, 1, 2)),
        query_pos=mx.array([[[0.25, 0.5, 0.0, 0.0]]], dtype=mx.float32),
    )
    mx.eval(out, captured[0])

    np.testing.assert_allclose(np.array(captured[0])[0, 0, 0, 0, 0], [0.25, 0.5], atol=1e-6)


def test_decoder_final_norm_applies_to_all_intermediate_states(monkeypatch):
    import mlx_cv.heads.detection.rfdetr as module

    def fake_core(value, spatial_shapes, sampling_locations, attention_weights):
        return mx.zeros((value.shape[0], sampling_locations.shape[1], value.shape[1] * value.shape[2]))

    monkeypatch.setattr(module, "ms_deform_attn_core", fake_core)
    decoder = RFDETRQueryDecoder(_tiny_nano_cfg(), num_levels=1)
    decoder.update(
        tree_unflatten(
            [
                ("norm.weight", mx.ones((8,), dtype=mx.float32) * 2.0),
                ("norm.bias", mx.ones((8,), dtype=mx.float32) * 3.0),
            ]
        )
    )

    out = decoder(_pyramid())
    mx.eval(out["hidden_states"], out["decoder_hidden_states"])

    layer_means = np.array(mx.mean(out["decoder_hidden_states"], axis=-1))
    np.testing.assert_allclose(layer_means, np.full((2, 1, 2), 3.0), atol=1e-5)
    np.testing.assert_allclose(
        np.array(out["hidden_states"]),
        np.array(out["decoder_hidden_states"][-1]),
        atol=1e-6,
    )


def test_two_stage_decoder_outputs_and_taps_are_layer_ordered(monkeypatch):
    import mlx_cv.heads.detection.rfdetr as module

    def fake_core(value, spatial_shapes, sampling_locations, attention_weights):
        return mx.zeros((value.shape[0], sampling_locations.shape[1], value.shape[1] * value.shape[2]))

    monkeypatch.setattr(module, "ms_deform_attn_core", fake_core)
    decoder = RFDETRQueryDecoder(_tiny_nano_cfg(), num_levels=1)

    out = decoder(_pyramid(), capture_taps=True)
    mx.eval(
        out["hidden_states"],
        out["decoder_hidden_states"],
        out["self_attention"],
        out["deformable_attention"],
        out["encoder_hidden_states"],
        out["encoder_boxes"],
    )

    assert out["hidden_states"].shape == (1, 2, 8)
    assert out["decoder_hidden_states"].shape == (2, 1, 2, 8)
    assert out["reference_points"].shape == (1, 2, 1, 4)
    assert out["self_attention"].shape == (2, 1, 2, 8)
    assert out["deformable_attention"].shape == (2, 1, 2, 8)
    assert out["encoder_hidden_states"].shape == (1, 2, 8)
    assert out["encoder_boxes"].shape == (1, 2, 4)
    np.testing.assert_allclose(
        np.array(out["hidden_states"]),
        np.array(out["decoder_hidden_states"][-1]),
        atol=1e-6,
    )
