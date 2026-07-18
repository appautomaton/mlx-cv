from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mlx_cv.models.sam3.real_config import (
    Sam3DETRDecoderConfig,
    Sam3DETREncoderConfig,
    Sam3DetectorConfig,
    Sam3GeometryEncoderConfig,
    Sam3MaskDecoderConfig,
    Sam3TextConfig,
    Sam3ViTConfig,
    Sam3VisionConfig,
)
from mlx_cv.models.sam3.real_modeling import Sam3Model as LegacyFaithfulModel
from mlx_cv.models.sam3.sam31_convert import map_sam31_detector_key
from mlx_cv.models.sam3.sam31_modeling import SAM3Model
from mlx_cv.models.sam3.sam31_predictor import SAM31ImagePredictor
from mlx_cv.models.sam3.sam31_modeling import SAM3ImageOutput
from mlx_cv.models.sam3.tokenizer import SAM3Tokenizer


def _tiny_config() -> Sam3DetectorConfig:
    return Sam3DetectorConfig(
        vision=Sam3VisionConfig(
            backbone=Sam3ViTConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                image_size=28,
                patch_size=14,
                window_size=2,
                global_attn_indexes=(1,),
                pretrain_image_size=28,
            ),
            fpn_hidden_size=32,
            backbone_feature_sizes=((8, 8), (4, 4), (2, 2)),
            scale_factors=(4.0, 2.0, 1.0),
        ),
        text=Sam3TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            projection_dim=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=8,
        ),
        geometry_encoder=Sam3GeometryEncoderConfig(
            hidden_size=32,
            num_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            roi_size=3,
        ),
        detr_encoder=Sam3DETREncoderConfig(
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
        ),
        detr_decoder=Sam3DETRDecoderConfig(
            hidden_size=32,
            num_layers=2,
            num_queries=4,
            num_attention_heads=4,
            intermediate_size=64,
        ),
        mask_decoder=Sam3MaskDecoderConfig(
            hidden_size=32,
            num_upsampling_stages=3,
            num_attention_heads=4,
        ),
    )


def test_sam31_detector_source_mapping_is_complete_for_real_checkpoint():
    try:
        import torch
    except ModuleNotFoundError:
        return

    checkpoint = "models/sam3-video/upstream/sam3.1_multiplex.pt"
    try:
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
    except FileNotFoundError:
        return
    if "model" in state:
        state = state["model"]

    target_keys = [
        target
        for source in state
        if source.startswith("detector.")
        for target in map_sam31_detector_key(source)
    ]

    assert sum(key.startswith("detector.") for key in state) == 1166
    assert len(target_keys) == 1506
    assert len(set(target_keys)) == 1506


def test_sam31_model_parameter_tree_matches_converted_real_checkpoint_shapes():
    try:
        import torch
    except ModuleNotFoundError:
        return

    checkpoint = "models/sam3-video/upstream/sam3.1_multiplex.pt"
    try:
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
    except FileNotFoundError:
        return
    if "model" in state:
        state = state["model"]

    expected: dict[str, tuple[int, ...]] = {}
    for source, value in state.items():
        if not source.startswith("detector."):
            continue
        targets = map_sam31_detector_key(source)
        if not targets:
            continue
        shape = list(value.shape)
        if len(targets) == 3:
            shape[0] //= 3
        for target in targets:
            converted = tuple(shape)
            if source.endswith("trunk.pos_embed"):
                converted = (converted[0], converted[1] - 1, converted[2])
            if source.endswith("encoder.text_projection"):
                converted = (converted[1], converted[0])
            if len(converted) == 4 and target.endswith(".weight"):
                converted = (
                    (converted[1], converted[2], converted[3], converted[0])
                    if ".scale_layers." in target
                    else (converted[0], converted[2], converted[3], converted[1])
                )
            expected[target] = converted

    actual = {key: tuple(value.shape) for key, value in tree_flatten(SAM3Model().parameters())}
    assert actual == expected


def test_sam31_tiny_image_forward_uses_shared_detector_head():
    config = _tiny_config()
    model = SAM3Model(config)
    pixels = mx.arange(1 * 3 * 28 * 28, dtype=mx.float32).reshape(1, 3, 28, 28) / 255.0
    input_ids = mx.array([[1, 7, 3, 0]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1, 0]], dtype=mx.int32)

    output = model(pixels, input_ids, attention_mask)
    mx.eval(output)

    assert output.pred_logits.shape == (1, 4)
    assert output.pred_boxes.shape == (1, 4, 4)
    assert output.presence_logits.shape == (1, 1)
    assert output.pred_masks.shape == (1, 4, 8, 8)
    assert output.semantic_seg.shape == (1, 1, 8, 8)
    assert output.vision_last_hidden_state.shape == (1, 4, 32)


def test_sam31_tiny_detector_preserves_shared_vision_math_with_official_prompt_path():
    config = _tiny_config()
    sam31 = SAM3Model(config)
    # The verified SAM 3.0 wrapper scalps the final 0.5x level from a four-level
    # neck. SAM 3.1 stores the equivalent detector pyramid directly as 4x/2x/1x.
    legacy_config = replace(
        config,
        vision=replace(
            config.vision,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            backbone_feature_sizes=((8, 8), (4, 4), (2, 2), (1, 1)),
        ),
    )
    legacy = LegacyFaithfulModel(legacy_config)
    sam31_params = dict(tree_flatten(sam31.parameters()))
    legacy_params = dict(tree_flatten(legacy.parameters()))
    common = [(key, sam31_params[key]) for key in legacy_params if key in sam31_params]
    legacy.update(tree_unflatten(common))

    pixels = mx.arange(1 * 3 * 28 * 28, dtype=mx.float32).reshape(1, 3, 28, 28) / 255.0
    input_ids = mx.array([[1, 7, 3, 0]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1, 0]], dtype=mx.int32)
    actual = sam31(pixels, input_ids, attention_mask)
    expected = legacy(pixels, input_ids, attention_mask)
    mx.eval(actual, expected)

    # SAM 3.1 always appends an encoded empty-geometry CLS token, so its DETR
    # outputs intentionally differ from the old text-only wrapper. The shared
    # TriHead vision path itself remains numerically identical.
    np.testing.assert_allclose(
        np.asarray(actual.vision_last_hidden_state),
        np.asarray(expected.vision_last_hidden_state),
        atol=0.0,
        rtol=0.0,
    )
    assert np.isfinite(np.asarray(actual.pred_logits)).all()
    assert np.isfinite(np.asarray(actual.pred_boxes)).all()
    assert ((np.asarray(actual.pred_boxes) >= 0.0) & (np.asarray(actual.pred_boxes) <= 1.0)).all()


def test_sam31_tokenizer_matches_official_clip_vocabulary():
    path = "references/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    tokenizer = SAM3Tokenizer(path, clean="lower")

    tokens = tokenizer("robot")

    np.testing.assert_array_equal(tokens[0, :4], [49406, 8797, 49407, 0])


def test_sam31_image_predictor_returns_public_boxes_scores_and_masks():
    class _FakeModel:
        def __call__(self, pixel_values, input_ids, attention_mask):
            return SAM3ImageOutput(
                pred_logits=mx.array([[4.0, -4.0]]),
                pred_boxes=mx.array([[[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.1, 0.1]]]),
                presence_logits=mx.array([[4.0]]),
                pred_masks=mx.array(
                    [[[[1.0, 1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, -1.0]]]]
                ),
                semantic_seg=mx.zeros((1, 1, 2, 2)),
                vision_last_hidden_state=mx.zeros((1, 1, 1)),
            )

    predictor = SAM31ImagePredictor(
        _FakeModel(),
        bpe_path="references/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        score_threshold=0.5,
    )
    prediction = predictor.predict(np.zeros((20, 40, 3), dtype=np.uint8), "robot")

    assert prediction.query_indices.tolist() == [0]
    np.testing.assert_allclose(prediction.boxes, [[10.0, 5.0, 30.0, 15.0]])
    assert prediction.scores[0] > 0.9
    assert prediction.masks.shape == (1, 20, 40)
    assert prediction.masks.all()
