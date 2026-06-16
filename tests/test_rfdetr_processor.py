import numpy as np
import pytest
import mlx.core as mx

from mlx_cv.core.features import HeadOutput
from mlx_cv.core.geometry import SpatialTransform
from mlx_cv.models.rfdetr import RFDETRProcessor, RFDETRProcessorConfig


def test_rfdetr_processor_preprocess_resizes_normalizes_and_records_transform():
    processor = RFDETRProcessor(
        RFDETRProcessorConfig(
            image_size=(4, 8),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
    )
    image = np.zeros((2, 4, 3), dtype=np.uint8)

    inputs, ctx = processor.preprocess(image)

    assert set(inputs) == {"pixel_values"}
    assert inputs["pixel_values"].shape == (1, 3, 4, 8)
    np.testing.assert_allclose(np.array(inputs["pixel_values"]), -1.0)
    assert ctx.image_size == (2, 4)
    assert ctx.model_size == (4, 8)
    assert ctx.transform == SpatialTransform.resize((2, 4), (4, 8))


def test_rfdetr_processor_preprocess_requires_image():
    processor = RFDETRProcessor(RFDETRProcessorConfig(image_size=(4, 8)))
    with pytest.raises(ValueError, match="requires an image"):
        processor.preprocess({"prompt": "ignored"})


def test_rfdetr_processor_postprocess_matches_reference_topk_box_scaling():
    processor = RFDETRProcessor(
        RFDETRProcessorConfig(
            image_size=(10, 20),
            top_k=2,
            score_threshold=0.5,
            labels=("cat", "dog"),
        )
    )
    ctx = SpatialTransform.resize((5, 10), (10, 20))
    logits = np.array(
        [
            [
                [-10.0, 10.0],
                [8.0, -10.0],
            ]
        ],
        dtype=np.float32,
    )
    boxes = np.array(
        [
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.25, 0.5, 0.2, 0.4],
            ]
        ],
        dtype=np.float32,
    )

    result = processor.postprocess(HeadOutput({"logits": mx.array(logits), "boxes": mx.array(boxes)}), ctx)

    assert result.image_size == (5, 10)
    assert result.detections is not None
    np.testing.assert_allclose(
        result.detections.boxes,
        np.array(
            [
                [2.5, 1.25, 7.5, 3.75],
                [1.5, 1.5, 3.5, 3.5],
            ]
        ),
    )
    np.testing.assert_allclose(result.detections.scores, [1 / (1 + np.exp(-10)), 1 / (1 + np.exp(-8))])
    np.testing.assert_array_equal(result.detections.class_ids, [1, 0])
    assert result.detections.labels == ["dog", "cat"]


def test_rfdetr_processor_postprocess_accepts_reference_output_keys():
    processor = RFDETRProcessor(RFDETRProcessorConfig(image_size=(10, 20), top_k=1))
    ctx = SpatialTransform.resize((5, 10), (10, 20))

    result = processor.postprocess(
        {
            "pred_logits": np.array([[[2.0]]], dtype=np.float32),
            "pred_boxes": np.array([[[0.5, 0.5, 1.0, 1.0]]], dtype=np.float32),
        },
        ctx,
    )

    assert len(result.detections) == 1
    np.testing.assert_allclose(result.detections.boxes, [[0.0, 0.0, 10.0, 5.0]])


def test_rfdetr_processor_postprocess_rejects_batch_results():
    processor = RFDETRProcessor(RFDETRProcessorConfig(image_size=(10, 20), top_k=1))
    ctx = SpatialTransform.resize((5, 10), (10, 20))
    with pytest.raises(ValueError, match="one image"):
        processor.postprocess(
            {
                "pred_logits": np.zeros((2, 1, 1), dtype=np.float32),
                "pred_boxes": np.zeros((2, 1, 4), dtype=np.float32),
            },
            ctx,
        )
