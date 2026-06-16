import numpy as np
import pytest
import mlx.core as mx

from mlx_cv.core.features import HeadOutput
from mlx_cv.core.geometry import SpatialTransform
from mlx_cv.models.sam3 import SAM3Processor, SAM3ProcessorConfig


def test_sam3_processor_preprocess_resizes_normalizes_and_prepares_prompt():
    processor = SAM3Processor(
        SAM3ProcessorConfig(
            image_size=(4, 8),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
    )
    image = np.zeros((2, 4, 3), dtype=np.uint8)

    inputs, ctx = processor.preprocess({"image": image, "text": "cat", "boxes": [[1, 0, 3, 2]]})

    assert set(inputs) == {"pixel_values", "prompt"}
    assert inputs["pixel_values"].shape == (1, 3, 4, 8)
    np.testing.assert_allclose(np.array(inputs["pixel_values"]), -1.0)
    assert ctx.image_size == (2, 4)
    assert ctx.model_size == (4, 8)
    assert ctx.transform == SpatialTransform.resize((2, 4), (4, 8))
    assert ctx.prompt.texts == ("cat",)
    np.testing.assert_allclose(ctx.prompt.geometry.boxes_cxcywh, [[0.5, 0.5, 0.5, 1.0]])


def test_sam3_processor_preprocess_rejects_missing_image():
    processor = SAM3Processor(SAM3ProcessorConfig(image_size=(4, 8)))
    with pytest.raises(ValueError, match="requires an image"):
        processor.preprocess({"prompt": "ignored"})


def test_sam3_processor_postprocess_inverts_masks_and_pairs_grounding_boxes():
    processor = SAM3Processor(
        SAM3ProcessorConfig(
            image_size=(6, 10),
            top_k=1,
            score_threshold=0.5,
            labels=("bg", "cat"),
        )
    )
    ctx = SpatialTransform.resize((3, 5), (6, 10))
    mask_logits = np.full((1, 2, 3, 5), -10.0, dtype=np.float32)
    mask_logits[0, 0, 1:, 2:] = 10.0
    raw = HeadOutput(
        {
            "mask_logits": mx.array(mask_logits),
            "object_scores": mx.array([[0.9, 0.1]], dtype=mx.float32),
            "labels": mx.array([[1, 0]], dtype=mx.int32),
            "boxes": mx.array([[[0.5, 0.5, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]], dtype=mx.float32),
        }
    )

    result = processor.postprocess(raw, ctx)

    assert result.image_size == (3, 5)
    assert result.masks is not None
    assert result.masks.data.shape == (1, 3, 5)
    assert result.masks.kind == "instance"
    assert result.masks.labels == ["cat"]
    assert result.masks.data.dtype == np.bool_
    assert result.masks.data[0, 2, 4]
    assert result.detections is not None
    np.testing.assert_allclose(result.detections.boxes, [[0.0, 0.0, 5.0, 3.0]])
    np.testing.assert_allclose(result.detections.scores, [0.9])
    np.testing.assert_array_equal(result.detections.class_ids, [1])
    assert result.detections.labels == ["cat"]


def test_sam3_processor_postprocess_rejects_batch_results():
    processor = SAM3Processor(SAM3ProcessorConfig(image_size=(6, 10)))
    with pytest.raises(ValueError, match="one image"):
        processor.postprocess({"mask_logits": np.zeros((2, 1, 3, 5), dtype=np.float32)}, SpatialTransform.resize((3, 5), (6, 10)))
