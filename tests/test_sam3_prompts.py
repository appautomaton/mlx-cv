import numpy as np
import pytest

from mlx_cv.core.geometry import SpatialTransform
from mlx_cv.heads.segmentation import SAM3PCSPromptEncoder
from mlx_cv.models.sam3 import normalize_sam3_prompt, prepare_sam3_prompt
from mlx_cv.prompts import BoxPrompt, ExemplarPrompt, PointPrompt, TextPrompt


def test_sam3_prompt_normalization_accepts_strings_dataclasses_and_dicts():
    assert normalize_sam3_prompt("red car").texts == ("red car",)
    assert normalize_sam3_prompt(TextPrompt("dog")).texts == ("dog",)

    bundle = normalize_sam3_prompt({"text": ["cat", "dog"], "boxes": [[0, 0, 10, 20]]})
    assert bundle.texts == ("cat", "dog")
    assert bundle.boxes.boxes.shape == (1, 4)

    exemplar = normalize_sam3_prompt(
        {
            "exemplar_image": np.zeros((4, 8, 3), dtype=np.uint8),
            "exemplar_boxes": [[1, 1, 4, 3]],
        }
    )
    assert exemplar.exemplar.boxes.shape == (1, 4)


def test_sam3_prompt_normalization_rejects_deferred_prompt_state():
    with pytest.raises(NotImplementedError, match="PointPrompt"):
        normalize_sam3_prompt(PointPrompt([[1, 2]], labels=[1]))
    with pytest.raises(NotImplementedError, match="point prompts"):
        normalize_sam3_prompt({"points": [[1, 2]], "point_labels": [1]})
    with pytest.raises(NotImplementedError, match="mask"):
        normalize_sam3_prompt({"mask_prompt": np.zeros((4, 4))})
    with pytest.raises(NotImplementedError, match="video"):
        normalize_sam3_prompt({"video_state": {}})


def test_sam3_pcs_prompt_encoder_maps_boxes_to_model_space_cxcywh():
    encoder = SAM3PCSPromptEncoder(model_size=(20, 40))
    transform = SpatialTransform.resize((10, 20), (20, 40))
    encoded = encoder.encode_boxes(BoxPrompt([[2, 1, 10, 5]]), transform)

    np.testing.assert_allclose(encoded.boxes_cxcywh, [[0.3, 0.3, 0.4, 0.4]])
    np.testing.assert_array_equal(encoded.box_labels, [True])


def test_sam3_pcs_prompt_encoder_maps_exemplar_boxes_to_model_space():
    encoder = SAM3PCSPromptEncoder(model_size=(20, 40))
    exemplar = ExemplarPrompt(image=np.zeros((10, 20, 3), dtype=np.uint8), boxes=[[2, 1, 10, 5]])
    encoded = encoder.encode_exemplar(exemplar)

    assert encoded.boxes_cxcywh.shape == (0, 4)
    np.testing.assert_allclose(encoded.exemplar_boxes_cxcywh, [[0.3, 0.3, 0.4, 0.4]])
    np.testing.assert_array_equal(encoded.exemplar_labels, [True])


def test_prepare_sam3_prompt_combines_text_boxes_and_exemplar():
    prepared = prepare_sam3_prompt(
        [
            "red car",
            BoxPrompt([[2, 1, 10, 5]]),
            ExemplarPrompt(image=np.zeros((10, 20, 3), dtype=np.uint8), boxes=[[2, 1, 10, 5]]),
        ],
        transform=SpatialTransform.resize((10, 20), (20, 40)),
        model_size=(20, 40),
    )

    assert prepared.texts == ("red car",)
    np.testing.assert_allclose(prepared.geometry.boxes_cxcywh, [[0.3, 0.3, 0.4, 0.4]])
    np.testing.assert_allclose(prepared.geometry.exemplar_boxes_cxcywh, [[0.3, 0.3, 0.4, 0.4]])
