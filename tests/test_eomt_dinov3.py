import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv.core import HeadOutput, Result
from mlx_cv.models.eomt_dinov3 import (
    EoMTDINOv3,
    EoMTDINOv3Config,
    EoMTDINOv3Processor,
    EoMTDINOv3ProcessorConfig,
    EoMTDINOv3ProcessorContext,
)


def test_official_config_axes_are_parsed_without_transformers():
    config = EoMTDINOv3Config.from_dict(
        {
            "model_type": "eomt_dinov3",
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
            "patch_size": 16,
            "num_channels": 3,
            "num_register_tokens": 4,
            "layer_norm_eps": 1e-5,
            "layerscale_value": 1.0,
            "rope_theta": 100.0,
            "image_size": 640,
            "num_queries": 200,
            "num_blocks": 3,
            "num_upscale_blocks": 2,
            "id2label": {"0": "thing", "1": "stuff"},
        }
    )

    assert config.num_classes == 2
    assert config.labels == ("thing", "stuff")
    assert config.backbone.embed_dim == 384
    assert config.backbone.n_storage_tokens == 4
    assert config.backbone.layerscale_init == 1.0
    assert config.backbone.layer_norm_eps == pytest.approx(1e-5)


def test_tiny_forward_has_query_boundary_and_final_contract():
    config = EoMTDINOv3Config.tiny_fixture()
    model = EoMTDINOv3(config)
    output = model(mx.zeros((1, 3, 8, 8)), capture_taps=True)
    mx.eval(output.data)

    assert isinstance(output, HeadOutput)
    assert output.data["masks_queries_logits"].shape == (1, 3, 4, 4)
    assert output.data["class_queries_logits"].shape == (1, 3, 5)
    assert output.data["last_hidden_state"].shape == (1, 10, 16)
    assert len(output.data["masks_queries_logits_per_layer"]) == 3
    assert output.data["taps"]["query_insertion"].shape == (1, 10, 16)


def test_processor_preprocess_preserves_aspect_ratio_and_pads_bottom_right():
    processor = EoMTDINOv3Processor(EoMTDINOv3ProcessorConfig(image_size=8, stuff_classes=()))
    image = np.full((2, 4, 3), 255, dtype=np.uint8)

    model_input, context = processor.preprocess(image)
    values = np.asarray(model_input["pixel_values"])

    assert values.shape == (1, 3, 8, 8)
    assert context.image_size == (2, 4)
    assert context.resized_size == (4, 8)
    np.testing.assert_allclose(values[0, :, 0, 0], (1.0 - np.array(processor.config.mean)) / processor.config.std)
    np.testing.assert_allclose(values[0, :, 7, 7], -np.array(processor.config.mean) / processor.config.std)


def test_processor_returns_panoptic_result_with_segment_metadata():
    processor = EoMTDINOv3Processor(
        EoMTDINOv3ProcessorConfig(
            image_size=2,
            threshold=0.5,
            mask_threshold=0.5,
            overlap_mask_area_threshold=0.5,
            stuff_classes=(2,),
            labels=("zero", "one", "two", "three"),
        )
    )
    mask_logits = np.array(
        [[[[10.0, -10.0], [10.0, -10.0]], [[-10.0, 10.0], [-10.0, 10.0]], [[0.0, 0.0], [0.0, 0.0]]]],
        dtype=np.float32,
    )
    class_logits = np.full((1, 3, 5), -10.0, dtype=np.float32)
    class_logits[0, 0, 0] = 10.0
    class_logits[0, 1, 2] = 10.0
    class_logits[0, 2, 4] = 10.0
    context = EoMTDINOv3ProcessorContext(
        image_size=(2, 2), resized_size=(2, 2), model_size=(2, 2)
    )

    result = processor.postprocess(
        {"masks_queries_logits": mask_logits, "class_queries_logits": class_logits},
        context,
    )

    assert isinstance(result, Result)
    assert result.masks.kind == "panoptic"
    np.testing.assert_array_equal(result.masks.data, [[0, 1], [0, 1]])
    assert result.masks.labels == ["zero", "two"]
    assert [segment["label_id"] for segment in result.metadata["segments_info"]] == [0, 2]
