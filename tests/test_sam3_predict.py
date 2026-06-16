import numpy as np
import pytest

from mlx_cv.parity import SAM3_FIXTURE_CONFIG, sam3_fixed_image, sam3_pcs_prompt, sam3_text_prompt

mx = pytest.importorskip("mlx.core")

from mlx_cv.core.types import Detections, Masks, Result  # noqa: E402
from mlx_cv.heads.segmentation import SAM3DecoderConfig  # noqa: E402
from mlx_cv.models.sam3 import (  # noqa: E402
    SAM3Config,
    SAM3ImageBackboneConfig,
    SAM3Model,
    SAM3Processor,
    SAM3ProcessorConfig,
    SAM3TextConfig,
    SAM3Tokenizer,
    load_sam3_weights,
)


def _cfg():
    cfg = SAM3_FIXTURE_CONFIG
    tokenizer = SAM3Tokenizer(context_length=int(cfg["text"]["context_length"]))
    text_cfg = dict(cfg["text"])
    text_cfg["vocab_size"] = tokenizer.vocab_size
    return SAM3Config(
        image=SAM3ImageBackboneConfig(**cfg["image"]),
        text=SAM3TextConfig(**text_cfg),
        decoder=SAM3DecoderConfig(**cfg["decoder"]),
    )


def _processor():
    cfg = SAM3_FIXTURE_CONFIG
    return SAM3Processor(
        SAM3ProcessorConfig(
            image_size=tuple(cfg["image_size"]),
            top_k=int(cfg["num_select"]),
            labels=tuple(cfg["labels"]),
        )
    )


def _model():
    tokenizer = SAM3Tokenizer(context_length=int(SAM3_FIXTURE_CONFIG["text"]["context_length"]))
    return load_sam3_weights(SAM3Model(_cfg(), tokenizer=tokenizer), "tests/fixtures/sam3_tiny_fixture_weights.npz")


def _assert_typed_result(result):
    cfg = SAM3_FIXTURE_CONFIG
    assert isinstance(result, Result)
    assert isinstance(result.masks, Masks)
    assert isinstance(result.detections, Detections)
    assert result.image_size == tuple(cfg["image_size"])
    assert result.masks.data.shape == (int(cfg["num_select"]), *tuple(cfg["image_size"]))
    assert result.masks.kind == "instance"
    assert result.detections.boxes.shape == (int(cfg["num_select"]), 4)
    assert result.detections.scores.shape == (int(cfg["num_select"]),)
    assert result.detections.class_ids.shape == (int(cfg["num_select"]),)
    assert len(result.detections.labels) == int(cfg["num_select"])
    assert np.all(result.detections.boxes[:, 0::2] >= 0)
    assert np.all(result.detections.boxes[:, 1::2] >= 0)


def test_sam3_predict_returns_typed_masks_and_detections_for_text_prompt():
    result = _model().predict(sam3_fixed_image(), sam3_text_prompt(), processor=_processor())
    _assert_typed_result(result)


def test_sam3_predict_returns_typed_masks_and_detections_for_pcs_prompt():
    result = _model().predict(sam3_fixed_image(), sam3_pcs_prompt(), processor=_processor())
    _assert_typed_result(result)


def test_sam3_predict_builds_default_processor_from_options():
    cfg = SAM3_FIXTURE_CONFIG
    result = _model().predict(
        sam3_fixed_image(),
        sam3_text_prompt(),
        image_size=tuple(cfg["image_size"]),
        top_k=int(cfg["num_select"]),
        labels=tuple(cfg["labels"]),
    )

    assert isinstance(result.masks, Masks)
    assert isinstance(result.detections, Detections)


def test_sam3_predict_rejects_processor_and_options_together():
    processor = _processor()
    with pytest.raises(ValueError, match="processor is not provided"):
        _model().predict(sam3_fixed_image(), sam3_text_prompt(), processor=processor, top_k=1)
