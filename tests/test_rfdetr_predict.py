import numpy as np
import pytest

from mlx_cv.parity import RFDETR_FIXTURE_CONFIG, rfdetr_fixed_image

mx = pytest.importorskip("mlx.core")

from mlx_cv.backbones.vision.dinov2 import DINOv2Config  # noqa: E402
from mlx_cv.core.types import Detections, Result  # noqa: E402
from mlx_cv.heads.detection import RFDETRDecoderConfig  # noqa: E402
from mlx_cv.models.rfdetr import (  # noqa: E402
    RFDETRConfig,
    RFDETRModel,
    RFDETRProcessor,
    RFDETRProcessorConfig,
    load_rfdetr_weights,
)


def _cfg():
    cfg = RFDETR_FIXTURE_CONFIG
    return RFDETRConfig(
        backbone=DINOv2Config(**cfg["backbone"]),
        out_layers=tuple(cfg["out_layers"]),
        projector_out_channels=cfg["projector_out_channels"],
        projector_scale_factors=tuple(cfg["projector_scale_factors"]),
        decoder=RFDETRDecoderConfig(**cfg["decoder"]),
    )


def _model():
    model = RFDETRModel(_cfg())
    return load_rfdetr_weights(model, "tests/fixtures/rfdetr_tiny_fixture_weights.npz")


def test_rfdetr_predict_returns_typed_detections_for_fixed_image():
    cfg = RFDETR_FIXTURE_CONFIG
    processor = RFDETRProcessor(
        RFDETRProcessorConfig(
            image_size=tuple(cfg["image_size"]),
            top_k=int(cfg["num_select"]),
            labels=tuple(cfg["labels"]),
        )
    )

    result = _model().predict(rfdetr_fixed_image(), processor=processor)

    assert isinstance(result, Result)
    assert isinstance(result.detections, Detections)
    assert result.image_size == tuple(cfg["image_size"])
    assert len(result.detections) == int(cfg["num_select"])
    assert result.detections.boxes.shape == (int(cfg["num_select"]), 4)
    assert result.detections.scores.shape == (int(cfg["num_select"]),)
    assert result.detections.class_ids.shape == (int(cfg["num_select"]),)
    assert len(result.detections.labels) == int(cfg["num_select"])
    assert np.all(result.detections.boxes[:, 0::2] >= 0)
    assert np.all(result.detections.boxes[:, 1::2] >= 0)


def test_rfdetr_predict_builds_default_processor_from_options():
    cfg = RFDETR_FIXTURE_CONFIG
    result = _model().predict(
        rfdetr_fixed_image(),
        image_size=tuple(cfg["image_size"]),
        top_k=int(cfg["num_select"]),
        labels=tuple(cfg["labels"]),
    )

    assert isinstance(result.detections, Detections)
    assert result.detections.labels is not None


def test_rfdetr_predict_rejects_processor_and_options_together():
    cfg = RFDETR_FIXTURE_CONFIG
    processor = RFDETRProcessor(RFDETRProcessorConfig(image_size=tuple(cfg["image_size"])))
    with pytest.raises(ValueError, match="processor is not provided"):
        _model().predict(rfdetr_fixed_image(), processor=processor, top_k=1)
