import pathlib

import numpy as np
import pytest

from mlx_cv.parity import RFDETR_FIXTURE_CONFIG, assert_parity, bisect, load_case, rfdetr_tap_order

mx = pytest.importorskip("mlx.core")

from mlx_cv.backbones.vision.dinov2 import DINOv2Config  # noqa: E402
from mlx_cv.core.geometry import SpatialTransform  # noqa: E402
from mlx_cv.heads.detection import RFDETRDecoderConfig  # noqa: E402
from mlx_cv.models.rfdetr import (  # noqa: E402
    RFDETRConfig,
    RFDETRModel,
    RFDETRProcessor,
    RFDETRProcessorConfig,
    load_rfdetr_weights,
)


_FIX = pathlib.Path(__file__).parent / "fixtures"
_ATOL = 1e-5
_TAP_ATOL = 1e-4


def _cfg():
    cfg = RFDETR_FIXTURE_CONFIG
    return RFDETRConfig(
        backbone=DINOv2Config(**cfg["backbone"]),
        out_layers=tuple(cfg["out_layers"]),
        projector_out_channels=cfg["projector_out_channels"],
        projector_scale_factors=tuple(cfg["projector_scale_factors"]),
        decoder=RFDETRDecoderConfig(**cfg["decoder"]),
    )


def _processor():
    cfg = RFDETR_FIXTURE_CONFIG
    return RFDETRProcessor(
        RFDETRProcessorConfig(
            image_size=tuple(cfg["image_size"]),
            top_k=int(cfg["num_select"]),
            labels=tuple(cfg["labels"]),
        )
    )


def _run_parity():
    cfg = RFDETR_FIXTURE_CONFIG
    with mx.stream(mx.cpu):
        case = load_case(_FIX / "rfdetr_tiny_fixture.npz")
        model = RFDETRModel(_cfg())
        load_rfdetr_weights(model, _FIX / "rfdetr_tiny_fixture_weights.npz")
        raw = model(mx.array(case.inputs["x"]), capture_taps=True)
        result = _processor().postprocess(raw, SpatialTransform.identity(tuple(cfg["image_size"])))
        mx.eval(raw.data)

    got = {
        "logits": np.array(raw["logits"]),
        "boxes": np.array(raw["boxes"]),
        "result_boxes": result.detections.boxes,
        "scores": result.detections.scores,
        "class_ids": result.detections.class_ids,
    }
    taps = {k: np.array(v) for k, v in raw["taps"].items()}
    taps["result.boxes"] = result.detections.boxes
    taps["result.scores"] = result.detections.scores
    taps["result.class_ids"] = result.detections.class_ids
    return case, got, taps


def test_rfdetr_tiny_fixture_raw_and_result_outputs_match():
    case, got, _ = _run_parity()
    assert_parity(got, case.expected, atol=_ATOL, rtol=_ATOL, name=case.name)


def test_rfdetr_taps_match_schema_and_bisect_clean():
    case, _, taps = _run_parity()
    assert list(taps.keys()) == rfdetr_tap_order()
    drift = bisect(case.taps, taps, atol=_TAP_ATOL, rtol=_TAP_ATOL)
    if drift is not None:
        np.testing.assert_allclose(
            taps[drift],
            case.taps[drift],
            atol=_TAP_ATOL,
            rtol=_TAP_ATOL,
            err_msg=f"first drifting tap: {drift}",
        )


def test_rfdetr_bisect_localizes_injected_drift():
    case, _, taps = _run_parity()
    corrupted = dict(taps)
    corrupted["decoder.deformable_attention_0"] = corrupted["decoder.deformable_attention_0"] + 1.0
    assert (
        bisect(case.taps, corrupted, atol=_TAP_ATOL, rtol=_TAP_ATOL)
        == "decoder.deformable_attention_0"
    )
