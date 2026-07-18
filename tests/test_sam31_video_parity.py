from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_cv.models.sam3.sam31_checkpoint import load_sam31_tracker_weights
from mlx_cv.models.sam3.sam31_tracker import SAM31MultiplexTracker


def binary_iou(reference, actual) -> float:
    reference = np.asarray(reference) > 0
    actual = np.asarray(actual) > 0
    union = np.logical_or(reference, actual).sum()
    return 1.0 if union == 0 else float(np.logical_and(reference, actual).sum() / union)


def compare_decoder_capture(reference: dict, actual: dict) -> dict:
    report = {
        "mask_iou": binary_iou(reference["masks"], actual["masks"]),
        "max_mask_logit_error": float(
            np.max(np.abs(reference["masks"] - actual["masks"]))
        ),
        "max_iou_error": float(
            np.max(np.abs(reference["iou_pred"] - actual["iou_pred"]))
        ),
        "max_object_logit_error": float(
            np.max(
                np.abs(
                    reference["object_score_logits"]
                    - actual["object_score_logits"]
                )
            )
        ),
    }
    report["passed"] = bool(
        report["mask_iou"] >= 0.98
        and report["max_mask_logit_error"] <= 0.1
        and report["max_iou_error"] <= 0.06
        and report["max_object_logit_error"] <= 0.15
    )
    return report


def test_sam31_video_component_gate_accepts_bf16_drift():
    reference = {
        "masks": np.ones((1, 1, 3, 4, 4), dtype=np.float32),
        "iou_pred": np.ones((1, 1, 3), dtype=np.float32),
        "object_score_logits": np.ones((1, 1, 1), dtype=np.float32),
    }
    actual = {
        "masks": reference["masks"] - 0.02,
        "iou_pred": reference["iou_pred"] + 0.01,
        "object_score_logits": reference["object_score_logits"] - 0.05,
    }
    assert compare_decoder_capture(reference, actual)["passed"]


def test_real_sam31_multiplex_decoder_gate_when_required():
    if os.environ.get("MLX_CV_REQUIRE_SAM31_GATE") != "1":
        pytest.skip("real SAM 3.1 gate is opt-in")
    capture = Path(
        os.environ.get(
            "MLX_CV_SAM31_VIDEO_CAPTURE", "/tmp/sam31_official_mux_decoder.npz"
        )
    )
    checkpoint = Path(os.environ["MLX_CV_SAM31_MLX"])
    if not capture.exists():
        pytest.fail(f"official SAM 3.1 video capture is missing: {capture}")

    reference = dict(np.load(capture))
    tracker = load_sam31_tracker_weights(SAM31MultiplexTracker(), checkpoint)
    tracker.eval()
    output = tracker.sam_mask_decoder(
        mx.array(reference["image"]).transpose(0, 2, 3, 1),
        mx.array(reference["pe"]).transpose(0, 2, 3, 1),
        [
            mx.array(reference["h0"]).transpose(0, 2, 3, 1),
            mx.array(reference["h1"]).transpose(0, 2, 3, 1),
        ],
        mx.array(reference["extra"]),
        True,
    )
    mx.eval(output)
    actual = {key: np.asarray(output[key]) for key in output}
    report = compare_decoder_capture(reference, actual)

    assert report["passed"], report
