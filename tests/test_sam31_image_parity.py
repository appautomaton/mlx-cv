from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_cv.models.sam3.sam31_checkpoint import load_sam3_weights
from mlx_cv.models.sam3.sam31_modeling import SAM3Model


def mask_iou(reference: np.ndarray, actual: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    union = np.logical_or(reference, actual).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(reference, actual).sum() / union)


def compare_sam31_image_outputs(reference: dict, actual: dict) -> dict:
    reference_boxes = np.asarray(reference["boxes"], dtype=np.float32)
    actual_boxes = np.asarray(actual["boxes"], dtype=np.float32)
    reference_scores = np.asarray(reference["scores"], dtype=np.float32)
    actual_scores = np.asarray(actual["scores"], dtype=np.float32)
    reference_masks = np.asarray(reference["masks"], dtype=bool)
    actual_masks = np.asarray(actual["masks"], dtype=bool)

    if reference_boxes.shape != actual_boxes.shape:
        raise AssertionError(
            f"box shape mismatch: {reference_boxes.shape} != {actual_boxes.shape}"
        )
    if reference_scores.shape != actual_scores.shape:
        raise AssertionError(
            f"score shape mismatch: {reference_scores.shape} != {actual_scores.shape}"
        )
    if reference_masks.shape != actual_masks.shape:
        raise AssertionError(
            f"mask shape mismatch: {reference_masks.shape} != {actual_masks.shape}"
        )

    ious = [mask_iou(ref, got) for ref, got in zip(reference_masks, actual_masks)]
    report = {
        "count_equal": len(reference_boxes) == len(actual_boxes),
        "min_mask_iou": min(ious, default=1.0),
        "max_box_error_px": float(
            np.max(np.abs(reference_boxes - actual_boxes))
            if reference_boxes.size
            else 0.0
        ),
        "max_score_error": float(
            np.max(np.abs(reference_scores - actual_scores))
            if reference_scores.size
            else 0.0
        ),
    }
    report["passed"] = bool(
        report["count_equal"]
        and report["min_mask_iou"] >= 0.98
        and report["max_box_error_px"] <= 2.0
        and report["max_score_error"] <= 0.02
    )
    return report


def test_sam31_image_public_output_gate_accepts_bf16_scale_drift():
    reference = {
        "boxes": np.array([[10.0, 12.0, 30.0, 32.0]], dtype=np.float32),
        "scores": np.array([0.75], dtype=np.float32),
        "masks": np.pad(np.ones((1, 4, 4), dtype=bool), ((0, 0), (2, 2), (2, 2))),
    }
    actual = {
        "boxes": reference["boxes"] + np.array([[0.5, -0.5, 1.0, -1.0]], dtype=np.float32),
        "scores": reference["scores"] + np.float32(0.01),
        "masks": reference["masks"].copy(),
    }

    report = compare_sam31_image_outputs(reference, actual)

    assert report["passed"]
    assert report["min_mask_iou"] == 1.0


def test_sam31_image_public_output_gate_rejects_semantic_drift():
    reference = {
        "boxes": np.zeros((1, 4), dtype=np.float32),
        "scores": np.array([0.5], dtype=np.float32),
        "masks": np.ones((1, 8, 8), dtype=bool),
    }
    actual = {
        "boxes": np.ones((1, 4), dtype=np.float32) * 3.0,
        "scores": np.array([0.55], dtype=np.float32),
        "masks": np.zeros((1, 8, 8), dtype=bool),
    }

    report = compare_sam31_image_outputs(reference, actual)

    assert not report["passed"]
    assert report["min_mask_iou"] == 0.0


def test_real_sam31_image_gate_when_required():
    if os.environ.get("MLX_CV_REQUIRE_SAM31_GATE") != "1":
        pytest.skip("real SAM 3.1 gate is opt-in")
    capture = Path(
        os.environ.get(
            "MLX_CV_SAM31_IMAGE_CAPTURE", "/tmp/sam31_official_robot_raw.npz"
        )
    )
    checkpoint = Path(os.environ["MLX_CV_SAM31_MLX"])
    if not capture.exists():
        pytest.fail(f"official SAM 3.1 image capture is missing: {capture}")

    reference = np.load(capture)
    model = load_sam3_weights(SAM3Model(), checkpoint)
    output = model(
        mx.array(reference["pixel_values"]),
        mx.array(reference["input_ids"]),
        mx.array(reference["attention_mask"]),
    )
    mx.eval(output)
    query = 44
    ref_mask = reference["pred_masks"][0, query] > 0
    got_mask = np.asarray(output.pred_masks)[0, query] > 0
    ref_box = reference["pred_boxes"][0, query] * 1008.0
    got_box = np.asarray(output.pred_boxes)[0, query] * 1008.0
    ref_score = (
        1.0 / (1.0 + np.exp(-reference["pred_logits"][0, query, 0]))
    ) * (1.0 / (1.0 + np.exp(-reference["presence_logits"][0, 0])))
    got_score = (1.0 / (1.0 + np.exp(-np.asarray(output.pred_logits)[0, query]))) * (
        1.0 / (1.0 + np.exp(-np.asarray(output.presence_logits)[0, 0]))
    )
    report = {
        "min_mask_iou": mask_iou(ref_mask, got_mask),
        "max_box_error_px": float(np.max(np.abs(ref_box - got_box))),
        "max_score_error": float(abs(ref_score - got_score)),
    }
    assert report["min_mask_iou"] >= 0.98, report
    assert report["max_box_error_px"] <= 2.0, report
    assert report["max_score_error"] <= 0.02, report
