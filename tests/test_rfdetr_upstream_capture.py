from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_tool(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rfdetr_checkpoint = _load_tool("rfdetr_checkpoint", "tools/rfdetr_checkpoint.py")
rfdetr_upstream = _load_tool("rfdetr_upstream", "tools/rfdetr_upstream.py")


def _checkpoint_or_skip():
    required = rfdetr_checkpoint.required_gate_enabled()
    try:
        info = rfdetr_checkpoint.resolve_rfdetr_nano_checkpoint(required=required)
    except rfdetr_checkpoint.CheckpointError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    if info is None:
        pytest.skip("RF-DETR Nano checkpoint not configured")
    return info, required


def test_rfdetr_upstream_capture_runs_real_checkpoint(capsys):
    checkpoint, required = _checkpoint_or_skip()
    try:
        capture = rfdetr_upstream.capture_rfdetr_nano_reference(checkpoint)
    except rfdetr_upstream.ReferenceDependencyError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))

    rfdetr_checkpoint.print_checkpoint_evidence(checkpoint)
    out = capsys.readouterr().out
    assert str(checkpoint.path) in out
    assert checkpoint.md5 in out

    assert capture.input_image.shape == (28, 28, 3)
    assert capture.input_image.dtype == np.uint8
    assert capture.raw_logits.ndim == 3
    assert capture.raw_boxes.ndim == 3
    assert capture.raw_logits.shape[:2] == capture.raw_boxes.shape[:2]
    assert capture.raw_boxes.shape[-1] == 4
    assert capture.boxes.ndim == 2
    assert capture.boxes.shape[-1] == 4
    assert capture.scores.ndim == 1
    assert capture.class_ids.ndim == 1
    assert len(capture.boxes) == len(capture.scores) == len(capture.class_ids)
    assert capture.tap_gaps
