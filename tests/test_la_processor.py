import re

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")

from mlx_cv.core.types import Detections, Points
from mlx_cv.models.locateanything.config import LocateAnythingConfig
from mlx_cv.models.locateanything.processor import (
    LocateAnythingProcessor,
    LocateAnythingProcessorConfig,
)


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.vocab = {
            "<IMG_CONTEXT>": 1,
            "<img>": 2,
            "</img>": 3,
            "cat": 4,
            "find": 5,
        }
        self.inv = {v: k for k, v in self.vocab.items()}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def __call__(self, texts, padding=True):
        rows = []
        for text in texts:
            tokens = re.findall(r"<IMG_CONTEXT>|<img>|</img>|\w+", text)
            rows.append([self.vocab.get(tok, 9) for tok in tokens])
        width = max(len(row) for row in rows)
        return {
            "input_ids": [row + [0] * (width - len(row)) for row in rows],
            "attention_mask": [[1] * len(row) + [0] * (width - len(row)) for row in rows],
        }

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.inv.get(int(i), f"tok{int(i)}") for i in ids)


def _processor():
    config = LocateAnythingProcessorConfig(patch_size=2, merge_kernel_size=(2, 2))
    model_config = LocateAnythingConfig(
        image_token_index=1,
        coord_start_token_id=100,
        coord_end_token_id=1100,
        ref_start_token_id=20,
        ref_end_token_id=21,
        box_start_token_id=22,
        box_end_token_id=23,
        none_token_id=24,
    )
    proc = LocateAnythingProcessor(config, tokenizer=FakeTokenizer())
    proc.model_config = model_config
    return proc


def test_preprocess_patchifies_and_expands_image_placeholder():
    proc = _processor()
    image = Image.new("RGB", (5, 3), color=(128, 64, 32))
    inputs, ctx = proc.preprocess(image, "find <image-0>")

    assert tuple(inputs["pixel_values"].shape) == (8, 3, 2, 2)
    assert np.array_equal(np.array(inputs["image_grid_hws"]), np.array([[2, 4]], dtype=np.int32))
    assert inputs["image_token_id"] == 1
    assert ctx.image_size == (3, 5)
    assert ctx.model_size == (4, 8)
    assert ctx.expanded_text == ["find <img><IMG_CONTEXT><IMG_CONTEXT></img>"]
    image_token_count = int(np.sum(np.array(inputs["input_ids"]) == 1))
    assert image_token_count == 2


def test_preprocess_rejects_placeholder_image_mismatch():
    proc = _processor()
    image = Image.new("RGB", (4, 4))
    with pytest.raises(ValueError, match="placeholders"):
        proc.preprocess(image, "find")


def test_postprocess_maps_boxes_and_points_to_original_pixels():
    proc = _processor()
    image = Image.new("RGB", (5, 3), color=(0, 0, 0))
    _, ctx = proc.preprocess(image, "<image-0>")
    cfg = proc.model_config
    toks = [
        cfg.ref_start_token_id,
        4,
        cfg.ref_end_token_id,
        cfg.box_start_token_id,
        cfg.coord_start_token_id + 250,
        cfg.coord_start_token_id + 250,
        cfg.coord_start_token_id + 750,
        cfg.coord_start_token_id + 750,
        cfg.box_end_token_id,
        cfg.box_start_token_id,
        cfg.coord_start_token_id + 500,
        cfg.coord_start_token_id + 500,
        cfg.box_end_token_id,
    ]

    result = proc.postprocess(toks, ctx)

    assert result.image_size == (3, 5)
    assert isinstance(result.detections, Detections)
    assert isinstance(result.points, Points)
    assert result.detections.labels == ["cat"]
    assert result.points.labels == ["cat"]
    assert np.allclose(result.detections.boxes[0], [1.25, 0.75, 3.75, 2.25])
    assert np.allclose(result.points.points[0], [2.5, 1.5])
