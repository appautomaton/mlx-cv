import re

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")

from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.backbones.vision.moonvit.config import MoonViTConfig
from mlx_cv.models.locateanything.config import LocateAnythingConfig
from mlx_cv.models.locateanything.modeling import LocateAnythingModel
from mlx_cv.models.locateanything.processor import LocateAnythingProcessor


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


def _config():
    text = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        block_size=6,
        text_mask_token_id=7,
        null_token_id=30,
        switch_token_id=31,
    )
    vision = MoonViTConfig(
        hidden_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        intermediate_size=16,
        patch_size=2,
        num_channels=3,
        init_pos_emb_height=2,
        init_pos_emb_width=2,
        merge_kernel_size=(2, 2),
    )
    return LocateAnythingConfig(
        vision_config=vision,
        text_config=text,
        vocab_size=32,
        image_token_index=1,
        box_start_token_id=22,
        box_end_token_id=23,
        coord_start_token_id=100,
        coord_end_token_id=1100,
        ref_start_token_id=20,
        ref_end_token_id=21,
        none_token_id=24,
        text_mask_token_id=7,
    )


def test_predict_wires_preprocess_pbd_generate_and_postprocess():
    cfg = _config()
    model = LocateAnythingModel(cfg)
    processor = LocateAnythingProcessor(cfg, tokenizer=FakeTokenizer())
    calls = {}

    def fake_pbd_generate(input_ids, pixel_values=None, **kwargs):
        calls["input_ids_shape"] = tuple(input_ids.shape)
        calls["pixel_values_shape"] = tuple(pixel_values.shape)
        calls["grid"] = np.array(kwargs["image_grid_hws"])
        return [
            cfg.ref_start_token_id,
            4,
            cfg.ref_end_token_id,
            cfg.box_start_token_id,
            cfg.coord_start_token_id + 250,
            cfg.coord_start_token_id + 250,
            cfg.coord_start_token_id + 750,
            cfg.coord_start_token_id + 750,
            cfg.box_end_token_id,
        ]

    model.pbd_generate = fake_pbd_generate
    result = model.predict(Image.new("RGB", (5, 3)), "find <image-0>", processor=processor)

    assert calls["input_ids_shape"] == (1, 5)
    assert calls["pixel_values_shape"] == (8, 3, 2, 2)
    assert np.array_equal(calls["grid"], np.array([[2, 4]], dtype=np.int32))
    assert result.image_size == (3, 5)
    assert result.detections.labels == ["cat"]
    assert np.allclose(result.detections.boxes[0], [1.25, 0.75, 3.75, 2.25])


def test_predict_requires_tokenized_prompt_inputs():
    model = LocateAnythingModel(_config())
    processor = LocateAnythingProcessor(_config(), tokenizer=None)
    with pytest.raises(ValueError, match="tokenizer"):
        model.predict(Image.new("RGB", (4, 4)), "<image-0>", processor=processor)


def test_predict_accepts_tokenizer_without_manual_processor():
    cfg = _config()
    model = LocateAnythingModel(cfg)

    def fake_pbd_generate(input_ids, pixel_values=None, **kwargs):
        del input_ids, pixel_values, kwargs
        return [
            cfg.ref_start_token_id,
            4,
            cfg.ref_end_token_id,
            cfg.box_start_token_id,
            cfg.coord_start_token_id + 250,
            cfg.coord_start_token_id + 250,
            cfg.coord_start_token_id + 750,
            cfg.coord_start_token_id + 750,
            cfg.box_end_token_id,
        ]

    model.pbd_generate = fake_pbd_generate
    result = model.predict(Image.new("RGB", (5, 3)), "find <image-0>", tokenizer=FakeTokenizer())

    assert result.detections.labels == ["cat"]
    assert np.allclose(result.detections.boxes[0], [1.25, 0.75, 3.75, 2.25])


def test_predict_rejects_processor_and_tokenizer_together():
    model = LocateAnythingModel(_config())
    processor = LocateAnythingProcessor(_config(), tokenizer=FakeTokenizer())
    with pytest.raises(ValueError, match="either processor or tokenizer"):
        model.predict(Image.new("RGB", (4, 4)), "<image-0>", processor=processor, tokenizer=FakeTokenizer())
