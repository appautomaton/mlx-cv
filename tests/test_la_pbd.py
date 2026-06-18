import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv.backbones.llm.qwen2.cache import Qwen2KVCache
from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.models.locateanything.config import LocateAnythingConfig
from mlx_cv.models.locateanything.modeling import LocateAnythingModel
from mlx_cv.models.locateanything.pbd import (
    PBDDecoder,
    get_token_ids,
    handle_pattern,
    sample_block,
)


def _tiny_config(block_size=6):
    text = Qwen2Config(
        vocab_size=20,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        block_size=block_size,
        text_mask_token_id=15,
        null_token_id=13,
        switch_token_id=14,
        eos_token_id=2,
    )
    return LocateAnythingConfig(
        text_config=text,
        vocab_size=20,
        image_token_index=1,
        box_start_token_id=3,
        box_end_token_id=4,
        coord_start_token_id=5,
        coord_end_token_id=9,
        ref_start_token_id=10,
        ref_end_token_id=11,
        none_token_id=12,
        text_mask_token_id=15,
    )


def _logits(tokens, vocab=20):
    arr = np.full((len(tokens), vocab), -20.0, dtype=np.float32)
    for i, token in enumerate(tokens):
        arr[i, token] = 20.0
    return mx.array(arr)


class _FakeEmbedding:
    def __init__(self, vocab_size=20, hidden_size=8):
        self.weight = mx.zeros((vocab_size, hidden_size), dtype=mx.float32)
        self.hidden_size = hidden_size

    def __call__(self, input_ids):
        return mx.zeros((*input_ids.shape, self.hidden_size), dtype=mx.float32)


class _FakeLM:
    def __init__(self, responses, vocab_size=20, hidden_size=8):
        self.responses = list(responses)
        self.vocab_size = vocab_size
        self.model = type("FakeInner", (), {})()
        self.model.layers = [object()]
        self.model.embed_tokens = _FakeEmbedding(vocab_size, hidden_size)
        self.cache_lengths = []

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def __call__(self, input_ids=None, *, inputs_embeds=None, past_key_values=None, **kwargs):
        del kwargs
        if input_ids is not None:
            batch, q_len = input_ids.shape
        else:
            batch, q_len = inputs_embeds.shape[:2]
        if batch != 1:
            raise AssertionError(f"fake LM only supports batch size 1, got {batch}")
        if not self.responses:
            raise AssertionError("fake LM response queue is empty")

        if past_key_values is not None:
            key = mx.zeros((1, 1, q_len, 1), dtype=mx.float32)
            past_key_values.update(key, key, 0)
            self.cache_lengths.append(past_key_values.get_seq_length(0))

        tokens = self.responses.pop(0)
        if len(tokens) > q_len:
            raise AssertionError(f"fake response length {len(tokens)} exceeds q_len {q_len}")
        logits = np.full((1, q_len, self.vocab_size), -20.0, dtype=np.float32)
        start = q_len - len(tokens)
        for i, token in enumerate(tokens):
            logits[0, start + i, token] = 20.0
        return (mx.array(logits),)


def test_get_token_ids_from_config_uses_text_specials():
    ids = get_token_ids(_tiny_config())
    assert ids["box_start_token_id"] == 3
    assert ids["default_mask_token_id"] == 15
    assert ids["null_token_id"] == 13
    assert ids["im_end_token_id"] == 2


def test_sample_block_prefers_legal_box_frame():
    ids = get_token_ids(_tiny_config())
    out = sample_block(_logits([3, 5, 6, 7, 8, 4]), ids)
    assert out == [3, 5, 6, 7, 8, 4]
    assert handle_pattern(out, ids)["type"] == "coord_box"


def test_sample_block_handles_point_and_empty_box_patterns():
    ids = get_token_ids(_tiny_config())
    assert handle_pattern([3, 5, 6, 4, 13, 13], ids)["type"] == "point_box"
    assert handle_pattern([3, 12, 4, 13, 13, 13], ids)["type"] == "empty_box"


def test_sample_block_decodes_ref_tokens_when_box_is_illegal():
    ids = get_token_ids(_tiny_config())
    out = sample_block(_logits([10, 16, 17]), ids)
    assert out == [10, 16, 17]
    assert handle_pattern(out, ids)["type"] == "ref_object"


def test_pbd_decoder_rejects_future_token_block_size_mismatch():
    model = LocateAnythingModel(_tiny_config(block_size=6))
    with pytest.raises(ValueError, match="n_future_tokens"):
        PBDDecoder(model, n_future_tokens=4)


def test_pbd_decoder_rejects_non_reference_block_size():
    model = LocateAnythingModel(_tiny_config(block_size=4))
    with pytest.raises(ValueError, match="six-token"):
        PBDDecoder(model)


def test_pbd_decoder_hybrid_switch_markers():
    model = LocateAnythingModel(_tiny_config())
    decoder = PBDDecoder(model, generation_mode="hybrid")
    ids = get_token_ids(model.config)
    assert decoder._consume_block(_logits([3, 5, 13, 13, 13, 13]))[0] == "error_box"
    assert decoder._sample_ar(_logits([4])[0]) == ("box_end_ar", ids["box_end_token_id"])
    assert decoder._sample_ar(_logits([5])[0]) == ("coord_ar", ids["coord_start_token_id"])
    assert decoder._sample_ar(_logits([2])[0]) == ("im_end", ids["im_end_token_id"])


def test_qwen2_cache_trim_removes_speculative_window():
    cache = Qwen2KVCache(num_layers=1)
    key = mx.array(np.arange(1 * 2 * 5 * 4, dtype=np.float32).reshape(1, 2, 5, 4))
    value = key + 1
    cache.keys[0] = key
    cache.values[0] = value
    cache.trim(2)
    assert cache.get_seq_length(0) == 3
    assert np.array_equal(np.array(cache.keys[0]), np.array(key)[:, :, :3, :])


def test_pbd_generate_runs_mtp_prefill_and_forward_mtp_with_real_decoder_loop():
    cfg = _tiny_config()
    model = LocateAnythingModel(cfg)
    fake_lm = _FakeLM(
        responses=[
            [3, 5, 6, 7, 8, 4],
            [2, 13, 13, 13, 13, 13],
        ],
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.text_config.hidden_size,
    )
    model.language_model = fake_lm
    input_ids = mx.array([[1, 2]], dtype=mx.int32)

    out = model.pbd_generate(input_ids, generation_mode="hybrid", max_tokens=16)

    assert out == [3, 5, 6, 7, 8, 4, 2]
    assert fake_lm.responses == []
    assert fake_lm.cache_lengths == [8, 14]


def test_pbd_generate_runs_ar_fallback_and_switches_back_to_mtp():
    cfg = _tiny_config()
    model = LocateAnythingModel(cfg)
    fake_lm = _FakeLM(
        responses=[
            [3, 5, 13, 13, 13, 13],
            [6],
            [7],
            [8],
            [4],
            [2, 13, 13, 13, 13, 13],
        ],
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.text_config.hidden_size,
    )
    model.language_model = fake_lm
    input_ids = mx.array([[1, 2]], dtype=mx.int32)

    out = model.pbd_generate(input_ids, generation_mode="hybrid", max_tokens=16)

    assert out == [3, 5, 6, 7, 8, 4, 2]
    assert fake_lm.responses == []


def test_pbd_generate_rejects_implicit_batch_and_cache_assumptions():
    cfg = _tiny_config()
    model = LocateAnythingModel(cfg)
    model.language_model = _FakeLM([], vocab_size=cfg.vocab_size, hidden_size=cfg.text_config.hidden_size)
    decoder = PBDDecoder(model)
    cache = Qwen2KVCache(num_layers=1)
    embeds = mx.zeros((1, 2, cfg.text_config.hidden_size), dtype=mx.float32)

    with pytest.raises(ValueError, match="batch size 1"):
        decoder.generate(mx.array([[1, 2], [1, 2]], dtype=mx.int32), mx.zeros((2, 2, 8)), cache)

    cache.keys[0] = mx.zeros((1, 1, 1, 1), dtype=mx.float32)
    cache.values[0] = mx.zeros((1, 1, 1, 1), dtype=mx.float32)
    with pytest.raises(ValueError, match="empty cache"):
        decoder.generate(mx.array([[1, 2]], dtype=mx.int32), embeds, cache)
