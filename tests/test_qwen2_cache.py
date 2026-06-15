import numpy as np
import mlx.core as mx
import pytest
from mlx.utils import tree_unflatten

from mlx_cv.backbones.llm.qwen2.cache import Qwen2KVCache
from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.backbones.llm.qwen2.modeling import Qwen2Attention, Qwen2ForCausalLM
from mlx_cv.parity import QWEN2_FIXTURE_CONFIG


WEIGHTS = "tests/fixtures/qwen2_tiny_fixture_weights.npz"


def _load_tiny_qwen2() -> Qwen2ForCausalLM:
    model = Qwen2ForCausalLM(Qwen2Config.from_dict(QWEN2_FIXTURE_CONFIG))
    weights = np.load(WEIGHTS, allow_pickle=False)
    params = []
    for key in weights.files:
        if key.startswith("__") or key == "lm_head.weight":
            continue
        params.append((key, mx.array(weights[key])))
    model.update(tree_unflatten(params))
    mx.eval(model.parameters())
    return model


def test_qwen2_attention_cache_stores_unrepeated_rope_applied_kv():
    cfg = Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    attn = Qwen2Attention(cfg, layer_idx=0)
    cache = Qwen2KVCache(num_layers=1)
    x = mx.array(np.arange(1 * 3 * 8, dtype=np.float32).reshape(1, 3, 8) / 10.0)
    mask = mx.zeros((1, 1, 3, 3), dtype=mx.float32)

    out, _, present = attn(
        x,
        attention_mask=mask,
        position_ids=mx.array([[0, 1, 2]], dtype=mx.int32),
        past_key_value=cache,
        use_cache=True,
    )
    mx.eval(out, cache.keys[0], cache.values[0])

    assert present is cache
    assert cache.keys[0].shape == (1, 2, 3, 2)
    assert cache.values[0].shape == (1, 2, 3, 2)
    assert cache.get_seq_length(0) == 3


def test_qwen2_cached_ar_step_matches_full_sequence_suffix():
    model = _load_tiny_qwen2()
    full_ids = mx.array([[3, 5, 4, 6]], dtype=mx.int32)
    prefix_ids = full_ids[:, :3]
    next_id = full_ids[:, 3:]

    with mx.stream(mx.cpu):
        full_hidden = model.model(input_ids=full_ids)[0]
        full_logits = model.compute_logits(full_hidden)
        cache = Qwen2KVCache(num_layers=model.config.num_hidden_layers)
        prefix_hidden, present = model.model(input_ids=prefix_ids, past_key_values=cache, use_cache=True)
        step_hidden, same_cache = model.model(input_ids=next_id, past_key_values=present, use_cache=True)
        step_logits = model.compute_logits(step_hidden)
        mx.eval(full_hidden, full_logits, prefix_hidden, step_hidden, step_logits)

    assert same_cache is cache
    assert cache.get_seq_length(0) == 4
    assert np.allclose(np.array(step_hidden), np.array(full_hidden[:, -1:]), rtol=1e-4, atol=1e-4)
    assert np.allclose(np.array(step_logits), np.array(full_logits[:, -1:]), rtol=1e-4, atol=1e-4)


def test_qwen2_cache_generation_window_mask_dispatch():
    model = _load_tiny_qwen2()
    input_ids = mx.array([[3, 5, 7, 7]], dtype=mx.int32)
    embeds = model.model.embed_tokens(input_ids)
    mask = model.model._prepare_attention_mask(
        input_ids=input_ids,
        inputs_embeds=embeds,
        attention_mask=None,
        position_ids=mx.array([[0, 1, 2, 3]], dtype=mx.int32),
        use_cache=True,
    )
    visible = np.isfinite(np.array(mask)[0, 0])
    expected = np.array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, False, True, True],
            [True, False, True, True],
        ]
    )
    assert np.array_equal(visible, expected)


def test_qwen2_cache_attention_mask_width_must_match_cached_keys():
    model = _load_tiny_qwen2()
    cache = Qwen2KVCache(num_layers=model.config.num_hidden_layers)
    model.model(input_ids=mx.array([[3, 5]], dtype=mx.int32), past_key_values=cache, use_cache=True)

    with pytest.raises(ValueError, match="width"):
        model.model(
            input_ids=mx.array([[4]], dtype=mx.int32),
            attention_mask=mx.ones((1, 1), dtype=mx.float32),
            past_key_values=cache,
            use_cache=True,
        )
