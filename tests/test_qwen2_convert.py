import numpy as np
import mlx.core as mx

from mlx_cv.backbones.llm.qwen2.cache import Qwen2KVCache
from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.backbones.llm.qwen2.convert import convert_qwen2_state_dict, load_qwen2_weights
from mlx_cv.backbones.llm.qwen2.modeling import Qwen2ForCausalLM
from mlx_cv.parity import QWEN2_FIXTURE_CONFIG, assert_parity, load_case


FIXTURE = "tests/fixtures/qwen2_tiny_fixture.npz"
WEIGHTS = "tests/fixtures/qwen2_tiny_fixture_weights.npz"


def _visible(mask: mx.array) -> np.ndarray:
    values = np.array(mask)
    return np.isfinite(values) & (values > -1e20)


def test_qwen2_convert_drops_only_lossless_tied_lm_head_and_metadata():
    embed = np.arange(32, dtype=np.float32).reshape(4, 8)
    state = {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": embed.copy(),
        "model.layers.0.self_attn.q_proj.bias": np.ones((8,), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.bias": np.ones((4,), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.bias": np.ones((4,), dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.ones((8, 8), dtype=np.float32),
        "__versions_json__": np.asarray("{}"),
    }

    out = dict(convert_qwen2_state_dict(state))

    assert "model.embed_tokens.weight" in out
    assert "lm_head.weight" not in out
    assert "__versions_json__" not in out
    assert "model.layers.0.self_attn.q_proj.bias" in out
    assert "model.layers.0.self_attn.k_proj.bias" in out
    assert "model.layers.0.self_attn.v_proj.bias" in out
    assert "model.layers.0.self_attn.o_proj.weight" in out
    assert "model.layers.0.self_attn.o_proj.bias" not in out


def test_qwen2_convert_rejects_untied_lm_head_drop():
    state = {
        "model.embed_tokens.weight": np.zeros((4, 8), dtype=np.float32),
        "lm_head.weight": np.ones((4, 8), dtype=np.float32),
    }
    try:
        convert_qwen2_state_dict(state)
    except ValueError as exc:
        assert "lm_head.weight" in str(exc)
    else:
        raise AssertionError("expected untied lm_head drop to fail")


def test_load_qwen2_weights_matches_fixture_no_cache_and_cached_step():
    case = load_case(FIXTURE)
    model = load_qwen2_weights(Qwen2ForCausalLM(Qwen2Config.from_dict(QWEN2_FIXTURE_CONFIG)), WEIGHTS)
    input_ids = mx.array(case.inputs["input_ids"].astype(np.int32))
    position_ids = mx.array(case.inputs["position_ids"].astype(np.int32))

    with mx.stream(mx.cpu):
        hidden = model.model(input_ids=input_ids, position_ids=position_ids)[0]
        logits = model.compute_logits(hidden)
        mask = model.model._prepare_attention_mask(
            input_ids=input_ids,
            inputs_embeds=model.model.embed_tokens(input_ids),
            attention_mask=None,
            position_ids=position_ids,
        )
        mx.eval(hidden, logits, mask)

    assert np.array_equal(_visible(mask), case.expected["attention_mask_visible"])
    assert_parity(np.array(hidden), case.expected["hidden_states"], atol=1e-4, rtol=1e-4, name="loaded hidden")
    assert_parity(np.array(logits), case.expected["logits"], atol=1e-4, rtol=1e-4, name="loaded logits")

    full_ids = mx.array([[3, 5, 4, 6]], dtype=mx.int32)
    with mx.stream(mx.cpu):
        full_hidden = model.model(input_ids=full_ids)[0]
        full_logits = model.compute_logits(full_hidden)
        cache = Qwen2KVCache(num_layers=model.config.num_hidden_layers)
        _, present = model.model(input_ids=full_ids[:, :3], past_key_values=cache, use_cache=True)
        step_hidden, _ = model.model(input_ids=full_ids[:, 3:], past_key_values=present, use_cache=True)
        step_logits = model.compute_logits(step_hidden)
        mx.eval(full_hidden, full_logits, step_hidden, step_logits)

    assert np.allclose(np.array(step_hidden), np.array(full_hidden[:, -1:]), rtol=1e-4, atol=1e-4)
    assert np.allclose(np.array(step_logits), np.array(full_logits[:, -1:]), rtol=1e-4, atol=1e-4)
