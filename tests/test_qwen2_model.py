import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.backbones.llm.qwen2.masks import make_causal_mask_4d
from mlx_cv.backbones.llm.qwen2.modeling import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from mlx_cv.core.registry import BACKBONES


def _tiny_config(**overrides) -> Qwen2Config:
    values = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        block_size=2,
        text_mask_token_id=7,
    )
    values.update(overrides)
    return Qwen2Config(**values)


def test_qwen2_model_input_ids_and_inputs_embeds_match_when_embeddings_equivalent():
    model = Qwen2Model(_tiny_config(num_hidden_layers=1))
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    position_ids = mx.array([[0, 1, 2]], dtype=mx.int32)
    inputs_embeds = model.embed_tokens(input_ids)

    with mx.stream(mx.cpu):
        from_ids = model(input_ids=input_ids, position_ids=position_ids)[0]
        from_embeds = model(inputs_embeds=inputs_embeds, position_ids=position_ids)[0]
        mx.eval(from_ids, from_embeds)

    assert np.allclose(np.array(from_ids), np.array(from_embeds), rtol=1e-6, atol=1e-6)


def test_qwen2_decoder_layer_residual_order_matches_manual_expansion():
    cfg = _tiny_config(num_hidden_layers=1)
    layer = Qwen2DecoderLayer(cfg, layer_idx=0)
    hidden = mx.array(np.arange(1 * 3 * 8, dtype=np.float32).reshape(1, 3, 8) / 10.0)
    position_ids = mx.array([[0, 1, 2]], dtype=mx.int32)
    mask = make_causal_mask_4d(1, 3)

    with mx.stream(mx.cpu):
        out = layer(hidden, attention_mask=mask, position_ids=position_ids)[0]
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn(normed, attention_mask=mask, position_ids=position_ids)[0]
        after_attn = residual + attn
        manual = after_attn + layer.mlp(layer.post_attention_layernorm(after_attn))
        mx.eval(out, manual)

    assert np.allclose(np.array(out), np.array(manual), rtol=1e-6, atol=1e-6)


def test_qwen2_for_causal_lm_uses_tied_embedding_logits_without_lm_head_param():
    lm = Qwen2ForCausalLM(_tiny_config(num_hidden_layers=0))
    weight = np.arange(32 * 8, dtype=np.float32).reshape(32, 8) / 100.0
    lm.model.embed_tokens.weight = mx.array(weight)
    hidden = mx.array(np.arange(1 * 2 * 8, dtype=np.float32).reshape(1, 2, 8) / 10.0)

    with mx.stream(mx.cpu):
        logits = lm.compute_logits(hidden)
        mx.eval(logits)
    assert np.allclose(np.array(logits), np.array(hidden) @ weight.T, rtol=1e-6, atol=1e-6)
    assert not any(key.startswith("lm_head") for key, _ in tree_flatten(lm.parameters()))


def test_qwen2_model_inference_mask_dispatch_wires_sdlm_branch():
    model = Qwen2Model(_tiny_config(num_hidden_layers=0, text_mask_token_id=7, block_size=2))
    input_ids = mx.array([[3, 5, 7, 7]], dtype=mx.int32)
    inputs_embeds = model.embed_tokens(input_ids)
    position_ids = mx.array([[0, 1, 0, 1]], dtype=mx.int32)

    mask = model._prepare_attention_mask(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        position_ids=position_ids,
    )
    visible = np.isfinite(np.array(mask)[0, 0])
    expected = np.array(
        [
            [True, False, False, False],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    assert np.array_equal(visible, expected)

    ar_input_ids = mx.array([[3, 5, 7, 4]], dtype=mx.int32)
    ar_mask = model._prepare_attention_mask(
        input_ids=ar_input_ids,
        inputs_embeds=model.embed_tokens(ar_input_ids),
        attention_mask=None,
        position_ids=mx.array([[0, 1, 2, 3]], dtype=mx.int32),
    )
    assert np.array_equal(
        np.isfinite(np.array(ar_mask)[0, 0]),
        np.isfinite(np.array(make_causal_mask_4d(1, 4))[0, 0]),
    )


def test_qwen2_modeling_import_registers_concrete_llm_builder_once():
    assert "qwen2.5-3b" in BACKBONES.list(kind="llm")
    model = BACKBONES.get("qwen2.5-3b")(_tiny_config(num_hidden_layers=0))
    assert isinstance(model, Qwen2ForCausalLM)
