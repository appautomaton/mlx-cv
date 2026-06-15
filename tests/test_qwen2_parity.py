import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten

from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.backbones.llm.qwen2.modeling import Qwen2ForCausalLM
from mlx_cv.parity import QWEN2_FIXTURE_CONFIG, assert_parity, load_case


FIXTURE = "tests/fixtures/qwen2_tiny_fixture.npz"
WEIGHTS = "tests/fixtures/qwen2_tiny_fixture_weights.npz"


def _visible(mask: mx.array) -> np.ndarray:
    values = np.array(mask)
    return np.isfinite(values) & (values > -1e20)


def _load_tiny_qwen2() -> Qwen2ForCausalLM:
    model = Qwen2ForCausalLM(Qwen2Config.from_dict(QWEN2_FIXTURE_CONFIG))
    weights = np.load(WEIGHTS, allow_pickle=False)
    assert np.array_equal(weights["lm_head.weight"], weights["model.embed_tokens.weight"])

    params = []
    for key in weights.files:
        if key.startswith("__") or key == "lm_head.weight":
            continue
        params.append((key, mx.array(weights[key])))
    model.update(tree_unflatten(params))
    mx.eval(model.parameters())
    return model


def test_qwen2_tiny_fixture_no_cache_hidden_logits_and_mask_visibility_match_reference():
    case = load_case(FIXTURE)
    model = _load_tiny_qwen2()
    input_ids = mx.array(case.inputs["input_ids"].astype(np.int32))
    position_ids = mx.array(case.inputs["position_ids"].astype(np.int32))

    with mx.stream(mx.cpu):
        inputs_embeds = model.model.embed_tokens(input_ids)
        mask = model.model._prepare_attention_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
        )
        hidden = model.model(input_ids=input_ids, position_ids=position_ids)[0]
        logits = model.compute_logits(hidden)
        mx.eval(mask, hidden, logits)

    assert np.array_equal(_visible(mask), case.expected["attention_mask_visible"])
    assert_parity(np.array(hidden), case.expected["hidden_states"], atol=1e-4, rtol=1e-4, name="qwen2 hidden")
    assert_parity(np.array(logits), case.expected["logits"], atol=1e-4, rtol=1e-4, name="qwen2 logits")
