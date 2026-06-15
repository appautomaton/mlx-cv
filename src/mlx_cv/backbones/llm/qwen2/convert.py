"""Qwen2 weight conversion/loading."""

from __future__ import annotations

import numpy as np

from ....hub.convert import Drop, convert_state_dict, load_into
from .modeling import Qwen2ForCausalLM

__all__ = ["QWEN2_CONVERT_RULES", "convert_qwen2_state_dict", "load_qwen2_weights"]


QWEN2_CONVERT_RULES = [
    Drop("lm_head.weight"),
    Drop("__versions_json__"),
    Drop("__config_json__"),
]


def _assert_tied_lm_head_is_lossless(state: dict[str, np.ndarray]) -> None:
    if "lm_head.weight" not in state:
        return
    if "model.embed_tokens.weight" not in state:
        raise ValueError("cannot drop lm_head.weight without model.embed_tokens.weight")
    if not np.array_equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
        raise ValueError("cannot drop lm_head.weight because it is not tied to model.embed_tokens.weight")


def convert_qwen2_state_dict(state: dict[str, np.ndarray]):
    """Map reference Qwen2 weights to local paths, dropping tied ``lm_head``."""
    _assert_tied_lm_head_is_lossless(state)
    return convert_state_dict(state, QWEN2_CONVERT_RULES)


def load_qwen2_weights(model: Qwen2ForCausalLM, weights_path) -> Qwen2ForCausalLM:
    """Load a minted Qwen2 ``*_weights.npz`` into ``model`` in place; returns it."""
    npz = np.load(weights_path, allow_pickle=False)
    state = {k: npz[k] for k in npz.files}
    _assert_tied_lm_head_is_lossless(state)
    return load_into(model, state, QWEN2_CONVERT_RULES)
