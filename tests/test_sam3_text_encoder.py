import numpy as np
import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mlx_cv.models.sam3 import SAM3TextConfig, SAM3TextEncoder, SAM3Tokenizer


def test_sam3_text_encoder_constructs_from_typed_config_and_tiny_weights():
    tokenizer = SAM3Tokenizer(context_length=8)
    cfg = SAM3TextConfig(
        d_model=6,
        context_length=8,
        vocab_size=tokenizer.vocab_size,
        width=8,
        heads=2,
        layers=1,
        mlp_ratio=2.0,
    )
    encoder = SAM3TextEncoder(cfg, tokenizer=tokenizer)
    tiny_weights = []
    for key, value in tree_flatten(encoder.parameters()):
        arr = np.arange(np.prod(value.shape), dtype=np.float32).reshape(value.shape)
        arr = (arr % 11 - 5) / 50.0
        tiny_weights.append((key, mx.array(arr, dtype=value.dtype)))
    encoder.update(tree_unflatten(tiny_weights))

    out = encoder(["cat"])
    mx.eval(out.language_features, out.language_mask, out.language_embeds, out.token_ids)

    assert out.token_ids.shape == (1, 8)
    assert out.language_mask.shape == (1, 8)
    assert out.language_features.shape == (8, 1, 6)
    assert out.language_embeds.shape == (8, 1, 8)
    assert bool(np.array(out.language_mask)[0, -1])
    assert not bool(np.array(out.language_mask)[0, 0])
    assert np.isfinite(np.array(out.language_features)).all()


def test_sam3_text_encoder_rejects_oversized_token_ids():
    cfg = SAM3TextConfig(d_model=4, context_length=4, vocab_size=8, width=4, heads=1, layers=1)
    encoder = SAM3TextEncoder(cfg)
    with pytest.raises(ValueError, match="vocab_size"):
        encoder(mx.array([[9, 0, 0, 0]], dtype=mx.int32))
