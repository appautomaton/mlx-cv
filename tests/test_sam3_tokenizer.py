import numpy as np

from mlx_cv.models.sam3 import SAM3Tokenizer, canonicalize_text, default_bpe_path


def test_sam3_tokenizer_uses_committed_reduced_bpe_asset():
    path = default_bpe_path()
    assert path.exists()
    assert path.name == "bpe_simple_vocab_tiny.txt"


def test_sam3_tokenizer_canonical_string_to_token_ids():
    tokenizer = SAM3Tokenizer(context_length=8)
    token_ids = tokenizer("Cat!", context_length=8)

    np.testing.assert_array_equal(token_ids, [[512, 66, 64, 339, 513, 0, 0, 0]])
    assert tokenizer.sot_token_id == 512
    assert tokenizer.eot_token_id == 513
    assert tokenizer.vocab_size == 514


def test_sam3_tokenizer_canonicalizes_like_clip_path_without_reference_helpers():
    assert canonicalize_text("  Red_car!!! ") == "red car"
    tokenizer = SAM3Tokenizer(context_length=10)
    np.testing.assert_array_equal(
        tokenizer("red car", context_length=10),
        [[512, 81, 68, 323, 66, 64, 337, 513, 0, 0]],
    )
