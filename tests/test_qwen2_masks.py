import numpy as np
import mlx.core as mx

from mlx_cv.backbones.llm.qwen2.masks import (
    create_block_diff_mask_by_pe_4d,
    find_prefix_seq_length_by_pe,
    make_causal_mask_4d,
    update_causal_mask_for_one_gen_window_2d,
    update_causal_mask_with_pad_non_visible_2d,
)


TEXT_MASK = 151676


def _visible(mask) -> np.ndarray:
    return np.isfinite(np.array(mask))


def test_base_causal_mask_4d_no_cache_and_with_past_width():
    mask = make_causal_mask_4d(batch_size=2, query_length=3)
    expected = np.array(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )

    assert mask.shape == (2, 1, 3, 3)
    assert np.array_equal(_visible(mask)[0, 0], expected)
    assert np.array_equal(_visible(mask)[1, 0], expected)

    cached = make_causal_mask_4d(batch_size=1, query_length=2, key_value_length=5)
    assert np.array_equal(_visible(cached)[0, 0], np.array([[True, True, True, True, False], [True, True, True, True, True]]))


def test_find_prefix_seq_length_by_position_id_drop():
    pe = mx.array([[0, 1, 2, 0, 1], [0, 1, 2, 3, 4]], dtype=mx.int32)
    assert np.array_equal(np.array(find_prefix_seq_length_by_pe(pe)), np.array([3, -1], dtype=np.int32))


def test_pad_non_visible_mask_matches_hand_expected_noncausal_and_causal():
    input_ids = mx.array([10, TEXT_MASK, TEXT_MASK, 20, 30], dtype=mx.int32)
    base = make_causal_mask_4d(1, 5)[0, 0]

    noncausal = update_causal_mask_with_pad_non_visible_2d(
        input_ids,
        base,
        TEXT_MASK,
        causal_attn=False,
    )
    expected_noncausal = np.array(
        [
            [True, False, False, False, False],
            [True, True, True, False, False],
            [True, True, True, False, False],
            [True, False, False, True, False],
            [True, False, False, True, True],
        ]
    )
    assert np.array_equal(_visible(noncausal), expected_noncausal)

    causal = update_causal_mask_with_pad_non_visible_2d(
        input_ids,
        base,
        TEXT_MASK,
        causal_attn=True,
    )
    expected_causal = np.array(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, False, False, True, False],
            [True, False, False, True, True],
        ]
    )
    assert np.array_equal(_visible(causal), expected_causal)


def test_one_generation_window_mask_matches_hand_expected():
    input_ids = mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32)
    base = make_causal_mask_4d(1, 6)[0, 0]

    noncausal = update_causal_mask_for_one_gen_window_2d(
        input_ids,
        base,
        block_size=2,
        use_cache=True,
        causal_attn=False,
    )
    expected_noncausal = np.array(
        [
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, False, True, True],
            [True, True, True, False, True, True],
        ]
    )
    assert np.array_equal(_visible(noncausal), expected_noncausal)

    causal = update_causal_mask_for_one_gen_window_2d(
        input_ids,
        base,
        block_size=2,
        use_cache=True,
        causal_attn=True,
    )
    expected_causal = np.array(
        [
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, False, True, False],
            [True, True, True, False, True, True],
        ]
    )
    assert np.array_equal(_visible(causal), expected_causal)


def test_block_diff_mask_by_pe_matches_hand_expected():
    position_ids = mx.array([[0, 1, 1, 2, 0, 1]], dtype=mx.int32)
    x0_len = mx.array([2], dtype=mx.int32)

    additive, visible = create_block_diff_mask_by_pe_4d(
        block_size=2,
        x0_len_list=x0_len,
        position_ids=position_ids,
        causal_attn=False,
    )
    expected = np.array(
        [
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],
            [True, False, True, True, False, False],
            [True, False, True, True, False, False],
            [False, False, False, False, True, True],
            [False, False, False, False, True, True],
        ]
    )
    assert np.array_equal(np.array(visible)[0, 0], expected)
    assert np.array_equal(_visible(additive)[0, 0], expected)

    _, causal_visible = create_block_diff_mask_by_pe_4d(
        block_size=2,
        x0_len_list=x0_len,
        position_ids=position_ids,
        causal_attn=True,
    )
    expected_causal = np.array(
        [
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],
            [True, False, True, False, False, False],
            [True, False, True, True, False, False],
            [False, False, False, False, True, False],
            [False, False, False, False, True, True],
        ]
    )
    assert np.array_equal(np.array(causal_visible)[0, 0], expected_causal)
