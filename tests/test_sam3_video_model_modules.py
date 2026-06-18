import numpy as np
import pytest
import mlx.core as mx

from mlx_cv.models.sam3 import (
    SAM3MultiplexController,
    SAM3MultiplexDecoderConfig,
    SAM3MultiplexMaskDecoder,
    SAM3VideoConfig,
    SAM3VideoModel,
    SAM3VideoMultiplexTrackerCore,
    bucket_features_to_object_space,
    build_multiplex_memory_mask_input,
    mask_logits_for_memory,
)
from mlx_cv.models.sam3.multiplex_state import REMOVED_OBJECT_INDEX


def _cfg():
    return SAM3VideoConfig.tiny_fixture()


def _state():
    return SAM3MultiplexController(2).get_state(3, object_ids=[10, 11, 12])


def _fixture_features():
    mx.random.seed(7)
    image_features = mx.random.normal((4, 1, 16))
    image_pos_enc = mx.random.normal((4, 1, 16))
    high_res = (
        mx.random.normal((2, 16, 8, 8)),
        mx.random.normal((2, 16, 4, 4)),
    )
    return image_features, image_pos_enc, high_res


def _mask_inputs():
    masks = np.zeros((3, 1, 32, 32), dtype=np.float32)
    masks[0, :, 2:12, 3:14] = 1.0
    masks[1, :, 8:22, 9:24] = 1.0
    masks[2, :, 16:30, 4:18] = 1.0
    return mx.array(masks)


def _assert_finite(*arrays):
    mx.eval(*arrays)
    for arr in arrays:
        assert np.isfinite(np.asarray(arr)).all()


def test_sam3_tracker_multiplex_state_mux_demux_padding_and_bookkeeping():
    state = _state()

    assert state.assignments == [[0, 1], [2, -1]]
    assert state.object_ids == (10, 11, 12)
    assert np.asarray(state.valid_mask).tolist() == [[True, True], [True, False]]

    probe = mx.arange(3, dtype=mx.float32).reshape(3, 1)
    muxed = state.mux(probe)
    demuxed = state.demux(muxed)
    mx.eval(muxed, demuxed)

    assert muxed.shape == (2, 2, 1)
    assert np.asarray(muxed)[1, 1, 0] == 0.0
    assert np.asarray(demuxed).tolist() == [[0.0], [1.0], [2.0]]

    poisoned = np.asarray(muxed).copy()
    poisoned[1, 1, 0] = 999.0
    assert np.asarray(state.demux(mx.array(poisoned))).tolist() == [[0.0], [1.0], [2.0]]

    state.remove_object_id(11)
    assert state.assignments == [[0, REMOVED_OBJECT_INDEX], [1, -1]]
    assert state.object_ids == (10, 12)
    assert np.asarray(state.valid_mask).tolist() == [[True, False], [True, False]]

    new_index = state.add_object_id(13)
    assert new_index == 2
    assert state.assignments == [[0, REMOVED_OBJECT_INDEX], [1, 2]]
    assert state.object_ids == (10, 12, 13)
    assert state.available_slots == 0


def test_sam3_multiplex_state_respects_eval_capacity_below_physical_width():
    state = SAM3MultiplexController(4, eval_multiplex_count=2).get_state(3, object_ids=[10, 11, 12])

    assert state.assignments == [[0, 1, -1, -1], [2, -1, -1, -1]]
    assert state.allowed_bucket_capacity == 2
    assert state.available_slots == 1
    assert np.asarray(state.valid_mask).tolist() == [
        [True, True, False, False],
        [True, False, False, False],
    ]

    probe = mx.arange(3, dtype=mx.float32).reshape(3, 1)
    muxed = state.mux(probe)
    demuxed = state.demux(muxed)
    mx.eval(muxed, demuxed)

    assert muxed.shape == (2, 4, 1)
    assert np.asarray(muxed)[:, 2:, :].sum() == 0.0
    assert np.asarray(demuxed).tolist() == [[0.0], [1.0], [2.0]]


def test_sam3_memory_encoder_builds_conditional_multiplex_channels():
    cfg = _cfg()
    state = _state()
    mx.random.seed(11)
    pix_feat = mx.random.normal((2, 16, 2, 2))
    mask_logits = mx.random.normal((3, 1, 32, 32))
    mask_for_mem = mask_logits_for_memory(mask_logits, scale=2.0, bias=-1.0)
    mask_input = build_multiplex_memory_mask_input(
        mask_for_mem,
        state,
        condition_as_mask_input=True,
        conditioning_objects=[0, 2],
    )

    assert mask_input.mask_for_mem_mux_space.shape == (2, 2, 32, 32)
    assert mask_input.condition_mask_channels.shape == (2, 2, 32, 32)
    assert mask_input.encoder_input_channels.shape == (2, 4, 32, 32)
    assert np.asarray(mask_input.condition_mask_channels)[1, 1].sum() == 0.0

    out = SAM3VideoMultiplexTrackerCore(cfg).memory_encoder(
        pix_feat,
        mask_input.encoder_input_channels,
        capture_taps=True,
    )
    object_space = bucket_features_to_object_space(out.features, state)
    mx.eval(out.features, out.pos_enc, object_space)

    assert out.features.shape == (2, 16, 2, 2)
    assert out.pos_enc.shape == (2, 16, 2, 2)
    assert object_space.shape == (3, 16, 2, 2)
    _assert_finite(out.features, out.pos_enc, object_space)


def test_sam3_memory_multiplex_decoder_returns_bucket_space_shapes():
    cfg = _cfg().decoder
    decoder = SAM3MultiplexMaskDecoder(cfg)
    image_embeddings, image_pe, high_res = _fixture_features()
    bucket_features = mx.broadcast_to(
        mx.transpose(image_embeddings, (1, 2, 0)).reshape(1, 16, 2, 2),
        (2, 16, 2, 2),
    )

    out = decoder(bucket_features, image_pe, high_res_features=high_res, capture_taps=True)
    mx.eval(out.masks, out.iou_pred, out.object_score_logits, out.sam_tokens_out)

    assert out.masks.shape == (2, 2, 1, 8, 8)
    assert out.iou_pred.shape == (2, 2, 1)
    assert out.object_score_logits.shape == (2, 2, 1)
    assert out.sam_tokens_out.shape == (2, 2, 1, 16)
    assert out.taps["mask_decoder.masks.bucket_space"].shape == (2, 2, 1, 8, 8)
    _assert_finite(out.masks, out.iou_pred, out.object_score_logits, out.sam_tokens_out)

    with pytest.raises(NotImplementedError, match="shared Object Multiplex"):
        SAM3MultiplexDecoderConfig(hidden_dim=16, multiplex_count=2, decode_mask_with_shared_tokens=True)


def test_sam3_tracker_core_mask_init_then_propagation_taps_and_shapes():
    cfg = _cfg()
    state = _state()
    tracker = SAM3VideoMultiplexTrackerCore(cfg)
    image_features, image_pos_enc, high_res = _fixture_features()
    output_dict = {}

    init_out = tracker.track_step(
        frame_index=0,
        image_features=image_features,
        image_pos_enc=image_pos_enc,
        high_res_features=high_res,
        mask_inputs=_mask_inputs(),
        is_init_cond_frame=True,
        output_dict=output_dict,
        multiplex_state=state,
        capture_taps=True,
    )
    prop_out = tracker.track_step(
        frame_index=1,
        image_features=image_features,
        image_pos_enc=image_pos_enc,
        high_res_features=high_res,
        output_dict=output_dict,
        multiplex_state=state,
        capture_taps=True,
    )
    mx.eval(
        init_out.high_res_masks,
        prop_out.low_res_masks,
        prop_out.high_res_masks,
        prop_out.obj_ptr_mux,
        prop_out.maskmem_features,
    )

    assert init_out.high_res_masks.shape == (3, 1, 32, 32)
    assert prop_out.low_res_masks.shape == (3, 1, 8, 8)
    assert prop_out.high_res_masks.shape == (3, 1, 32, 32)
    assert prop_out.obj_ptr.shape == (3, 16)
    assert prop_out.obj_ptr_mux.shape == (2, 2, 16)
    assert prop_out.maskmem_features.shape == (2, 16, 2, 2)
    assert output_dict["cond_frame_outputs"][0]["obj_ptr"].shape == (2, 2, 16)
    assert output_dict["non_cond_frame_outputs"][1]["maskmem_features"].shape == (2, 16, 2, 2)

    expected_taps = {
        "multiplex.assignments",
        "multiplex.valid_mask",
        "multiplex.object_ids",
        "multiplex.mux_probe",
        "multiplex.demux_probe",
        "tracker.backbone.top_features.image_space",
        "tracker.memory_conditioned_features.bucket_space",
        "mask_decoder.masks.bucket_space",
        "mask_decoder.low_res_masks.object_space",
        "mask_decoder.high_res_masks.object_space",
        "mask_decoder.iou_pred.bucket_space",
        "mask_decoder.iou_pred.object_space",
        "mask_decoder.object_score_logits.object_space",
        "tracker.obj_ptr.object_space",
        "tracker.obj_ptr.mux_space",
        "memory.mask_for_mem.object_space",
        "memory.mask_for_mem.mux_space",
        "memory.condition_mask_channels",
        "memory.encoder_input_channels",
        "memory.features.bucket_space",
        "memory.features.object_space",
        "memory.pos_enc.object_space",
        "memory.image_features",
        "memory.image_pos_enc",
    }
    assert expected_taps <= set(prop_out.taps)
    assert prop_out.taps["memory.encoder_input_channels"].shape == (2, 4, 32, 32)
    _assert_finite(prop_out.low_res_masks, prop_out.high_res_masks, prop_out.obj_ptr_mux)


def test_sam3_video_model_keeps_public_ids_and_output_taps_isolated():
    cfg = _cfg()
    model = SAM3VideoModel(cfg)
    state = model.init_multiplex_state([10, 11, 12])
    image_features, image_pos_enc, high_res = _fixture_features()

    out = model.track_step(
        frame_index=0,
        image_features=image_features,
        image_pos_enc=image_pos_enc,
        high_res_features=high_res,
        mask_inputs=_mask_inputs(),
        is_init_cond_frame=True,
        output_dict={},
        multiplex_state=state,
        capture_taps=True,
    )
    mx.eval(out.track_ids, out.masks_bool)

    assert np.asarray(out.track_ids).tolist() == [10, 11, 12]
    assert out.masks_bool.shape == (3, 32, 32)
    assert out.taps["output.track_ids"].shape == (3,)
    assert out.taps["output.multiplex"]["assignments"] == [[0, 1], [2, -1]]
    assert hasattr(model, "parameters")
    assert "tracker" in model.parameters()
