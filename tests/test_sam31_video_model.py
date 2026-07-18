from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_cv.models.sam3.sam31_convert import (
    convert_sam31_tracker_state_dict,
    map_sam31_tracker_key,
)
from mlx_cv.models.sam3.sam31_tracker import SAM31MultiplexTracker
from mlx_cv.models.sam3.sam31_processor import SAM3VideoProcessor, SAM3VideoProcessorConfig
from mlx_cv.models.sam3.sam31_session import SAM3VideoSession, _Memory


def test_sam31_tracker_has_the_exact_official_parameter_count():
    params = dict(tree_flatten(SAM31MultiplexTracker().parameters()))

    assert len(params) == 457
    assert params["maskmem_tpos_enc"].shape == (7, 1, 1, 256)
    assert params["sam_mask_decoder.iou_token.weight"].shape == (16, 256)
    assert params["sam_mask_decoder.mask_tokens.weight"].shape == (48, 256)


def test_sam31_tracker_key_mapping_covers_custom_layouts():
    cases = {
        "tracker.model.interactive_sam_prompt_encoder.mask_downscaling.3.weight":
            "interactive_sam_prompt_encoder.mask_downscaling.conv2.weight",
        "tracker.model.sam_mask_decoder.transformer.layers.0.self_attn.out_proj.weight":
            "sam_mask_decoder.transformer.layers.0.self_attn.o_proj.weight",
        "tracker.model.sam_mask_decoder.transformer.layers.0.norm1.weight":
            "sam_mask_decoder.transformer.layers.0.layer_norm1.weight",
        "tracker.model.sam_mask_decoder.transformer.layers.0.mlp.lin1.weight":
            "sam_mask_decoder.transformer.layers.0.mlp.proj_in.weight",
        "tracker.model.sam_mask_decoder.transformer.norm_final_attn.weight":
            "sam_mask_decoder.transformer.layer_norm_final_attn.weight",
        "tracker.model.sam_mask_decoder.output_upscaling.0.weight":
            "sam_mask_decoder.upscale_conv1.weight",
        "tracker.model.sam_mask_decoder.output_hypernetworks_mlps.0.layers.2.weight":
            "sam_mask_decoder.output_hypernetworks_mlps.0.proj_out.weight",
        "tracker.model.obj_ptr_proj.layers.1.weight":
            "obj_ptr_proj.layers.0.weight",
    }
    for source, target in cases.items():
        assert map_sam31_tracker_key(source) == (target,)


def test_sam31_tracker_parameter_tree_matches_real_checkpoint_shapes():
    try:
        import torch
    except ModuleNotFoundError:
        return

    checkpoint = "models/sam3-video/upstream/sam3.1_multiplex.pt"
    try:
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
    except FileNotFoundError:
        return

    converted = convert_sam31_tracker_state_dict(state)
    expected = {key: value.shape for key, value in converted.items()}
    actual = {
        key: tuple(value.shape)
        for key, value in tree_flatten(SAM31MultiplexTracker().parameters())
    }

    assert len(converted) == 457
    assert actual == expected


def test_sam31_multiplex_decoder_preserves_bucket_and_object_axes():
    tracker = SAM31MultiplexTracker()
    decoder = tracker.sam_mask_decoder
    image = mx.zeros((1, 2, 2, 256), dtype=mx.float32)
    position = mx.zeros_like(image)
    high_resolution = [
        mx.zeros((1, 8, 8, 32), dtype=mx.float32),
        mx.zeros((1, 4, 4, 64), dtype=mx.float32),
    ]
    suppression = mx.zeros((1, 16, 256), dtype=mx.float32)

    output = decoder(
        image,
        position,
        high_resolution,
        suppression,
        multimask_output=True,
    )
    mx.eval(output)

    assert output["masks"].shape == (1, 16, 3, 8, 8)
    assert output["iou_pred"].shape == (1, 16, 3)
    assert output["object_score_logits"].shape == (1, 16, 1)
    assert output["sam_tokens_out"].shape == (1, 16, 3, 256)


def test_sam31_session_api_handles_dynamic_buckets_remove_and_reset():
    processor = SAM3VideoProcessor(
        SAM3VideoProcessorConfig(
            image_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
    )
    manager = SAM3VideoSession(model=object(), processor=processor)
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    session = manager.start_session(frames=frames, session_id="test")
    for object_id in range(1, 18):
        manager.add_prompt(
            "test", frame_index=0, object_id=object_id, box=(1, 1, 6, 6)
        )

    multiplex = manager._state(session)
    assert multiplex.num_buckets == 2
    assert multiplex.assignments[0] == list(range(16))
    assert multiplex.assignments[1][0] == 16

    manager.remove_object("test", 5)
    assert 5 not in session.active_object_ids
    manager.reset_session("test")
    assert session.active_object_ids == []
    assert session.prompts == {}


def test_sam31_propagation_runs_memory_attention_and_demuxes_objects():
    tracker = SAM31MultiplexTracker()

    class _Model:
        pass

    model = _Model()
    model.tracker = tracker
    manager = SAM3VideoSession(model=model)
    mux = manager.controller.get_state(1, object_ids=[7])

    class _Vision:
        pass

    vision = _Vision()
    vision.propagation_hidden_states = (
        mx.zeros((1, 8, 8, 256)),
        mx.zeros((1, 4, 4, 256)),
        mx.zeros((1, 2, 2, 256)),
    )
    vision.propagation_position_encoding = (
        mx.zeros((1, 8, 8, 256)),
        mx.zeros((1, 4, 4, 256)),
        mx.zeros((1, 2, 2, 256)),
    )
    memory = _Memory(
        0,
        mx.zeros((1, 2, 2, 256)),
        mx.zeros((1, 2, 2, 256)),
        mx.zeros((1, 2, 2, 256)),
        mx.zeros((1, 2, 2, 256)),
        mx.zeros((1, 16, 256)),
    )

    masks, ious, logits, pointers = manager._propagation_outputs(
        vision, mux, [memory]
    )
    mx.eval(masks, ious, logits, pointers)

    assert masks.shape == (1, 8, 8)
    assert ious.shape == (1,)
    assert logits.shape == (1,)
    assert pointers.shape == (1, 256)
