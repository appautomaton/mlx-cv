from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _load_tool(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rfdetr_checkpoint = _load_tool("rfdetr_checkpoint", "tools/rfdetr_checkpoint.py")
rfdetr_contract = _load_tool("rfdetr_real_architecture_contract", "tools/rfdetr_real_architecture_contract.py")


def _checkpoint_for_real_audit(*, environ=None, cache_root=None):
    required = rfdetr_checkpoint.required_gate_enabled(environ)
    try:
        checkpoint = rfdetr_checkpoint.resolve_rfdetr_nano_checkpoint(
            environ=environ,
            cache_root=cache_root,
            required=required,
        )
    except rfdetr_checkpoint.CheckpointError as exc:
        if required:
            pytest.fail(str(exc))
        raise
    if checkpoint is None:
        pytest.skip("RF-DETR Nano checkpoint not configured")
    return checkpoint


def test_optional_no_checkpoint_real_audit_skips_cleanly(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="checkpoint not configured"):
        _checkpoint_for_real_audit(environ={}, cache_root=tmp_path)


def test_required_no_checkpoint_real_audit_fails_instead_of_skipping(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="checkpoint is required but missing"):
        _checkpoint_for_real_audit(
            environ={rfdetr_checkpoint.RFDETR_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
        )


def test_required_checksum_mismatch_fails_instead_of_skipping(tmp_path):
    checkpoint = tmp_path / rfdetr_checkpoint.RFDETR_NANO_CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"not the verified RF-DETR Nano checkpoint")

    with pytest.raises(pytest.fail.Exception, match="expected"):
        _checkpoint_for_real_audit(
            environ={
                rfdetr_checkpoint.RFDETR_REQUIRED_GATE_ENV: "1",
                rfdetr_checkpoint.RFDETR_NANO_CHECKPOINT_ENV: str(checkpoint),
            },
        )


def test_real_rfdetr_nano_checkpoint_records_architecture_contract():
    checkpoint = _checkpoint_for_real_audit()
    required = rfdetr_checkpoint.required_gate_enabled()
    try:
        contract = rfdetr_contract.audit_rfdetr_nano_checkpoint(checkpoint)
    except rfdetr_contract.ArchitectureContractDependencyError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))

    arch = contract.architecture
    assert arch["out_feature_indexes"] == (3, 6, 9, 12)
    assert arch["local_zero_based_layers"] == (2, 5, 8, 11)
    assert arch["projector_scale"] == ("P4",)
    assert arch["dec_layers"] == 2
    assert arch["group_detr"] == 13
    assert arch["bbox_reparam"] is True
    assert arch["lite_refpoint_refine"] is True
    assert arch["num_feature_levels"] == 1
    assert arch["two_stage"] is True
    assert arch["encoder"] == "dinov2_windowed_small"

    assert contract.checkpoint_class_head_shape == (91, 256)
    assert set(contract.required_tensor_groups) == set(rfdetr_contract.REQUIRED_TENSOR_GROUP_NAMES)
    assert contract.tensor_shapes["class_embed.weight"] == (91, 256)
    assert contract.tensor_shapes["query_feat.weight"] == (3900, 256)
    assert contract.tensor_shapes["refpoint_embed.weight"] == (3900, 4)
    assert contract.tensor_shapes["backbone.0.projector.stages.0.0.cv1.conv.weight"] == (256, 1536, 1, 1)

    assert contract.group_summary["selected_windowed_dinov2_layers"] == (2, 5, 8, 11)
    assert contract.group_summary["windowed_dinov2_layer_indexes"] == tuple(range(12))
    assert contract.group_summary["projector_stage_indexes"] == (0,)
    assert contract.group_summary["two_stage_group_indexes"] == tuple(range(13))
    assert contract.group_summary["decoder_layer_indexes"] == (0, 1)
    assert contract.group_summary["base_query_count"] == 300
    assert contract.group_summary["grouped_query_count"] == 3900

    groups = contract.required_tensor_groups
    assert "backbone.0.encoder.encoder.encoder.layer.2.attention.attention.query.weight" in groups["windowed_dinov2"]
    assert "backbone.0.projector.stages.0.0.m.2.cv2.conv.weight" in groups["p4_c2f_projector"]
    assert "transformer.enc_out_class_embed.12.weight" in groups["two_stage_encoder_proposal_heads"]
    assert "transformer.enc_out_bbox_embed.12.layers.2.bias" in groups["two_stage_encoder_proposal_heads"]
    assert "transformer.decoder.layers.1.self_attn.in_proj_weight" in groups[
        "decoder_self_attention_norm_refpoint_head"
    ]
    assert "transformer.decoder.ref_point_head.layers.1.weight" in groups[
        "decoder_self_attention_norm_refpoint_head"
    ]
    assert groups["grouped_query_slicing"] == ("query_feat.weight", "refpoint_embed.weight")
    assert "bbox_embed.layers.2.weight" in groups["detection_head"]

    gaps = contract.local_fixture_gaps
    assert len(gaps) >= 6
    assert any("out_layers" in gap for gap in gaps)
    assert any("feature levels" in gap for gap in gaps)
    assert any("decoder layers" in gap for gap in gaps)
    assert any("grouped queries" in gap for gap in gaps)
    assert any("class head rows" in gap for gap in gaps)
    assert contract.as_dict()["local_fixture_gaps"] == gaps
