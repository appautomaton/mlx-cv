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


da3_checkpoint = _load_tool("da3_checkpoint", "tools/da3_checkpoint.py")
da3_contract = _load_tool("da3_real_architecture_contract", "tools/da3_real_architecture_contract.py")


def _checkpoint_for_real_audit(*, environ=None, cache_root=None):
    required = da3_checkpoint.required_gate_enabled(environ)
    try:
        checkpoint = da3_checkpoint.resolve_da3_checkpoint(
            environ=environ,
            cache_root=cache_root,
            required=required,
        )
    except da3_checkpoint.DA3CheckpointError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    if checkpoint is None:
        if required:
            pytest.fail("DA3 checkpoint not configured")
        pytest.skip("DA3 checkpoint not configured")
    return checkpoint


def test_optional_no_checkpoint_real_architecture_audit_skips_cleanly(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="DA3 checkpoint not configured"):
        _checkpoint_for_real_audit(environ={}, cache_root=tmp_path)


def test_required_no_checkpoint_real_architecture_audit_fails_instead_of_skipping(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="checkpoint is required but missing"):
        _checkpoint_for_real_audit(
            environ={da3_checkpoint.DA3_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
        )


def test_required_missing_config_real_architecture_audit_fails_instead_of_skipping(tmp_path):
    model_dir = da3_checkpoint.model_cache_dir(tmp_path, da3_checkpoint.DA3_DEFAULT_MODEL_ID)
    model_dir.mkdir(parents=True)
    (model_dir / da3_checkpoint.DA3_CHECKPOINT_FILENAME).write_bytes(b"fake weights")

    with pytest.raises(pytest.fail.Exception, match="config file is missing"):
        _checkpoint_for_real_audit(
            environ={da3_checkpoint.DA3_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
        )


def test_real_architecture_contract_rejects_missing_provenance(tmp_path):
    config = tmp_path / da3_checkpoint.DA3_CONFIG_FILENAME
    checkpoint = tmp_path / da3_checkpoint.DA3_CHECKPOINT_FILENAME
    config.write_text(
        """
        {
          "model_name": "da3-small",
          "config": {
            "__object__": {"path": "depth_anything_3.model.da3", "name": "DepthAnything3Net"},
            "net": {
              "__object__": {"path": "depth_anything_3.model.dinov2.dinov2", "name": "DinoV2"},
              "name": "vits",
              "out_layers": [5, 7, 9, 11],
              "alt_start": 4,
              "qknorm_start": 4,
              "rope_start": 4,
              "cat_token": true
            },
            "head": {
              "__object__": {"path": "depth_anything_3.model.dualdpt", "name": "DualDPT"},
              "dim_in": 768,
              "output_dim": 2,
              "features": 64,
              "out_channels": [48, 96, 192, 384]
            },
            "cam_enc": {
              "__object__": {"path": "depth_anything_3.model.cam_enc", "name": "CameraEnc"},
              "dim_out": 384
            },
            "cam_dec": {
              "__object__": {"path": "depth_anything_3.model.cam_dec", "name": "CameraDec"},
              "dim_in": 768
            }
          }
        }
        """
    )
    checkpoint.write_bytes(b"fake weights")
    info = da3_checkpoint.DA3CheckpointInfo(
        model_id=da3_checkpoint.DA3_DEFAULT_MODEL_ID,
        checkpoint_path=checkpoint,
        config_path=config,
        checkpoint_sha256="",
        config_sha256="config-sha",
        checkpoint_url="https://example.test/model.safetensors",
        config_url="https://example.test/config.json",
        revision="main",
        license_note="Apache-2.0",
        source="test",
    )

    with pytest.raises(da3_contract.DA3ArchitectureContractError, match="provenance"):
        da3_contract.audit_da3_real_architecture_contract(info)


def test_real_da3_checkpoint_records_architecture_contract():
    checkpoint = _checkpoint_for_real_audit()
    required = da3_checkpoint.required_gate_enabled()
    try:
        contract = da3_contract.audit_da3_real_architecture_contract(checkpoint)
    except da3_contract.DA3ArchitectureContractDependencyError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))

    assert contract.model_id == da3_checkpoint.DA3_DEFAULT_MODEL_ID
    assert contract.provenance["checkpoint_sha256"] == checkpoint.checkpoint_sha256
    assert contract.provenance["config_sha256"] == checkpoint.config_sha256
    assert contract.provenance["revision"] == "main"
    assert contract.provenance["license_note"] == "Apache-2.0"

    arch = contract.architecture
    assert arch["model_id"] == da3_checkpoint.DA3_DEFAULT_MODEL_ID
    assert arch["model_name"] == "da3-small"
    assert arch["dinov2"]["variant"] in {"vits", "vitb"}
    assert arch["dinov2"]["variant"] == "vits"
    assert arch["dinov2"]["embed_dim"] == 384
    assert arch["dinov2"]["out_layers"] == (5, 7, 9, 11)
    assert arch["dinov2"]["alt_start"] == 4
    assert arch["dinov2"]["qknorm_start"] == 4
    assert arch["dinov2"]["rope_start"] == 4
    assert arch["dinov2"]["cat_token"] is True
    assert arch["dinov2"]["head_input_dim"] == arch["dinov2"]["embed_dim"] * 2
    assert arch["dinov2"]["qknorm_layers"] == tuple(range(4, 12))
    assert arch["dinov2"]["rope_layers"] == tuple(range(4, 12))
    assert arch["dinov2"]["global_attention_layers"] == (5, 7, 9, 11)

    assert arch["dualdpt"] == {
        "dim_in": 768,
        "dim_in_rule": "embed_dim*2",
        "output_dim": 2,
        "aux_output_dim": 7,
        "features": 64,
        "out_channels": (48, 96, 192, 384),
        "head_names": ("depth", "ray"),
        "intermediate_layer_idx": (0, 1, 2, 3),
        "aux_pyramid_levels": 4,
    }
    assert arch["camera_encoder"] == {
        "dim_in": 9,
        "dim_out": 384,
        "target_dim": 9,
        "trunk_depth": 4,
        "pose_branch_hidden_dim": 192,
    }
    assert arch["camera_decoder"] == {
        "dim_in": 768,
        "translation_dim": 3,
        "quaternion_dim": 4,
        "fov_dim": 2,
        "pose_encoding_dim": 9,
    }
    assert "depth_anything_3.model.utils.transform.pose_encoding_to_extri_intri" in arch[
        "camera_pose_utility_dependencies"
    ]
    assert "depth_anything_3.model.utils.transform.extri_intri_to_pose_encoding" in arch[
        "camera_pose_utility_dependencies"
    ]
    assert "depth_anything_3.utils.geometry.affine_inverse" in arch["camera_pose_utility_dependencies"]
    assert any("Gaussian splatting" in branch for branch in arch["unsupported_branches"])
    assert any("NestedDepthAnything3Net" in branch for branch in arch["unsupported_branches"])

    assert set(contract.tensor_groups) == set(da3_contract.TENSOR_GROUP_NAMES)
    assert len(contract.tensor_groups["backbone"]) == 207
    assert len(contract.tensor_groups["dualdpt"]) == 156
    assert len(contract.tensor_groups["camera_encoder"]) == 64
    assert len(contract.tensor_groups["camera_decoder"]) == 10
    assert contract.tensor_groups["excluded_branches"] == ()
    assert "backbone.pretrained.camera_token" in contract.tensor_groups["backbone"]
    assert "head.scratch.output_conv2_aux.3.5.weight" in contract.tensor_groups["dualdpt"]
    assert "head.scratch.output_conv2_aux.0.2.weight" in contract.required_tensor_groups["dualdpt"]
    assert "head.scratch.output_conv2_aux.0.2.bias" in contract.required_tensor_groups["dualdpt"]
    assert "cam_enc.pose_branch.fc1.weight" in contract.tensor_groups["camera_encoder"]
    assert "cam_dec.fc_qvec.weight" in contract.tensor_groups["camera_decoder"]

    assert contract.required_tensor_groups["backbone"]
    assert contract.required_tensor_groups["dualdpt"]
    assert contract.required_tensor_groups["camera_encoder"]
    assert contract.required_tensor_groups["camera_decoder"]

    shapes = contract.tensor_shapes
    assert shapes["backbone.pretrained.camera_token"] == (1, 2, 384)
    assert shapes["backbone.pretrained.blocks.4.attn.q_norm.weight"] == (64,)
    assert "backbone.pretrained.blocks.3.attn.q_norm.weight" not in shapes
    assert shapes["head.norm.weight"] == (768,)
    assert shapes["head.projects.0.weight"] == (48, 768, 1, 1)
    assert shapes["head.projects.3.weight"] == (384, 768, 1, 1)
    assert shapes["head.scratch.output_conv2.2.weight"] == (2, 32, 1, 1)
    assert shapes["head.scratch.output_conv2_aux.0.2.weight"] == (32,)
    assert shapes["head.scratch.output_conv2_aux.0.2.bias"] == (32,)
    assert shapes["head.scratch.output_conv2_aux.3.5.weight"] == (7, 32, 1, 1)
    assert shapes["cam_enc.pose_branch.fc1.weight"] == (192, 9)
    assert shapes["cam_enc.trunk.3.attn.qkv.weight"] == (1152, 384)
    assert shapes["cam_dec.backbone.0.weight"] == (768, 768)
    assert shapes["cam_dec.fc_t.weight"] == (3, 768)
    assert shapes["cam_dec.fc_qvec.weight"] == (4, 768)
    assert shapes["cam_dec.fc_fov.0.weight"] == (2, 768)

    summary = contract.group_summary
    assert summary["state_key_count"] == 437
    assert summary["required_tensor_count"] == 437
    assert summary["group_counts"] == {
        "backbone": 207,
        "dualdpt": 156,
        "camera_encoder": 64,
        "camera_decoder": 10,
        "excluded_branches": 0,
    }
    assert summary["backbone_block_indexes"] == tuple(range(12))
    assert summary["selected_out_layers"] == (5, 7, 9, 11)
    assert summary["qknorm_layer_indexes"] == tuple(range(4, 12))
    assert summary["dualdpt_project_channels"] == (48, 96, 192, 384)
    assert summary["camera_token_shape"] == (1, 2, 384)
    assert summary["dualdpt_head_input_shape"] == (768,)
    assert summary["camera_encoder_trunk_indexes"] == tuple(range(4))
    assert summary["camera_decoder_pose_head_shapes"] == {
        "translation": (3, 768),
        "quaternion": (4, 768),
        "fov": (2, 768),
    }

    gaps = contract.local_monocular_gaps
    assert any("single NCHW" in gap for gap in gaps)
    assert any("any-view DINOv2 behavior" in gap for gap in gaps)
    assert any("DualDPT" in gap for gap in gaps)
    assert any("camera tokens" in gap for gap in gaps)
    assert any("camera geometry modules" in gap for gap in gaps)
    assert any("pose conversion utilities" in gap for gap in gaps)
    assert any("cam_enc.* and cam_dec.* tensors are unmapped" in gap for gap in gaps)
    assert contract.as_dict()["local_monocular_gaps"] == gaps
