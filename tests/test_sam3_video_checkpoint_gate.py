import json
from pathlib import Path

import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.models.sam3 import (
    SAM3VideoConfig,
    SAM3VideoModel,
    convert_sam3_state_dict,
    convert_sam3_video_state_dict,
    inspect_sam3_video_state_dict,
    load_sam3_video_weights,
    remap_sam3_video_key,
)


STATUS_PATH = Path(
    ".agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json"
)
LOCAL_CONTRACT_STATUS_PATH = Path(".agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json")
CONTRACT_PATH = Path(".agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-contract.md")
RELEASE_PARITY_STATUS = Path(".agent/work/2026-06-16-release-parity-hardening/parity-status.json")


def _status():
    return json.loads(STATUS_PATH.read_text())


def test_sam3_video_status_records_checkpoint_admission_blocker():
    status = _status()

    assert status["schema_version"] == 1
    assert status["phase"] == "sam3-video-real-checkpoint-admission"
    assert status["model"] == "sam3_video"
    assert status["status"].startswith("BLOCKED:")
    assert status["checkpoint_env"] == "MLX_CV_SAM3_VIDEO_CHECKPOINT"
    assert status["config_env"] == "MLX_CV_SAM3_VIDEO_CONFIG"
    assert status["model_id_env"] == "MLX_CV_SAM3_VIDEO_MODEL_ID"
    assert status["cache_dir_env"] == "MLX_CV_SAM3_VIDEO_CACHE_DIR"
    assert status["required_gate_env"] == "MLX_CV_REQUIRE_SAM3_VIDEO_GATE"
    assert status["official_model_id"] == "facebook/sam3.1"
    assert status["checkpoint_name"] == "sam3.1_multiplex.pt"
    assert status["config_name"] == "config.json"
    assert status["source_url"] == "https://huggingface.co/facebook/sam3.1"
    assert "gated access" in status["license_or_terms"]
    assert status["reference_path"] == "references/sam3"
    assert status["blocked_reason"]
    assert status["claim_level"] == "external_blocker"
    assert status["local_contract_status"] == str(LOCAL_CONTRACT_STATUS_PATH)
    assert "not expanded for sam3_video" in status["release_parity_matrix"]


def test_sam3_video_contract_names_upstream_surfaces():
    contract = CONTRACT_PATH.read_text()
    status = _status()
    model_builder = Path("references/sam3/sam3/model_builder.py").read_text()
    base_predictor = Path("references/sam3/sam3/model/sam3_base_predictor.py").read_text()

    for name in status["reference_surfaces"]:
        assert name in contract

    assert "def build_sam3_video_predictor" in model_builder
    assert "def build_sam3_multiplex_video_predictor" in model_builder
    assert "def build_sam3_predictor" in model_builder
    assert "Sam3TrackerPredictor" in model_builder
    assert "SimpleMaskEncoder" in model_builder
    assert "MultiplexController" in model_builder
    assert "VideoTrackingDynamicMultiplex" in model_builder
    assert "def start_session" in base_predictor
    assert "def add_prompt" in base_predictor
    assert "def propagate_in_video" in base_predictor


def test_sam3_video_status_does_not_expand_release_parity_matrix():
    release_status = json.loads(RELEASE_PARITY_STATUS.read_text())
    assert set(release_status["models"]) == {
        "da3_multiview",
        "locateanything",
        "rfdetr",
        "sam3_image",
    }
    assert "sam3_video" not in release_status["models"]


def test_sam3_video_gate_recognizes_video_keys_without_image_loader_regression():
    state = {
        "tracker.maskmem_backbone.conv.weight": np.zeros((1,), dtype=np.float32),
        "detector.backbone.visual.weight": np.ones((1,), dtype=np.float32),
        "__config_json__": np.array('{"model": {"multiplex": true}}'),
    }

    inspected = inspect_sam3_video_state_dict(state)
    assert inspected["is_video_candidate"] is True
    assert "tracker" in inspected["matched_key_parts"]
    assert "maskmem" in inspected["matched_key_parts"]
    assert "multiplex" in inspected["matched_key_parts"]

    with pytest.raises(ValueError, match="video/tracker"):
        convert_sam3_state_dict(state)


def test_sam3_video_converter_maps_supported_reference_key_families():
    conv = np.arange(16 * 16 * 1 * 1, dtype=np.float32).reshape(16, 16, 1, 1)
    out = dict(
        convert_sam3_video_state_dict(
            {
                "tracker.maskmem_backbone.pix_feat_proj.weight": conv,
                "tracker.sam_mask_decoder.context_proj.bias": np.ones((16,), dtype=np.float32),
                "tracker.obj_ptr_proj.bias": np.ones((16,), dtype=np.float32) * 2,
            }
        )
    )

    assert remap_sam3_video_key("tracker.maskmem_backbone.pix_feat_proj.bias") == (
        "tracker.memory_encoder.pix_feat_proj.bias",
        True,
    )
    assert sorted(out) == [
        "tracker.mask_decoder.context_proj.bias",
        "tracker.memory_encoder.pix_feat_proj.weight",
        "tracker.obj_ptr_proj.bias",
    ]
    assert out["tracker.memory_encoder.pix_feat_proj.weight"].shape == (16, 1, 1, 16)
    np.testing.assert_array_equal(out["tracker.obj_ptr_proj.bias"], np.ones((16,), dtype=np.float32) * 2)


def test_sam3_video_converter_rejects_unsupported_key_families():
    with pytest.raises(ValueError, match="unsupported SAM3 video checkpoint key family"):
        convert_sam3_video_state_dict({"detector.backbone.weight": np.zeros((1,), dtype=np.float32)})

    with pytest.raises(ValueError, match="unsupported SAM3 video checkpoint key family"):
        convert_sam3_video_state_dict({"tracker.interactive_obj_ptr_proj.bias": np.zeros((16,), dtype=np.float32)})

    with pytest.raises(ValueError, match="unsupported SAM3 video checkpoint keys"):
        convert_sam3_video_state_dict({"unexpected.weight": np.zeros((1,), dtype=np.float32)})

    with pytest.raises(ValueError, match="duplicate SAM3 video checkpoint mapping"):
        convert_sam3_video_state_dict(
            {
                "tracker.obj_ptr_proj.bias": np.zeros((16,), dtype=np.float32),
                "model.tracker.obj_ptr_proj.bias": np.ones((16,), dtype=np.float32),
            }
        )


def test_load_sam3_video_weights_populates_tiny_video_model(tmp_path):
    model = SAM3VideoModel(SAM3VideoConfig.tiny_fixture())
    weights_path = tmp_path / "sam3_video_tiny.npz"
    np.savez(weights_path, **{"tracker.obj_ptr_proj.bias": np.ones((16,), dtype=np.float32) * 0.25})

    loaded = load_sam3_video_weights(model, weights_path)
    params = dict(tree_flatten(loaded.parameters()))
    np.testing.assert_allclose(np.asarray(params["tracker.obj_ptr_proj.bias"]), np.ones((16,)) * 0.25)


def test_load_sam3_video_weights_rejects_shape_mismatch(tmp_path):
    model = SAM3VideoModel(SAM3VideoConfig.tiny_fixture())
    weights_path = tmp_path / "bad_sam3_video.npz"
    np.savez(weights_path, **{"tracker.obj_ptr_proj.bias": np.zeros((15,), dtype=np.float32)})

    with pytest.raises(ValueError, match="expected \\(16,\\)"):
        load_sam3_video_weights(model, weights_path)
