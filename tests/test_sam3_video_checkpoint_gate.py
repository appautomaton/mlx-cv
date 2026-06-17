import json
from pathlib import Path

import numpy as np
import pytest

from mlx_cv.models.sam3 import convert_sam3_state_dict, inspect_sam3_video_state_dict


STATUS_PATH = Path(".agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json")
CONTRACT_PATH = Path(".agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-contract.md")
RELEASE_PARITY_STATUS = Path(".agent/work/2026-06-16-release-parity-hardening/parity-status.json")


def _status():
    return json.loads(STATUS_PATH.read_text())


def test_sam3_video_status_records_phase_local_blocker():
    status = _status()

    assert status["phase"] == "sam3-video-object-multiplex"
    assert status["model"] == "sam3_video"
    assert status["status"].startswith("BLOCKED:")
    assert status["checkpoint_env"] == "MLX_CV_SAM3_VIDEO_CHECKPOINT"
    assert status["config_env"] == "MLX_CV_SAM3_VIDEO_CONFIG"
    assert status["model_id_env"] == "MLX_CV_SAM3_VIDEO_MODEL_ID"
    assert status["reference_path"] == "references/sam3"
    assert status["blocked_reason"]
    assert status["claim_level"] == "external_blocker"


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
