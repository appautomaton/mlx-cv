import json
from pathlib import Path


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
    assert status["claim_level"] == "contract_skeleton_only"


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

