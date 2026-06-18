"""Slice 1: faithful SAM3 detector config ingestion from the HF ``config.json``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_cv.models.sam3.real_config import (
    Sam3DetectorConfig,
    from_hf_config,
    from_hf_config_file,
)

REPO = Path(__file__).resolve().parents[1]

# Representative slice of the real facebook/sam3 config.json (values verified against
# the downloaded checkpoint). Used so this test needs no weights and runs in CI.
_DETECTOR_CONFIG = {
    "model_type": "sam3",
    "initializer_range": 0.02,
    "vision_config": {
        "model_type": "sam3_vision_model",
        "fpn_hidden_size": 256,
        "scale_factors": [4.0, 2.0, 1.0, 0.5],
        "backbone_feature_sizes": [[288, 288], [144, 144], [72, 72]],
        "backbone_config": {
            "model_type": "sam3_vit_model",
            "hidden_size": 1024,
            "intermediate_size": 4736,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "patch_size": 14,
            "image_size": 1008,
            "window_size": 24,
            "global_attn_indexes": [7, 15, 23, 31],
            "rope_theta": 10000.0,
        },
    },
    "text_config": {
        "model_type": "clip_text_model",
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "vocab_size": 49408,
        "max_position_embeddings": 32,
        "projection_dim": 512,
    },
    "geometry_encoder_config": {"hidden_size": 256, "num_layers": 3, "num_attention_heads": 8, "roi_size": 7},
    "detr_encoder_config": {"hidden_size": 256, "num_layers": 6, "num_attention_heads": 8},
    "detr_decoder_config": {"hidden_size": 256, "num_layers": 6, "num_queries": 200, "num_attention_heads": 8},
    "mask_decoder_config": {"hidden_size": 256, "num_upsampling_stages": 3, "num_attention_heads": 8},
}
_VIDEO_CONFIG = {"model_type": "sam3_video", "detector_config": _DETECTOR_CONFIG, "tracker_config": {"model_type": "sam3_tracker_video"}}

REAL_CONFIG = REPO / "models" / "sam3-image" / "upstream" / "config.json"


def _assert_detector(cfg: Sam3DetectorConfig) -> None:
    assert cfg.vision.backbone.hidden_size == 1024
    assert cfg.vision.backbone.num_hidden_layers == 32
    assert cfg.vision.backbone.num_attention_heads == 16
    assert cfg.vision.backbone.patch_size == 14
    assert cfg.vision.backbone.image_size == 1008
    assert cfg.vision.backbone.window_size == 24
    assert cfg.vision.backbone.global_attn_indexes == (7, 15, 23, 31)
    assert cfg.vision.backbone.rope_theta == 10000.0
    assert cfg.vision.fpn_hidden_size == 256
    assert cfg.vision.scale_factors == (4.0, 2.0, 1.0, 0.5)
    assert cfg.vision.backbone_feature_sizes == ((288, 288), (144, 144), (72, 72))
    assert cfg.vision.image_size == 1008
    assert cfg.text.vocab_size == 49408
    assert cfg.text.hidden_size == 1024
    assert cfg.text.num_hidden_layers == 24
    assert cfg.text.max_position_embeddings == 32
    assert cfg.text.projection_dim == 512
    assert cfg.geometry_encoder.num_layers == 3
    assert cfg.geometry_encoder.roi_size == 7
    assert cfg.detr_encoder.num_layers == 6
    assert cfg.detr_decoder.num_layers == 6
    assert cfg.detr_decoder.num_queries == 200
    assert cfg.mask_decoder.num_upsampling_stages == 3
    assert cfg.image_size == 1008


def test_from_hf_config_ingests_video_wrapped_detector():
    _assert_detector(from_hf_config(_VIDEO_CONFIG))


def test_from_hf_config_ingests_bare_detector():
    _assert_detector(from_hf_config(_DETECTOR_CONFIG))


def test_from_hf_config_ingests_raw_detector_config_dict():
    # A raw detector_config dict (no wrapper, has vision_config) is accepted directly.
    _assert_detector(from_hf_config(dict(_DETECTOR_CONFIG)))


def test_defaults_match_canonical_architecture():
    cfg = Sam3DetectorConfig()
    assert cfg.vision.backbone.num_hidden_layers == 32
    assert cfg.vision.backbone.global_attn_indexes == (7, 15, 23, 31)
    assert cfg.text.vocab_size == 49408
    assert cfg.detr_decoder.num_queries == 200


def test_missing_keys_fall_back_to_defaults():
    cfg = from_hf_config({"vision_config": {"backbone_config": {"hidden_size": 768}}})
    assert cfg.vision.backbone.hidden_size == 768  # overridden
    assert cfg.vision.backbone.num_hidden_layers == 32  # defaulted
    assert cfg.text.vocab_size == 49408  # defaulted whole subtree


def test_rejects_config_without_vision():
    with pytest.raises(ValueError, match="vision_config"):
        from_hf_config({"model_type": "sam3", "text_config": {}})


def test_rejects_non_mapping():
    with pytest.raises(TypeError):
        from_hf_config([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.skipif(not REAL_CONFIG.exists(), reason="real facebook/sam3 config.json not present (gated weights)")
def test_ingests_real_facebook_sam3_config_when_present():
    raw = json.loads(REAL_CONFIG.read_text())
    # The real repo ships the sam3_video wrapper.
    assert raw.get("model_type") in {"sam3", "sam3_video"}
    cfg = from_hf_config_file(REAL_CONFIG)
    _assert_detector(cfg)
