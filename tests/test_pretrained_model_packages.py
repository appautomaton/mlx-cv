import json
from pathlib import Path

import pytest


mlx = pytest.importorskip("mlx")


def test_rfdetr_package_uses_declared_nano_variant(tmp_path, monkeypatch):
    from mlx_cv.models.rfdetr import RFDETRConfig, RFDETRModel
    import mlx_cv.models.rfdetr.convert as convert

    checkpoint = tmp_path / "model.npz"
    checkpoint.touch()
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "rfdetr", "variant": "nano"})
    )
    monkeypatch.setattr(
        RFDETRModel,
        "__init__",
        lambda self, cfg: setattr(self, "cfg", cfg),
    )
    monkeypatch.setattr(
        convert,
        "load_rfdetr_weights",
        lambda model, path, strict: (model.cfg, Path(path), strict),
    )

    config, path, strict = RFDETRModel.from_pretrained(tmp_path)

    assert config == RFDETRConfig.rfdetr_nano()
    assert path == checkpoint
    assert strict is True


def test_da3_packages_select_the_declared_mode(tmp_path, monkeypatch):
    from mlx_cv.models.depth_anything_v3 import (
        DA3MonocularConfig,
        DA3MultiViewConfig,
        DepthAnythingV3Monocular,
        DepthAnythingV3MultiView,
    )
    import mlx_cv.models.depth_anything_v3.convert as convert

    checkpoint = tmp_path / "model.npz"
    checkpoint.touch()
    monkeypatch.setattr(
        DepthAnythingV3Monocular,
        "__init__",
        lambda self, cfg: setattr(self, "cfg", cfg),
    )
    monkeypatch.setattr(
        DepthAnythingV3MultiView,
        "__init__",
        lambda self, cfg: setattr(self, "cfg", cfg),
    )
    monkeypatch.setattr(
        convert,
        "load_da3_monocular_weights",
        lambda model, path, strict: (model.cfg, Path(path), strict),
    )
    monkeypatch.setattr(
        convert,
        "load_da3_multiview_weights",
        lambda model, path, strict: (model.cfg, Path(path), strict),
    )

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "depth-anything-v3", "mode": "monocular"})
    )
    mono_config, mono_path, mono_strict = DepthAnythingV3Monocular.from_pretrained(
        tmp_path
    )
    assert mono_config == DA3MonocularConfig()
    assert mono_path == checkpoint
    assert mono_strict is True

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "depth-anything-v3", "mode": "multiview"})
    )
    multi_config, multi_path, multi_strict = DepthAnythingV3MultiView.from_pretrained(
        tmp_path
    )
    assert multi_config == DA3MultiViewConfig.small()
    assert multi_path == checkpoint
    assert multi_strict is True


def test_normalized_model_configs_roundtrip():
    from mlx_cv.models.eomt_dinov3 import EoMTDINOv3Config
    from mlx_cv.models.depth_anything_v3 import DA3MonocularConfig, DA3MultiViewConfig
    from mlx_cv.models.rfdetr import RFDETRConfig

    rfdetr = RFDETRConfig.rfdetr_nano()
    assert RFDETRConfig.from_dict(rfdetr.to_dict()) == rfdetr

    monocular = DA3MonocularConfig.tiny_fixture()
    assert DA3MonocularConfig.from_dict(monocular.to_dict()) == monocular

    multiview = DA3MultiViewConfig.small()
    assert DA3MultiViewConfig.from_dict(multiview.to_dict()) == multiview

    eomt = EoMTDINOv3Config.tiny_fixture()
    assert EoMTDINOv3Config.from_dict(eomt.to_dict()) == eomt


def test_eomt_package_accepts_official_transformers_config(tmp_path, monkeypatch):
    from mlx_cv.models.eomt_dinov3 import EoMTDINOv3, EoMTDINOv3Config
    import mlx_cv.models.eomt_dinov3.convert as convert

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "eomt_dinov3",
                "hidden_size": 384,
                "num_hidden_layers": 12,
                "num_attention_heads": 6,
                "patch_size": 16,
                "num_channels": 3,
                "num_register_tokens": 4,
                "intermediate_size": 1536,
                "layer_norm_eps": 1e-5,
                "layerscale_value": 1.0,
                "rope_theta": 100.0,
                "image_size": 640,
                "num_queries": 200,
                "num_blocks": 3,
                "num_upscale_blocks": 2,
                "id2label": {"0": "first", "1": "second"},
            }
        )
    )
    monkeypatch.setattr(EoMTDINOv3, "__init__", lambda self, cfg: setattr(self, "cfg", cfg))
    monkeypatch.setattr(
        convert,
        "load_eomt_dinov3_weights",
        lambda model, path, strict: (model.cfg, Path(path), strict),
    )

    config, path, strict = EoMTDINOv3.from_pretrained(tmp_path)

    assert isinstance(config, EoMTDINOv3Config)
    assert config.num_classes == 2
    assert config.labels == ("first", "second")
    assert config.backbone.layerscale_init == 1.0
    assert path == checkpoint
    assert strict is True
