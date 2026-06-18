import mlx.core as mx

from mlx_cv.backbones.vision.dinov2 import DINOv2Config, DINOv2ViT
from mlx_cv.heads.detection import RFDETRDecoderConfig
from mlx_cv.models.rfdetr import RFDETRConfig, RFDETRDINOv2Adapter, RFDETRFeatureExtractor, RFDETRModel


def _cfg():
    return RFDETRConfig(
        backbone=DINOv2Config(
            embed_dim=16,
            depth=2,
            num_heads=2,
            patch_size=14,
            n_register_tokens=2,
            pretrain_grid=2,
        ),
        out_layers=(0, 1),
        projector_out_channels=8,
        projector_scale_factors=(2.0, 1.0),
        decoder=RFDETRDecoderConfig(
            hidden_dim=8,
            num_queries=4,
            num_heads=2,
            num_layers=1,
            num_points=2,
            num_classes=3,
        ),
    )


def test_rfdetr_adapter_uses_dinov2_not_dinov3():
    adapter = RFDETRDINOv2Adapter(_cfg())
    assert isinstance(adapter.backbone, DINOv2ViT)


def test_rfdetr_adapter_returns_selected_dinov2_intermediates():
    adapter = RFDETRDINOv2Adapter(_cfg())
    mx.eval(adapter.parameters())
    features = adapter(mx.zeros((1, 3, 28, 28)))
    assert len(features.intermediates) == 2
    assert [fm.grid for fm in features.intermediates] == [(2, 2), (2, 2)]
    assert [fm.stride for fm in features.intermediates] == [14, 14]


def test_rfdetr_feature_extractor_returns_projected_pyramid():
    model = RFDETRFeatureExtractor(_cfg())
    mx.eval(model.parameters())
    pyramid = model(mx.zeros((1, 3, 28, 28)))
    assert [level.feature.grid for level in pyramid.levels] == [(4, 4), (2, 2)]
    assert [level.data.shape for level in pyramid.levels] == [(1, 4, 4, 8), (1, 2, 2, 8)]
    assert [level.stride for level in pyramid.levels] == [7, 14]


def test_tiny_rfdetr_model_forwards_image_to_raw_detection_outputs():
    mx.random.seed(0)
    model = RFDETRModel(_cfg())
    mx.eval(model.parameters())
    out = model(mx.zeros((1, 3, 28, 28)), capture_taps=True)
    assert out["logits"].shape == (1, 4, 3)
    assert out["boxes"].shape == (1, 4, 4)
    assert out["hidden_states"].shape == (1, 4, 8)
    assert "pyramid" in out.data
