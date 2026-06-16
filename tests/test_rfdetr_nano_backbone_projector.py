import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from mlx_cv.backbones.vision.dinov2 import DINOv2Config  # noqa: E402
from mlx_cv.backbones.vision.necks.rfdetr import RFDETRP4C2fProjector  # noqa: E402
from mlx_cv.core.features import BackboneFeatures, FeatureMap, Layout  # noqa: E402
from mlx_cv.models.rfdetr import RFDETRConfig, RFDETRDINOv2Adapter, RFDETRFeatureExtractor  # noqa: E402


def _nano_cfg() -> RFDETRConfig:
    return RFDETRConfig(
        backbone=DINOv2Config.rfdetr_nano(),
        out_layers=(2, 5, 8, 11),
        projector_out_channels=256,
        projector_scale_factors=(1.0,),
        projector_kind="p4_c2f",
        projector_layer_norm=True,
    )


def _tiny_windowed_cfg() -> RFDETRConfig:
    return RFDETRConfig(
        backbone=DINOv2Config(
            embed_dim=16,
            depth=2,
            num_heads=2,
            patch_size=16,
            n_register_tokens=0,
            pretrain_grid=2,
            num_windows=2,
            windowed_full_attention_layers=(1,),
        ),
        out_layers=(1,),
        projector_out_channels=8,
        projector_scale_factors=(1.0,),
    )


def _p4_features(channels: int = 384) -> BackboneFeatures:
    maps = []
    for index in range(4):
        data = np.full((1, 4, channels), float(index + 1), dtype=np.float32)
        maps.append(FeatureMap(mx.array(data), layout=Layout.BNC, grid=(2, 2), stride=16))
    return BackboneFeatures(
        patch_tokens=maps[-1],
        intermediates=maps,
    )


def test_rfdetr_nano_backbone_config_matches_slice3_contract():
    cfg = _nano_cfg()

    assert cfg.backbone.embed_dim == 384
    assert cfg.backbone.depth == 12
    assert cfg.backbone.num_heads == 6
    assert cfg.backbone.patch_size == 16
    assert cfg.backbone.pretrain_grid == 24
    assert cfg.backbone.final_norm_eps == 1e-6
    assert cfg.backbone.num_windows == 2
    assert cfg.backbone.windowed_full_attention_layers == (3, 6, 9)
    assert cfg.out_layers == (2, 5, 8, 11)
    assert cfg.projector_scale_factors == (1.0,)
    assert cfg.projector_kind == "p4_c2f"


def test_windowed_dinov2_adapter_reports_inference_contract_and_validates_grid():
    adapter = RFDETRDINOv2Adapter(_tiny_windowed_cfg())
    mx.eval(adapter.parameters())

    features = adapter(mx.zeros((1, 3, 64, 64)))

    assert [fm.grid for fm in features.intermediates] == [(4, 4)]
    assert [fm.stride for fm in features.intermediates] == [16]
    assert features.extras["windowed_dinov2"] == {
        "num_windows": 2,
        "full_attention_layers": (1,),
        "unsupported_training_behaviors": ("drop_path", "gradient_checkpointing"),
    }

    with pytest.raises(ValueError, match="patch_size \\* num_windows"):
        adapter(mx.zeros((1, 3, 48, 64)))


def test_rfdetr_nano_feature_extractor_uses_p4_c2f_projector():
    model = RFDETRFeatureExtractor(_nano_cfg())
    params = dict(tree_flatten(model.parameters()))

    assert isinstance(model.projector, RFDETRP4C2fProjector)
    assert model.projector.in_channels == (384, 384, 384, 384)
    assert model.projector.out_channels == 256
    assert model.projector.scale_factors == (1.0,)
    assert model.projector.inference_exclusions == ("training_feature_drop",)
    assert tuple(params["projector.stages.0.0.cv1.conv.weight"].shape) == (256, 1, 1, 1536)


def test_p4_c2f_projector_consumes_four_dino_maps_to_one_p4_level():
    projector = RFDETRP4C2fProjector(in_channels=(384, 384, 384, 384), out_channels=256)
    mx.eval(projector.parameters())

    pyramid = projector(_p4_features())

    assert len(pyramid.levels) == 1
    level = pyramid.levels[0]
    assert level.data.shape == (1, 2, 2, 256)
    assert level.feature.grid == (2, 2)
    assert level.feature.stride == 16
    assert level.stride == 16
    assert level.mask.shape == (1, 2, 2)
    assert level.position.shape == (1, 2, 2, 2)


def test_p4_c2f_projector_admits_checkpoint_stage_parameter_group():
    projector = RFDETRP4C2fProjector(in_channels=(384, 384, 384, 384), out_channels=256)
    params = dict(tree_flatten(projector.parameters()))

    assert tuple(params["stages.0.0.cv1.conv.weight"].shape) == (256, 1, 1, 1536)
    assert tuple(params["stages.0.0.cv1.bn.weight"].shape) == (256,)
    assert tuple(params["stages.0.0.cv1.bn.bias"].shape) == (256,)
    assert tuple(params["stages.0.0.cv2.conv.weight"].shape) == (256, 1, 1, 640)
    assert tuple(params["stages.0.0.m.2.cv2.conv.weight"].shape) == (128, 3, 3, 128)
    assert tuple(params["stages.0.1.weight"].shape) == (256,)
    assert tuple(params["stages.0.1.bias"].shape) == (256,)
