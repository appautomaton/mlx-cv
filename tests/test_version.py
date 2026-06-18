import mlx_cv


def test_version():
    assert mlx_cv.__version__ == "0.0.2"


def test_public_surface():
    for name in ["Result", "Detections", "Points", "SpatialTransform",
                 "Task", "Processor", "Predictor", "register_model",
                 "register_backbone", "BACKBONES", "DepthMap"]:
        assert hasattr(mlx_cv, name), name
