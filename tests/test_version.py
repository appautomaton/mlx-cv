from importlib.metadata import version

import mlx_cv


def test_version_matches_distribution_metadata():
    # The version is written twice: in pyproject.toml, which becomes the
    # installed distribution metadata, and in __init__.py, which is what
    # callers read. Asserting a hard-coded literal here only proved that
    # someone had edited this file; comparing the two catches the release
    # that bumps one and forgets the other.
    assert mlx_cv.__version__ == version("mlx-cv")


def test_public_surface():
    for name in ["Result", "Detections", "Points", "SpatialTransform",
                 "Task", "Processor", "Predictor", "register_model",
                 "register_backbone", "BACKBONES", "DepthMap"]:
        assert hasattr(mlx_cv, name), name
