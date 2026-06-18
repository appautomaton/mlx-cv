import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv.core import HeadOutput, SpatialTransform
from mlx_cv.models.depth_anything_v3 import DA3Processor, DA3ProcessorConfig


def test_da3_preprocess_returns_nchw_mlx_and_spatial_context():
    processor = DA3Processor(DA3ProcessorConfig(process_res=28, patch_size=14))
    image = np.full((10, 20, 3), 128, dtype=np.uint8)
    x, ctx = processor.preprocess(image)

    assert x.shape == (1, 3, 28, 28)
    assert ctx.orig_size == (10, 20)
    assert ctx.model_size == (28, 28)
    assert ctx.scale == (1.4, 1.4)
    assert "cv2" not in sys.modules


def test_da3_preprocess_applies_imagenet_normalization():
    processor = DA3Processor(
        DA3ProcessorConfig(process_res=14, patch_size=14, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    )
    x, _ = processor.preprocess(np.zeros((14, 14, 3), dtype=np.uint8))
    assert np.allclose(np.array(x), -1.0)


def test_da3_postprocess_inverts_depth_and_confidence_to_original_shape():
    processor = DA3Processor(DA3ProcessorConfig(process_res=4, patch_size=2))
    ctx = SpatialTransform.resize((4, 4), (2, 2))
    raw = HeadOutput(data={
        "depth": mx.array(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)),
        "depth_conf": mx.array(np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)),
    })

    result = processor.postprocess(raw, ctx)

    assert result.image_size == (4, 4)
    assert result.depth.depth.shape == (4, 4)
    assert result.depth.depth_conf.shape == (4, 4)


def test_da3_postprocess_accepts_missing_confidence():
    processor = DA3Processor()
    ctx = SpatialTransform.identity((2, 2))
    result = processor.postprocess({"depth": np.ones((1, 2, 2), dtype=np.float32)}, ctx)
    assert result.depth.depth_conf is None
