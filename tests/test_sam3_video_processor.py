import numpy as np
import pytest
from PIL import Image

from mlx_cv.models.sam3 import SAM3VideoProcessor, SAM3VideoProcessorConfig


def _frame(value: int, shape=(4, 6, 3)):
    return np.full(shape, value, dtype=np.uint8)


def test_sam3_video_processor_accepts_numpy_frame_sequence():
    processor = SAM3VideoProcessor(SAM3VideoProcessorConfig(image_size=(8, 10)))
    inputs, ctx = processor.preprocess([_frame(10), _frame(20, shape=(5, 7, 3))])

    assert inputs["pixel_values"].shape == (2, 3, 8, 10)
    assert ctx.frame_count == 2
    assert ctx.image_sizes == ((4, 6), (5, 7))
    assert ctx.frames[0].frame_index == 0
    assert ctx.frames[0].model_size == (8, 10)


def test_sam3_video_processor_accepts_sorted_frame_directory(tmp_path):
    Image.fromarray(_frame(2)).save(tmp_path / "b.jpg")
    Image.fromarray(_frame(1)).save(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("ignore")

    processor = SAM3VideoProcessor(SAM3VideoProcessorConfig(image_size=4))
    _, ctx = processor.preprocess({"resource_path": tmp_path})

    assert ctx.frame_count == 2
    assert ctx.frames[0].source.endswith("a.png")
    assert ctx.frames[1].source.endswith("b.jpg")


def test_sam3_video_processor_rejects_video_file_without_decoder(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not-a-real-video")

    with pytest.raises(ValueError, match="video-file decoding is optional"):
        SAM3VideoProcessor().preprocess(video_path)


def test_sam3_video_processor_validates_config():
    with pytest.raises(ValueError, match="image_size"):
        SAM3VideoProcessorConfig(image_size=0)
    with pytest.raises(ValueError, match="std"):
        SAM3VideoProcessorConfig(std=(1.0, 0.0, 1.0))
