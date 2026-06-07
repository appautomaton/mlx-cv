import numpy as np

from mlx_cv.ops import box_convert, box_iou, clip_boxes, nms


def test_box_convert_roundtrip():
    b = np.array([[10, 20, 30, 60]], float)
    xywh = box_convert(b, "xyxy", "xywh")
    assert np.allclose(xywh, [[10, 20, 20, 40]])
    assert np.allclose(box_convert(xywh, "xywh", "xyxy"), b)
    assert np.allclose(box_convert(b, "xyxy", "cxcywh"), [[20, 40, 20, 40]])


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    assert np.isclose(box_iou(a, a)[0, 0], 1.0)
    assert box_iou(a, np.array([[10, 10, 20, 20]], float))[0, 0] == 0.0


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    scores = np.array([0.9, 0.8, 0.7])
    assert set(nms(boxes, scores, 0.5).tolist()) == {0, 2}


def test_clip_boxes():
    assert clip_boxes([[-5, -5, 200, 200]], (100, 100))[0].tolist() == [0, 0, 100, 100]
