import numpy as np

from mlx_cv.prompts import BoxPrompt, ExemplarPrompt, PointPrompt, TextPrompt


def test_text_prompt():
    assert TextPrompt("a cat").text == "a cat"


def test_point_prompt():
    p = PointPrompt(points=[[1, 2], [3, 4]], labels=[1, 0])
    assert p.points.shape == (2, 2)
    assert p.labels.tolist() == [1, 0]


def test_box_prompt():
    assert BoxPrompt(boxes=[[0, 0, 1, 1]]).boxes.shape == (1, 4)


def test_exemplar_prompt():
    e = ExemplarPrompt(image=np.zeros((2, 2, 3), np.uint8), boxes=[[0, 0, 1, 1]])
    assert e.boxes.shape == (1, 4)
