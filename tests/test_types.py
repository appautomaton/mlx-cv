import json

import numpy as np
import pytest

from mlx_cv import Detections, Points, Result


def test_detections_length_validation():
    d = Detections(boxes=[[0, 0, 10, 10]], scores=[0.9], labels=["cat"])
    assert len(d) == 1
    with pytest.raises(ValueError):
        Detections(boxes=[[0, 0, 10, 10]], scores=[0.1, 0.2])


def test_detections_scores_optional():
    # LocateAnything emits no per-box score (§16)
    d = Detections(boxes=[[0, 0, 1, 1]], labels=["thing"])
    assert d.scores is None


def test_to_coco():
    d = Detections(boxes=[[10, 20, 30, 50]], scores=[0.8], class_ids=[1], labels=["cat"])
    c = Result(image_size=(100, 100), detections=d).to_coco(image_id=7)
    a = c["annotations"][0]
    assert a["image_id"] == 7
    assert a["bbox"] == [10.0, 20.0, 20.0, 30.0]  # xyxy -> xywh
    assert a["score"] == 0.8 and a["category_name"] == "cat"


def test_save_roundtrip(tmp_path):
    r = Result(image_size=(8, 8),
               detections=Detections(boxes=[[0, 0, 4, 4]]),
               points=Points(points=[[1, 2]]))
    p = tmp_path / "r.json"
    r.save(p)
    d = json.loads(p.read_text())
    assert d["image_size"] == [8, 8]
    assert d["detections"]["boxes"][0] == [0, 0, 4, 4]
    assert d["points"]["points"][0] == [1, 2]


def test_draw_is_reserved():
    with pytest.raises(NotImplementedError):
        Result(image_size=(8, 8)).draw()
