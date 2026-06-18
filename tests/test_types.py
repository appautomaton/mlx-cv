import json
import subprocess
import sys

import numpy as np
import pytest

from mlx_cv import CameraGeometry, DepthMap, Detections, Embedding, Keypoints, Masks, Points, Result, Tracks, VideoResult


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


def test_depthmap_confidence_optional_and_serialized():
    d = DepthMap(depth=[[1, 2], [3, 4]], depth_conf=[[0.1, 0.2], [0.3, 0.4]])
    assert d.depth.dtype == np.float64
    assert d.depth_conf.dtype == np.float64
    out = Result(image_size=(2, 2), depth=d).to_dict()
    assert out["depth"]["depth"] == [[1.0, 2.0], [3.0, 4.0]]
    assert out["depth"]["depth_conf"] == [[0.1, 0.2], [0.3, 0.4]]


def test_depthmap_confidence_defaults_to_none():
    d = DepthMap(depth=np.zeros((2, 2), dtype=np.float32))
    assert d.depth_conf is None
    assert Result(image_size=(2, 2), depth=d).to_dict()["depth"]["depth_conf"] is None


def test_depthmap_confidence_shape_must_match_depth():
    with pytest.raises(ValueError, match="depth_conf shape"):
        DepthMap(depth=np.zeros((2, 2)), depth_conf=np.zeros((2, 3)))


def test_camera_geometry_validates_and_serializes():
    geom = CameraGeometry(
        extrinsics=np.zeros((2, 3, 4), dtype=np.float32),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
    )
    result = Result(
        image_size=(2, 2),
        depth_views=[
            DepthMap(depth=np.ones((2, 2), dtype=np.float32)),
            DepthMap(depth=np.full((2, 2), 2.0, dtype=np.float32)),
        ],
        camera_geometry=geom,
    )
    out = result.to_dict()

    assert geom.view_count == 2
    assert out["depth_views"][0]["depth"] == [[1.0, 1.0], [1.0, 1.0]]
    assert out["depth_views"][1]["depth"] == [[2.0, 2.0], [2.0, 2.0]]
    assert out["camera_geometry"]["convention"] == "w2c"
    assert out["camera_geometry"]["view_count"] == 2
    assert np.asarray(out["camera_geometry"]["extrinsics"]).shape == (2, 3, 4)
    assert np.asarray(out["camera_geometry"]["intrinsics"]).shape == (2, 3, 3)


def test_camera_geometry_rejects_invalid_shapes_and_counts():
    with pytest.raises(ValueError, match="extrinsics must have shape"):
        CameraGeometry(extrinsics=np.zeros((2, 3, 3)))
    with pytest.raises(ValueError, match="intrinsics must have shape"):
        CameraGeometry(intrinsics=np.zeros((2, 4, 4)))
    with pytest.raises(ValueError, match="intrinsics has 3 views, expected 2"):
        CameraGeometry(
            extrinsics=np.zeros((2, 3, 4)),
            intrinsics=np.zeros((3, 3, 3)),
        )
    with pytest.raises(ValueError, match="view_count must be positive"):
        CameraGeometry(view_count=0)


def test_result_depth_views_must_be_depthmaps():
    with pytest.raises(TypeError, match=r"Result\.depth_views\[0\] must be a DepthMap"):
        Result(image_size=(2, 2), depth_views=[np.zeros((2, 2))])


def test_result_new_multiview_fields_preserve_existing_positional_order():
    embedding = Embedding([1.0, 2.0])
    r = Result((2, 2), None, None, None, None, None, embedding)

    assert r.embedding is embedding
    assert r.depth_views is None
    assert r.camera_geometry is None


def test_masks_labels_validate_instance_count():
    m = Masks(data=np.zeros((2, 3, 4), dtype=np.uint8), labels=["cat", "dog"])
    assert m.data.shape == (2, 3, 4)
    with pytest.raises(ValueError, match="Masks.labels has length 1, expected 2"):
        Masks(data=np.zeros((2, 3, 4), dtype=np.uint8), labels=["cat"])
    with pytest.raises(ValueError, match="requires data shape"):
        Masks(data=np.zeros((3, 4), dtype=np.uint8), labels=["cat"])


def test_result_to_dict_serializes_masks():
    masks = Masks(data=np.array([[[0, 1], [1, 0]]], dtype=np.uint8), labels=["object"])
    out = Result(image_size=(2, 2), masks=masks).to_dict()
    assert out["masks"] == {
        "data": [[[0, 1], [1, 0]]],
        "shape": [1, 2, 2],
        "kind": "instance",
        "labels": ["object"],
    }


def test_result_to_dict_serializes_keypoints_and_embedding():
    result = Result(
        image_size=(4, 4),
        keypoints=Keypoints(
            keypoints=np.array([[[1.0, 2.0], [3.0, 4.0]]]),
            skeleton=[(0, 1)],
            names=["nose", "eye"],
        ),
        embedding=Embedding([0.5, 1.5, 2.5]),
    )
    out = result.to_dict()
    assert out["keypoints"]["keypoints"] == [[[1.0, 2.0], [3.0, 4.0]]]
    assert out["keypoints"]["skeleton"] == [[0, 1]]
    assert out["keypoints"]["names"] == ["nose", "eye"]
    assert out["embedding"]["data"] == [0.5, 1.5, 2.5]


def test_tracks_validate_and_serialize_with_result():
    tracks = Tracks(ids=[7], frame_index=3, scores=[0.8], labels=["person"], metadata=[{"bucket": 0}])
    result = Result(
        image_size=(2, 2),
        masks=Masks(np.ones((1, 2, 2), dtype=bool)),
        detections=Detections([[0, 0, 2, 2]], track_ids=[7]),
        tracks=tracks,
    )
    assert result.to_dict()["tracks"] == {
        "ids": [7],
        "frame_index": 3,
        "scores": [0.8],
        "labels": ["person"],
        "metadata": [{"bucket": 0}],
    }


def test_result_rejects_mismatched_track_lengths():
    with pytest.raises(ValueError, match="to match detections"):
        Result(
            image_size=(2, 2),
            detections=Detections([[0, 0, 1, 1], [1, 1, 2, 2]]),
            tracks=Tracks([1]),
        )
    with pytest.raises(ValueError, match="to match masks"):
        Result(
            image_size=(2, 2),
            masks=Masks(np.ones((2, 2, 2), dtype=bool)),
            tracks=Tracks([1]),
        )


def test_video_result_serializes_ordered_frames(tmp_path):
    video = VideoResult(
        frames=[
            Result(image_size=(2, 2), tracks=Tracks([1], frame_index=2)),
            Result(image_size=(2, 2), tracks=Tracks([1], frame_index=4)),
        ],
        session_id="session-1",
        metadata={"source": "fixture"},
    )
    out = video.to_dict()
    assert out["frame_indices"] == [2, 4]
    assert out["session_id"] == "session-1"
    assert out["frames"][0]["tracks"]["ids"] == [1]

    path = tmp_path / "video.json"
    video.save(path)
    assert json.loads(path.read_text())["metadata"] == {"source": "fixture"}


def test_video_result_rejects_frame_index_mismatch():
    with pytest.raises(ValueError, match="frame_indices has length"):
        VideoResult(frames=[Result(image_size=(2, 2))], frame_indices=[0, 1])


def test_core_import_is_mlx_free_for_depth_result_contract():
    code = ("import sys, mlx_cv.core; "
            "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
