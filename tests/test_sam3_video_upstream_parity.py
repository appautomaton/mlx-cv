from __future__ import annotations

import hashlib
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sam3_video_upstream", REPO / "tools" / "sam3_video_upstream.py")
assert SPEC is not None and SPEC.loader is not None
sam3_video_upstream = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sam3_video_upstream
SPEC.loader.exec_module(sam3_video_upstream)

SAM3_VIDEO_CHECKPOINT_ENV = sam3_video_upstream.SAM3_VIDEO_CHECKPOINT_ENV
SAM3_VIDEO_CONFIG_ENV = sam3_video_upstream.SAM3_VIDEO_CONFIG_ENV
SAM3_VIDEO_MODEL_ID_ENV = sam3_video_upstream.SAM3_VIDEO_MODEL_ID_ENV
SAM3_VIDEO_CACHE_DIR_ENV = sam3_video_upstream.SAM3_VIDEO_CACHE_DIR_ENV
SAM3_VIDEO_REQUIRED_GATE_ENV = sam3_video_upstream.SAM3_VIDEO_REQUIRED_GATE_ENV
SAM3_VIDEO_LOCAL_CHECKPOINT_ENV = sam3_video_upstream.SAM3_VIDEO_LOCAL_CHECKPOINT_ENV
SAM3_VIDEO_OFFICIAL_MODEL_ID = sam3_video_upstream.SAM3_VIDEO_OFFICIAL_MODEL_ID
SAM3_VIDEO_CHECKPOINT_NAME = sam3_video_upstream.SAM3_VIDEO_CHECKPOINT_NAME
SAM3_VIDEO_CONFIG_NAME = sam3_video_upstream.SAM3_VIDEO_CONFIG_NAME
evaluate_sam3_video_gate = sam3_video_upstream.evaluate_sam3_video_gate
evaluate_sam3_video_reference_gate = sam3_video_upstream.evaluate_sam3_video_reference_gate
evaluate_sam3_video_comparison_gate = sam3_video_upstream.evaluate_sam3_video_comparison_gate
status_dict = sam3_video_upstream.status_dict


def _write_admitted_checkpoint_pair(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    config = tmp_path / "config.json"
    config.write_text("{}")
    return checkpoint, config


def _admitted_env(checkpoint: Path, config: Path) -> dict[str, str]:
    return {
        SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
        SAM3_VIDEO_CHECKPOINT_ENV: str(checkpoint),
        SAM3_VIDEO_CONFIG_ENV: str(config),
    }


def _write_local_checkpoint(path: Path) -> None:
    np.savez(path, **{"tracker.obj_ptr_proj.bias": np.zeros((16,), dtype=np.float32)})


def _write_full_local_checkpoint(path: Path) -> None:
    from mlx.utils import tree_flatten
    from mlx_cv.models.sam3 import SAM3VideoConfig, SAM3VideoModel

    model = SAM3VideoModel(SAM3VideoConfig.tiny_fixture())
    params = {key: np.asarray(value) for key, value in tree_flatten(model.parameters())}
    np.savez(path, **params)


def _write_comparison_checkpoints(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    local = tmp_path / "local_video.npz"
    _write_local_checkpoint(local)
    return checkpoint, config, local


def _write_reference_surfaces(reference: Path) -> Path:
    files = {
        "sam3/model_builder.py": [
            "def build_sam3_video_predictor",
            "def build_sam3_multiplex_video_predictor",
            'version == "sam3.1"',
            "Sam3TrackerPredictor",
            "SimpleMaskEncoder",
            "MultiplexController",
            "VideoTrackingDynamicMultiplex",
        ],
        "sam3/model/sam3_base_predictor.py": [
            "def start_session",
            "def add_prompt",
            "def propagate_in_video",
        ],
    }
    for relative, patterns in files.items():
        path = reference / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(patterns))
    return reference


def _capture(*, drift: float = 0.0):
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    prompt = {"boxes": [[0, 0, 3, 3]]}
    frame_indices = np.array([0, 1], dtype=np.int64)
    track_ids = np.array([[7], [7]], dtype=np.int64)
    masks = np.array(
        [
            [[[True, False, False, False], [False, True, False, False], [False, False, False, False], [False, False, False, False]]],
            [[[False, True, False, False], [False, False, True, False], [False, False, False, False], [False, False, False, False]]],
        ],
        dtype=np.bool_,
    )
    boxes = np.array([[[0.0, 0.0, 2.0, 2.0]], [[1.0, 0.0, 3.0, 2.0]]], dtype=np.float64)
    scores = np.array([[0.75], [0.8]], dtype=np.float32)
    multiplex = {"bucket_capacity": 2, "active_object_ids": [7], "object_to_bucket": {"7": 0}}
    reference = sam3_video_upstream.SAM3VideoCapture(
        source="reference",
        frames=frames,
        prompt=prompt,
        prompt_kind="box",
        frame_indices=frame_indices,
        track_ids=track_ids,
        masks=masks,
        boxes=boxes,
        scores=scores,
        multiplex=multiplex,
        taps={"score_probs": scores.copy()},
    )
    local = sam3_video_upstream.SAM3VideoCapture(
        source="local",
        frames=frames,
        prompt=prompt,
        prompt_kind="box",
        frame_indices=frame_indices.copy(),
        track_ids=track_ids.copy(),
        masks=masks.copy(),
        boxes=boxes.copy(),
        scores=scores + np.float32(drift),
        multiplex=dict(multiplex),
        taps={"score_probs": scores.copy() + np.float32(drift)},
    )
    return reference, local


def test_sam3_video_required_gate_reports_missing_checkpoint_blocker():
    result = evaluate_sam3_video_gate(environ={SAM3_VIDEO_REQUIRED_GATE_ENV: "1"})

    assert result.status == f"BLOCKED:{SAM3_VIDEO_CHECKPOINT_ENV} is unset"
    assert result.blocked is True
    assert result.blocked_reason == f"{SAM3_VIDEO_CHECKPOINT_ENV} is unset"
    assert status_dict(result)["claim_level"] == "external_blocker"
    assert status_dict(result)["official_model_id"] == SAM3_VIDEO_OFFICIAL_MODEL_ID


def test_sam3_video_gate_reports_missing_config_blocker(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-real-but-present")
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CHECKPOINT_ENV: str(checkpoint),
        },
        min_checkpoint_bytes=1,
    )

    assert result.status.startswith("BLOCKED:")
    assert result.blocked_reason == f"{SAM3_VIDEO_CONFIG_ENV} is unset for SAM3 video checkpoint admission"
    assert result.checkpoint_path == str(checkpoint)


def test_sam3_video_gate_reports_unusable_checkpoint_blocker(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    checkpoint.write_bytes(b"stub")
    config = tmp_path / "config.json"
    config.write_text("{}")
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CHECKPOINT_ENV: str(checkpoint),
            SAM3_VIDEO_CONFIG_ENV: str(config),
        }
    )

    assert result.status.startswith("BLOCKED:")
    assert "not a usable SAM3 video checkpoint" in result.blocked_reason
    assert result.checkpoint_path == str(checkpoint)
    assert result.config_path == str(config)


def test_sam3_video_gate_reports_missing_checkpoint_path_blocker(tmp_path):
    checkpoint = tmp_path / "missing.pt"
    config = tmp_path / "config.json"
    config.write_text("{}")
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CHECKPOINT_ENV: str(checkpoint),
            SAM3_VIDEO_CONFIG_ENV: str(config),
        },
        min_checkpoint_bytes=1,
    )

    assert result.status.startswith("BLOCKED:")
    assert "does not point to an existing path" in result.blocked_reason
    assert result.checkpoint_path == str(checkpoint)
    assert result.config_path == str(config)


def test_sam3_video_gate_reports_unsupported_model_id_blocker():
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_MODEL_ID_ENV: "facebook/sam3",
        }
    )

    assert result.status.startswith("BLOCKED:")
    assert result.blocked_reason == (
        f"unsupported SAM3 video model id: facebook/sam3; expected {SAM3_VIDEO_OFFICIAL_MODEL_ID}"
    )
    assert status_dict(result)["blocker_kind"] == "source"


def test_sam3_video_gate_reports_uncached_hf_blocker(tmp_path):
    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CACHE_DIR_ENV: str(tmp_path),
        }
    )

    assert result.status.startswith("BLOCKED:")
    assert "not cached" in result.blocked_reason
    assert "Hugging Face auth" in result.blocked_reason
    assert result.checkpoint_path.endswith(SAM3_VIDEO_CHECKPOINT_NAME)
    assert result.config_path.endswith(SAM3_VIDEO_CONFIG_NAME)
    assert status_dict(result)["blocker_kind"] == "download_auth"


def test_sam3_video_gate_admits_explicit_checkpoint_and_config(tmp_path):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    result = evaluate_sam3_video_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
    )

    assert result.status == "ADMITTED"
    assert result.blocked is False
    assert result.admitted is True
    assert result.checkpoint_sha256 == hashlib.sha256(b"checkpoint-bytes").hexdigest()
    assert result.config_sha256 == hashlib.sha256(b"{}").hexdigest()
    assert status_dict(result)["claim_level"] == "checkpoint_admitted"
    assert status_dict(result)["local_checkpoint_env"] == SAM3_VIDEO_LOCAL_CHECKPOINT_ENV
    assert status_dict(result)["provenance_status"] == "cached"


def test_sam3_video_gate_admits_cache_checkpoint_and_config(tmp_path):
    model_dir = tmp_path / SAM3_VIDEO_OFFICIAL_MODEL_ID.replace("/", "--")
    model_dir.mkdir()
    checkpoint = model_dir / SAM3_VIDEO_CHECKPOINT_NAME
    checkpoint.write_bytes(b"cache-checkpoint")
    config = model_dir / SAM3_VIDEO_CONFIG_NAME
    config.write_text("{}")

    result = evaluate_sam3_video_gate(
        environ={
            SAM3_VIDEO_REQUIRED_GATE_ENV: "1",
            SAM3_VIDEO_CACHE_DIR_ENV: str(tmp_path),
        },
        min_checkpoint_bytes=1,
    )

    assert result.status == "ADMITTED"
    assert result.checkpoint_path == str(checkpoint)
    assert result.config_path == str(config)
    assert result.cache_dir == str(tmp_path)


def test_sam3_video_reference_gate_reports_missing_reference_path(tmp_path, monkeypatch):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    monkeypatch.setattr(sam3_video_upstream, "SAM3_VIDEO_REFERENCE_PATH", tmp_path / "missing-reference")

    result = evaluate_sam3_video_reference_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
    )

    assert result.status.startswith("BLOCKED:")
    assert "reference path is missing" in result.blocked_reason
    assert result.checkpoint_sha256 == hashlib.sha256(b"checkpoint-bytes").hexdigest()
    assert status_dict(result)["blocker_kind"] == "reference_path"


def test_sam3_video_reference_gate_reports_missing_reference_surface(tmp_path, monkeypatch):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    reference = tmp_path / "reference"
    reference.mkdir()
    monkeypatch.setattr(sam3_video_upstream, "SAM3_VIDEO_REFERENCE_PATH", reference)

    result = evaluate_sam3_video_reference_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
    )

    assert result.status.startswith("BLOCKED:")
    assert "missing expected Object Multiplex surface" in result.blocked_reason
    assert status_dict(result)["blocker_kind"] == "reference_surface"


def test_sam3_video_comparison_gate_reports_missing_reference_surface(tmp_path, monkeypatch):
    checkpoint, config, local_checkpoint = _write_comparison_checkpoints(tmp_path)
    reference = tmp_path / "reference"
    reference.mkdir()
    monkeypatch.setattr(sam3_video_upstream, "SAM3_VIDEO_REFERENCE_PATH", reference)

    result = evaluate_sam3_video_comparison_gate(
        environ={
            **_admitted_env(checkpoint, config),
            SAM3_VIDEO_LOCAL_CHECKPOINT_ENV: str(local_checkpoint),
        },
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
    )

    assert result.status.startswith("BLOCKED:")
    assert "missing expected Object Multiplex surface" in result.blocked_reason
    assert status_dict(result)["blocker_kind"] == "reference_surface"


def test_sam3_video_reference_gate_reports_missing_torch_dependency(tmp_path, monkeypatch):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)

    def _missing_torch(name):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return __import__(name)

    monkeypatch.setattr(sam3_video_upstream.importlib, "import_module", _missing_torch)
    result = evaluate_sam3_video_reference_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
    )

    assert result.status.startswith("BLOCKED:")
    assert result.blocked_reason == "SAM3 video upstream reference execution requires torch: No module named 'torch'"
    assert status_dict(result)["blocker_kind"] == "reference_runtime"


def test_sam3_video_reference_gate_reports_reference_capture_blocker(tmp_path):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    result = evaluate_sam3_video_reference_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
    )

    assert result.status.startswith("BLOCKED:")
    assert "upstream video/Object Multiplex output capture has not completed" in result.blocked_reason
    assert status_dict(result)["blocker_kind"] == "reference_capture"


def test_sam3_video_compare_captures_passes_with_documented_tolerances():
    reference, local = _capture()

    report = sam3_video_upstream.compare_sam3_video_captures(reference, local)

    assert report["passed"] is True
    assert report["tolerances"]["masks"] == {"atol": 0.0, "rtol": 0.0}
    assert report["tolerances"]["scores"] == {"atol": 1e-4, "rtol": 1e-4}
    assert report["metadata"] == {"multiplex_equal": True}
    assert [field["name"] for field in report["fields"]] == [
        "frame_indices",
        "track_ids",
        "masks",
        "boxes",
        "scores",
        "tap.score_probs",
    ]


def test_sam3_video_upstream_capture_uses_reference_predictor_api(tmp_path, monkeypatch):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    reference_path = _write_reference_surfaces(tmp_path / "reference")

    class FakePredictor:
        def __init__(self):
            self.calls = []

        def start_session(self, *, resource_path, session_id):
            self.calls.append(("start_session", Path(resource_path).exists(), session_id))
            return {"session_id": session_id}

        def add_prompt(self, **kwargs):
            self.calls.append(
                (
                    "add_prompt",
                    kwargs["frame_idx"],
                    kwargs["obj_id"],
                    np.asarray(kwargs["bounding_boxes"]).copy(),
                    kwargs["rel_coordinates"],
                )
            )
            return {"frame_index": kwargs["frame_idx"], "outputs": {}}

        def propagate_in_video(self, **kwargs):
            self.calls.append(("propagate_in_video", kwargs["start_frame_idx"], kwargs["max_frame_num_to_track"]))
            for frame_idx in range(2):
                yield {
                    "frame_index": frame_idx,
                    "outputs": {
                        "out_obj_ids": np.array([3], dtype=np.int64),
                        "out_binary_masks": np.ones((1, 4, 4), dtype=np.bool_),
                        "out_boxes_xywh": np.array([[0.0, 0.0, 4.0, 4.0]], dtype=np.float32),
                        "out_probs": np.array([0.9], dtype=np.float32),
                    },
                }

    fake = FakePredictor()
    monkeypatch.setattr(
        sam3_video_upstream,
        "_import_reference_builder",
        lambda _reference_path: SimpleNamespace(build_sam3_multiplex_video_predictor=lambda **_kwargs: fake),
    )
    frames = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    capture = sam3_video_upstream.capture_sam3_video_upstream_reference(
        checkpoint,
        config_path=config,
        reference_path=reference_path,
        inputs={"frames": frames, "prompt": {"boxes": [[2, 1, 7, 5]]}, "object_id": 3},
    )

    assert capture.source == "upstream_reference"
    assert capture.frame_indices.tolist() == [0, 1]
    assert capture.track_ids.tolist() == [[3], [3]]
    assert capture.masks.shape == (2, 1, 4, 4)
    assert fake.calls[0][0] == "start_session"
    assert fake.calls[1][0:3] == ("add_prompt", 0, 3)
    np.testing.assert_allclose(fake.calls[1][3], np.array([[0.2, 0.125, 0.5, 0.5]], dtype=np.float32))
    assert fake.calls[1][4] is True
    assert fake.calls[2] == ("propagate_in_video", 0, 2)


def test_sam3_video_comparison_gate_passes_with_injected_captures(tmp_path, monkeypatch):
    checkpoint, config, local_checkpoint = _write_comparison_checkpoints(tmp_path)
    reference, local = _capture()
    reference_path = _write_reference_surfaces(tmp_path / "reference")
    monkeypatch.setattr(sam3_video_upstream, "SAM3_VIDEO_REFERENCE_PATH", reference_path)
    seen = {}

    def reference_capture(path, *, config_path, reference_path):
        seen["checkpoint"] = Path(path)
        seen["config"] = Path(config_path)
        seen["reference_path"] = Path(reference_path)
        return reference

    def local_capture(path, *, inputs=None):
        seen["local"] = Path(path)
        assert inputs["frames"].shape == reference.frames.shape
        return local

    result = evaluate_sam3_video_comparison_gate(
        environ={
            **_admitted_env(checkpoint, config),
            SAM3_VIDEO_LOCAL_CHECKPOINT_ENV: str(local_checkpoint),
        },
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
        reference_capture_func=reference_capture,
        local_capture_func=local_capture,
    )

    assert result.status == "UPSTREAM_PASSED"
    assert result.comparison_report["passed"] is True
    assert result.local_checkpoint_path == str(local_checkpoint)
    assert status_dict(result)["claim_level"] == "upstream_passed"
    assert seen == {
        "checkpoint": checkpoint,
        "config": config,
        "reference_path": reference_path,
        "local": local_checkpoint,
    }


def test_sam3_video_comparison_gate_uses_real_local_capture_with_reference_capture_available(tmp_path, monkeypatch):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    local_checkpoint = tmp_path / "local_full.npz"
    _write_full_local_checkpoint(local_checkpoint)
    reference_path = _write_reference_surfaces(tmp_path / "reference")
    monkeypatch.setattr(sam3_video_upstream, "SAM3_VIDEO_REFERENCE_PATH", reference_path)

    def reference_capture(_path, *, config_path, reference_path):
        return sam3_video_upstream.capture_sam3_video_local(local_checkpoint, inputs=None)

    monkeypatch.setattr(sam3_video_upstream, "capture_sam3_video_upstream_reference", reference_capture)
    result = evaluate_sam3_video_comparison_gate(
        environ={
            **_admitted_env(checkpoint, config),
            SAM3_VIDEO_LOCAL_CHECKPOINT_ENV: str(local_checkpoint),
        },
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
    )

    assert result.status == "UPSTREAM_PASSED"
    assert result.local_checkpoint_path == str(local_checkpoint)
    assert result.comparison_report["local_summary"]["source"] == "mlx_local"


def test_sam3_video_comparison_gate_blocks_on_numeric_drift(tmp_path, monkeypatch):
    checkpoint, config, local_checkpoint = _write_comparison_checkpoints(tmp_path)
    reference, local = _capture(drift=1e-2)
    reference_path = _write_reference_surfaces(tmp_path / "reference")
    monkeypatch.setattr(sam3_video_upstream, "SAM3_VIDEO_REFERENCE_PATH", reference_path)

    result = evaluate_sam3_video_comparison_gate(
        environ={
            **_admitted_env(checkpoint, config),
            SAM3_VIDEO_LOCAL_CHECKPOINT_ENV: str(local_checkpoint),
        },
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
        reference_capture_func=lambda _path, *, config_path, reference_path: reference,
        local_capture_func=lambda _path, *, inputs=None: local,
    )

    assert result.status.startswith("BLOCKED:")
    assert "parity drift" in result.blocked_reason
    assert "scores" in result.blocked_reason
    assert status_dict(result)["blocker_kind"] == "parity_drift"


def test_sam3_video_comparison_gate_blocks_when_local_checkpoint_is_missing(tmp_path, monkeypatch):
    checkpoint, config = _write_admitted_checkpoint_pair(tmp_path)
    reference_path = _write_reference_surfaces(tmp_path / "reference")
    monkeypatch.setattr(sam3_video_upstream, "SAM3_VIDEO_REFERENCE_PATH", reference_path)

    result = evaluate_sam3_video_comparison_gate(
        environ=_admitted_env(checkpoint, config),
        min_checkpoint_bytes=1,
        check_reference_dependencies=False,
        reference_capture_func=lambda *args, **kwargs: pytest.fail("reference capture should not run"),
        local_capture_func=lambda *args, **kwargs: pytest.fail("local capture should not run"),
    )

    assert result.status.startswith("BLOCKED:")
    assert SAM3_VIDEO_LOCAL_CHECKPOINT_ENV in result.blocked_reason
    assert status_dict(result)["blocker_kind"] == "local_checkpoint"
