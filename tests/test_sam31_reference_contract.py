from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.sam31_reference import (
    SAM31_CHECKPOINT_PATH,
    SAM31_CONFIG_PATH,
    SAM31_REFERENCE_ROOT,
    SAM31CheckpointInventory,
    SAM31ContractError,
    SAM31ReferenceCapture,
    admit_sam31_reference,
    inspect_sam31_state_dict,
    load_reference_capture,
    load_sam31_state_dict,
    verify_official_reference_surfaces,
    write_reference_capture,
)


class _Tensor:
    def __init__(self, dtype: str):
        self.dtype = dtype


def test_sam31_inventory_counts_namespaces_and_complex_rope_contract():
    state = {
        "detector.weight": _Tensor("float32"),
        "tracker.weight": _Tensor("float32"),
        "detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis": _Tensor(
            "complex64"
        ),
    }

    inventory = inspect_sam31_state_dict(state, require_exact=False)

    assert inventory == SAM31CheckpointInventory(
        tensor_count=3,
        detector_tensor_count=2,
        tracker_tensor_count=1,
        float32_tensor_count=2,
        complex64_tensor_count=1,
        complex_rope_blocks=(0,),
    )


def test_sam31_inventory_rejects_wrong_namespace_and_complex_tensor():
    with pytest.raises(SAM31ContractError, match="top-level"):
        inspect_sam31_state_dict({"model.weight": _Tensor("float32")}, require_exact=False)

    with pytest.raises(SAM31ContractError, match="not an official RoPE"):
        inspect_sam31_state_dict(
            {"tracker.freq": _Tensor("complex64")}, require_exact=False
        )


def test_sam31_admission_returns_precise_missing_checkpoint_blocker(tmp_path):
    result = admit_sam31_reference(
        checkpoint_path=tmp_path / "missing.pt",
        config_path=tmp_path / "missing.json",
        reference_root=tmp_path / "missing-source",
    )

    assert result.status == "BLOCKED:missing checkpoint"
    assert result.blocked_reason == f"missing checkpoint: {tmp_path / 'missing.pt'}"


def test_sam31_admission_uses_official_source_and_injected_state(tmp_path):
    source = tmp_path / "sam3"
    for relative_path, marker in {
        "sam3/model_builder.py": "\n".join(
            [
                "def build_sam3_image_model(): pass",
                "def build_sam3_multiplex_video_model(): pass",
                "def build_sam3_multiplex_video_predictor(): pass",
                "MultiplexController",
                "SimpleMaskEncoder",
            ]
        ),
        "sam3/model/sam3_base_predictor.py": "\n".join(
            [
                "def start_session(): pass",
                "def add_prompt(): pass",
                "def propagate_in_video(): pass",
            ]
        ),
    }.items():
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker)
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"architectures": ["Sam3VideoModel"]}))

    def state_loader(_path):
        state = {
            f"detector.weight.{i}": _Tensor("float32") for i in range(1134)
        }
        state.update(
            {f"tracker.weight.{i}": _Tensor("float32") for i in range(457)}
        )
        state.update(
            {
                f"detector.backbone.vision_backbone.trunk.blocks.{i}.attn.freqs_cis": _Tensor(
                    "complex64"
                )
                for i in range(32)
            }
        )
        return state

    result = admit_sam31_reference(
        checkpoint,
        config,
        source,
        state_loader=state_loader,
    )

    assert result.admitted
    assert result.inventory is not None
    assert result.inventory.tensor_count == 1623
    assert result.checkpoint_sha256 is not None
    assert result.config_sha256 is not None


def test_sam31_capture_round_trip_is_pickle_free(tmp_path):
    path = tmp_path / "image-reference.npz"
    capture = SAM31ReferenceCapture(
        kind="image",
        inputs={"pixels": np.arange(12, dtype=np.uint8).reshape(2, 2, 3)},
        outputs={"boxes": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)},
        taps={"vision": np.array([0.25, -0.5], dtype=np.float32)},
        metadata={"checkpoint_sha256": "abc", "config_sha256": "def"},
    )

    write_reference_capture(path, capture)
    loaded = load_reference_capture(path)

    assert loaded.kind == capture.kind
    assert loaded.metadata == capture.metadata
    np.testing.assert_array_equal(loaded.inputs["pixels"], capture.inputs["pixels"])
    np.testing.assert_array_equal(loaded.outputs["boxes"], capture.outputs["boxes"])
    np.testing.assert_array_equal(loaded.taps["vision"], capture.taps["vision"])


def test_local_official_sam31_contract_when_assets_are_available():
    if not (SAM31_REFERENCE_ROOT.exists() and SAM31_CHECKPOINT_PATH.exists()):
        pytest.skip("official SAM 3.1 source/checkpoint are external assets")

    verify_official_reference_surfaces()
    inventory = inspect_sam31_state_dict(load_sam31_state_dict(SAM31_CHECKPOINT_PATH))

    assert inventory.tensor_count == 1623
    assert inventory.detector_tensor_count == 1166
    assert inventory.tracker_tensor_count == 457
    assert inventory.float32_tensor_count == 1591
    assert inventory.complex64_tensor_count == 32
    assert SAM31_CONFIG_PATH.exists()


def test_sam31_reference_contract_has_no_transformers_dependency():
    source = Path("tools/sam31_reference.py").read_text()
    assert "import transformers" not in source
    assert "from transformers" not in source

