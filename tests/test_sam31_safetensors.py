from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_cv.models.sam3.sam31_checkpoint import (
    SAM31_CHECKPOINT_METADATA,
    SAM31CheckpointError,
    load_sam31_detector_weights,
    load_sam31_weights,
    read_safetensors_metadata,
)


class _TinyLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)


class _TinyFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = _TinyLeaf()
        self.tracker = _TinyLeaf()


def _save(path: Path, weights: dict, **metadata: str) -> None:
    mx.save_safetensors(
        str(path),
        weights,
        metadata={**SAM31_CHECKPOINT_METADATA, "scope": "multiplex", **metadata},
    )


def _tiny_weights(dtype=mx.bfloat16) -> dict:
    return {
        "detector.proj.weight": mx.ones((2, 2), dtype=dtype),
        "detector.proj.bias": mx.zeros((2,), dtype=dtype),
        "tracker.proj.weight": mx.ones((2, 2), dtype=dtype),
        "tracker.proj.bias": mx.zeros((2,), dtype=dtype),
    }


def test_sam31_combined_checkpoint_loads_complete_and_detector_scopes(tmp_path):
    path = tmp_path / "sam31.safetensors"
    _save(path, _tiny_weights())

    load_sam31_weights(_TinyFull(), path)
    detector = load_sam31_detector_weights(_TinyLeaf(), path)
    mx.eval(detector.parameters())

    assert read_safetensors_metadata(path)["layout"] == "mlx-final"
    assert bool(mx.all(detector.proj.weight == 1).item())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda weights: weights.pop("tracker.proj.bias"), "parameter names"),
        (
            lambda weights: weights.__setitem__(
                "tracker.proj.bias", mx.zeros((3,), dtype=mx.bfloat16)
            ),
            "has shape",
        ),
        (
            lambda weights: weights.__setitem__(
                "tracker.proj.bias", mx.zeros((2,), dtype=mx.float32)
            ),
            "expected bfloat16",
        ),
    ],
)
def test_sam31_direct_loader_rejects_invalid_tensor_contract(
    tmp_path, mutation, message
):
    weights = _tiny_weights()
    mutation(weights)
    path = tmp_path / "bad.safetensors"
    _save(path, weights)

    with pytest.raises(SAM31CheckpointError, match=message):
        load_sam31_weights(_TinyFull(), path)


def test_sam31_direct_loader_rejects_bad_metadata_and_non_safetensors(tmp_path):
    bad = tmp_path / "bad.safetensors"
    _save(bad, _tiny_weights(), layout="pytorch")
    with pytest.raises(SAM31CheckpointError, match="metadata"):
        load_sam31_weights(_TinyFull(), bad)

    with pytest.raises(SAM31CheckpointError, match="must use .safetensors"):
        load_sam31_weights(_TinyFull(), tmp_path / "weights.npz")


def test_real_sam31_checkpoint_has_complete_final_layout_when_present():
    path = Path("models/sam3.1/mlx/sam3.1-multiplex-bf16.safetensors")
    if not path.exists():
        return

    metadata = read_safetensors_metadata(path)
    weights = mx.load(str(path))

    assert metadata["scope"] == "multiplex"
    assert metadata["source_tensor_count"] == "1623"
    assert metadata["tensor_count"] == "1963"
    assert len(weights) == 1963
    assert {value.dtype for value in weights.values()} == {mx.bfloat16}
