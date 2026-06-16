"""Mint RF-DETR parity fixtures out-of-band.

This script may import torch and the RF-DETR reference checkout. Those imports are
never package runtime dependencies. The committed MLX tests use fixed tiny inputs
from ``mlx_cv.parity.fixtures`` and compare against the saved reference outputs.

Usage in a throwaway torch env:

    PYTHONPATH=references/rf-detr python tools/mint_rfdetr_fixture.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "references" / "rf-detr" / "src"))

from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch  # noqa: E402

from mlx_cv.parity.fixtures import (  # noqa: E402
    RFDETR_MS_DEFORM_ATTN_FIXTURE_CONFIG,
    rfdetr_ms_deform_attn_fixed_inputs,
)


def main() -> None:
    inputs = rfdetr_ms_deform_attn_fixed_inputs()
    with torch.no_grad():
        expected = ms_deform_attn_core_pytorch(
            torch.from_numpy(inputs["value"]),
            torch.from_numpy(inputs["value_spatial_shapes"]),
            torch.from_numpy(inputs["sampling_locations"]),
            torch.from_numpy(inputs["attention_weights"]),
            value_spatial_shapes_hw=[
                tuple(int(x) for x in row) for row in inputs["value_spatial_shapes"].tolist()
            ],
        ).numpy()

    out = dict(inputs)
    out["expected"] = expected.astype(np.float32)
    out["__fixture_name__"] = np.asarray(RFDETR_MS_DEFORM_ATTN_FIXTURE_CONFIG["name"])
    path = REPO / "tests" / "fixtures" / "rfdetr_ms_deform_attn_tiny_fixture.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
