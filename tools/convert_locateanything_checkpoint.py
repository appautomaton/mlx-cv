#!/usr/bin/env python3
"""Finalize LocateAnything BF16 weights with the strict mlx-cv metadata contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx_cv.hub import read_safetensors_header, rewrite_safetensors_metadata, sha256_file
from mlx_cv.models.locateanything import LOCATEANYTHING_CHECKPOINT_METADATA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Existing final-layout BF16 Safetensors")
    parser.add_argument("output", type=Path, help="Strict output Safetensors")
    parser.add_argument("--source", type=Path, required=True, help="Source checkpoint for provenance")
    parser.add_argument("--converter-version", default="mlx-cv-0.0.3")
    args = parser.parse_args()

    header = read_safetensors_header(args.input)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if len(tensors) != 769:
        raise SystemExit(f"expected 769 tensors, found {len(tensors)}")
    wrong = [key for key, value in tensors.items() if value.get("dtype") != "BF16"]
    if wrong:
        raise SystemExit(f"expected every tensor to be BF16, found {wrong[:5]!r}")
    metadata = {
        **LOCATEANYTHING_CHECKPOINT_METADATA,
        "source_checkpoint": args.source.name,
        "source_sha256": sha256_file(args.source),
        "converter_version": args.converter_version,
    }
    rewrite_safetensors_metadata(args.input, args.output, metadata)
    print(args.output)


if __name__ == "__main__":
    main()
