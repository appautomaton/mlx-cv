#!/usr/bin/env python3
"""Sequential real-image regression gate for a staged LocateAnything package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


CASES = (
    (
        "kitchen_knobs",
        "models/locateanything/upstream/assets/teaser.jpg",
        (20, 40, 710, 385),
        "Locate all the instances that match the following description: knobs on the stove.",
    ),
    (
        "parking_lot",
        "models/locateanything/upstream/assets/teaser.jpg",
        (710, 40, 1580, 385),
        "Point to: yellow cars.",
    ),
    (
        "gui_crop_tool",
        "models/locateanything/upstream/assets/teaser.jpg",
        (20, 385, 490, 700),
        "Locate the region that matches the following description: crop tool.",
    ),
    (
        "flying_birds",
        "models/locateanything/upstream/assets/teaser.jpg",
        (490, 385, 1070, 700),
        "Locate all the instances that match the following description: birds.",
    ),
)


def _serialize_result(result) -> dict:
    output: dict = {"image_size": list(result.image_size)}
    if result.detections is not None:
        output["boxes"] = result.detections.boxes.tolist()
        output["box_labels"] = list(result.detections.labels or [])
    if result.points is not None:
        output["points"] = result.points.points.tolist()
        output["point_labels"] = list(result.points.labels or [])
    return output


def capture(package: Path, source_root: Path, *, max_tokens: int = 256) -> dict:
    import mlx.core as mx
    from PIL import Image

    from mlx_cv.models.locateanything import LocateAnythingPipeline

    pipeline = LocateAnythingPipeline.from_pretrained(package)
    captures = {}
    for name, relative_image, crop, question in CASES:
        image_path = source_root / relative_image
        with Image.open(image_path) as image:
            image = image.convert("RGB").crop(crop)
            model_inputs, context = pipeline.processor.preprocess(
                image, pipeline.format_prompt(question)
            )
        generated = pipeline.model.pbd_generate(
            model_inputs["input_ids"],
            model_inputs["pixel_values"],
            image_grid_hws=model_inputs["image_grid_hws"],
            image_token_id=model_inputs.get("image_token_id"),
            generation_mode="hybrid",
            max_tokens=max_tokens,
        )
        result = pipeline.processor.postprocess(generated, context)
        captures[name] = {
            "image": relative_image,
            "crop": list(crop),
            "prompt": question,
            "generated_tokens": [int(token) for token in generated],
            "result": _serialize_result(result),
        }
        mx.clear_cache()
        print(
            f"{name}: {len(generated)} tokens, "
            f"{0 if result.detections is None else len(result.detections.boxes)} boxes, "
            f"{0 if result.points is None else len(result.points.points)} points",
            flush=True,
        )
    return {"schema_version": 1, "cases": captures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "tests/fixtures/locateanything_bf16_release_baseline.json",
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    actual = capture(args.package.resolve(), args.source_root.resolve(), max_tokens=args.max_tokens)
    if args.record:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        print(f"recorded {args.baseline}")
        return
    expected = json.loads(args.baseline.read_text())
    if actual != expected:
        raise SystemExit("LocateAnything BF16 release regression differs from baseline")
    print("LocateAnything BF16 release regression passed")


if __name__ == "__main__":
    main()
