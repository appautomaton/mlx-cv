# mlx-cv

**MLX-native computer vision for Apple Silicon** — an inference-only pipeline for current-generation (2025+) detection, segmentation, depth, pose, tracking, and text-prompted grounding.

[![PyPI](https://img.shields.io/pypi/v/mlx-cv.svg)](https://pypi.org/project/mlx-cv/)
[![Python](https://img.shields.io/pypi/pyversions/mlx-cv.svg)](https://pypi.org/project/mlx-cv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **Pre-alpha.** `v0.0.2` is the architecture spine plus early model ports. The public API may change without notice.

## What is this?

`mlx-cv` aims to be a single, consistent, parity-tested way to run modern computer-vision models natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Load weights, run, get typed results — boxes, masks, keypoints, points, depth.

It is **inference-only** and **weight-agnostic**: the code is MIT and can load weights of any license (complying with a given model's weight license is the user's responsibility). Scope is decided on each model's own merits — is it the best current model, is it portable, does it fit the spine.

## The spine (v0.0.2)

The core that every model plugs into:

- **`Result`** — one typed container for every task (`detections` / `masks` / `keypoints` / `points` / `depth` / …), with COCO + JSON export.
- **`SpatialTransform`** — an invertible coordinate context, so every output maps losslessly back to the original image.
- **Registry** — name → builder for models, backbones (vision **and** LLM kinds), and heads; third-party plugins via entry points.
- **Ops & transforms** — pure box / coordinate ops and resize / letterbox that carry the coordinate context.
- **Parity harness** — a golden-fixture + bisect contract; models are gated in CI against their reference implementation (the trust differentiator over ad-hoc ports).

Full design and rationale: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Roadmap

- **Anchor model — LocateAnything-3B** (open-vocabulary grounding / detection / pointing). The end-to-end port plan onto the spine is written up in [ARCHITECTURE.md §16](docs/ARCHITECTURE.md).
- Then, foundation-first by backbone reuse — verified MVP set: **DINOv3** backbone → **Depth Anything V3**, **RF-DETR**, **SAM 3.1**. Full building-block inventory + build sequence: [docs/BUILDING-BLOCKS.md](docs/BUILDING-BLOCKS.md).

## Installation

```bash
pip install mlx-cv          # the spine — numpy-based, import-light
```

The MLX runtime and the GPU-backed models arrive as an optional extra in a later release:

```bash
pip install "mlx-cv[mlx]"   # (reserved) MLX runtime — needed to run models, on Apple Silicon
```

> Requires Python 3.9+. Running models will require an Apple Silicon Mac.

## Status

| Stage | Status |
|-------|--------|
| Name reserved on PyPI | ✅ |
| Architecture design | ✅ `docs/ARCHITECTURE.md` |
| Spine scaffold (`v0.0.2`) | ✅ core types · geometry · registry · ops · parity |
| First model (LocateAnything) | 🚧 Phase 4 integration active: Qwen2 + MoonViT + local VLM path |

## License

[MIT](LICENSE) © AppAutomaton — **code only**. Model weights are fetched separately from their original sources and carry their own licenses.
