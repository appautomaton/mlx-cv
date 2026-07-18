# mlx-cv

**MLX-native computer vision for Apple Silicon** — an inference-only pipeline for current-generation detection, segmentation, depth, tracking, and text-prompted grounding.

[![PyPI](https://img.shields.io/pypi/v/mlx-cv.svg)](https://pypi.org/project/mlx-cv/)
[![Python](https://img.shields.io/pypi/pyversions/mlx-cv.svg)](https://pypi.org/project/mlx-cv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Pre-alpha.** `v0.0.2` includes the task-agnostic architecture spine and early model families. The public API may change without notice.

## What is this?

`mlx-cv` provides one parity-tested way to run modern computer-vision models natively on Apple Silicon through [MLX](https://github.com/ml-explore/mlx). Models return typed results for boxes, masks, points, depth, camera geometry, and tracks.

The library is **inference-only** and **weight-agnostic**. Code is MIT licensed; model weights are fetched separately and retain their original licenses.

## Architecture spine

- **`Result`** — one typed container for detections, masks, points, depth, cameras, tracks, and video results.
- **`SpatialTransform`** — deterministic coordinate and dense-output mapping back to the original input.
- **Registry** — model, vision-backbone, language-backbone, and head builders with plugin entry points.
- **Ops and transforms** — box, coordinate, sampling, resize, and letterbox primitives.
- **Parity harness** — committed fixtures plus required real-checkpoint gates that distinguish PASS from an external blocker.

Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Installation

```bash
pip install mlx-cv
```

The top-level spine remains import-light. Install the optional MLX runtime to execute model families on Apple Silicon:

```bash
pip install "mlx-cv[mlx]"
```

Requires Python 3.13+. Model execution requires an MLX-supported environment.

## Model status

The canonical release-parity matrix is `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`.

| Family | Current status |
|---|---|
| LocateAnything-3B | **`UPSTREAM_PASSED`** — 769/769 converted parameters, decoded boxes/points, and selected taps matched the real upstream checkpoint. |
| RF-DETR Nano | **`UPSTREAM_PASSED`** — real COCO checkpoint gate passed with recorded MD5 provenance. |
| Depth Anything V3 | **`UPSTREAM_PASSED`** for DA3-SMALL multi-view depth, confidence, cameras, and selected taps; monocular and processor paths also have committed coverage. |
| SAM 3.1 image | **`UPSTREAM_PASSED`** — official detector on MLX Metal BF16 reached mask IoU 0.999618, box error 0.1626px, and score error 0.001305. |
| SAM 3.1 video / Object Multiplex | **`UPSTREAM_PASSED`** — exact 457-tensor tracker, official MPS component captures, and real MLX Metal propagation passed; multiplex mask IoU was 0.99215. |

Normal checkpoint-less CI keeps external gates as honest skips or blocker records. It does not infer upstream parity from local fixtures.

## Forward work

1. Admit EoMT-DINOv3 through a bounded real-checkpoint gate.

See `.agent/steering/ROADMAP.md` for the current phase contract.

## License

[MIT](LICENSE) © AppAutomaton — code only. LocateAnything weights use NVIDIA's non-commercial license; RF-DETR N–L weights are Apache-2.0; DA3-SMALL/BASE weights are Apache-2.0; SAM weights use the SAM license.
