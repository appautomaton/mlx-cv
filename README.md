# mlx-cv

**MLX-native computer vision for Apple Silicon** — an inference-only pipeline for current-generation detection, segmentation, depth, tracking, and text-prompted grounding.

[![PyPI](https://img.shields.io/pypi/v/mlx-cv.svg)](https://pypi.org/project/mlx-cv/)
[![Python](https://img.shields.io/pypi/pyversions/mlx-cv.svg)](https://pypi.org/project/mlx-cv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Pre-alpha.** `v0.0.3` adds package-native Hugging Face loading for the verified BF16 LocateAnything-3B and SAM 3.1 runtimes. The public API may change without notice.

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

Install the Hub extra to resolve App Automaton aliases or exact Hugging Face repository IDs:

```bash
pip install "mlx-cv[mlx,hub]==0.0.3"
```

Local package directories work without a network lookup. Hub downloads are revision-aware, honor `HF_HUB_OFFLINE=1`, and never execute remote model code.

## Published MLX weights

| Alias | Hugging Face repository | Precision | License |
|---|---|---|---|
| `locateanything-3b-bf16` | [`appautomaton/locateanything-3b-bf16-mlx`](https://huggingface.co/appautomaton/locateanything-3b-bf16-mlx) | BF16, unquantized | NVIDIA non-commercial |
| `sam3.1` | [`appautomaton/sam3.1-multiplex-bf16-mlx`](https://huggingface.co/appautomaton/sam3.1-multiplex-bf16-mlx) | BF16, unquantized | SAM License |

The repository links become live after the 0.0.3 publication checkpoint.

```python
from mlx_cv.models.locateanything import LocateAnythingPipeline
from mlx_cv.models.sam3 import SAM3Processor, SAM3VideoSession

locate = LocateAnythingPipeline.from_pretrained("locateanything-3b-bf16")
grounding = locate.predict(image, "find every traffic sign")

sam_image = SAM3Processor.from_pretrained("sam3.1")
segments = sam_image.predict(image, "traffic sign")

sam_video = SAM3VideoSession.from_pretrained("sam3.1")
```

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

## 0.0.3 release notes

- Added final-layout BF16 Safetensors contracts for LocateAnything-3B and the combined SAM 3.1 detector/Object Multiplex tracker.
- Added local path, exact Hugging Face ID, alias, revision, cache, and offline resolution.
- Added self-contained LocateAnything, SAM image, and SAM video `from_pretrained` entry points.
- Added reproducible model-package staging, SHA256 manifests, tracked model cards, license bundling, safe sequential upload, and remote verification tooling.
- Preserved the measured SAM 3.1 Metal parity results and the four-image LocateAnything BF16 regression baseline.

## License

[MIT](LICENSE) © AppAutomaton — code only. LocateAnything weights use NVIDIA's non-commercial license; RF-DETR N–L weights are Apache-2.0; DA3-SMALL/BASE weights are Apache-2.0; SAM weights use the SAM license.
