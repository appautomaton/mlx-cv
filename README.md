# mlx-cv

**MLX-native computer vision for Apple Silicon** — object detection, segmentation, and open-vocabulary grounding.

[![PyPI](https://img.shields.io/pypi/v/mlx-cv.svg)](https://pypi.org/project/mlx-cv/)
[![Python](https://img.shields.io/pypi/pyversions/mlx-cv.svg)](https://pypi.org/project/mlx-cv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **Pre-alpha / placeholder.** This release reserves the package name. The public API is not yet defined and will change without notice.

## What is this?

`mlx-cv` aims to be the detection & segmentation layer for the [MLX](https://github.com/ml-explore/mlx) ecosystem — sitting alongside `mlx-lm` and `mlx-vlm`, but focused on computer-vision tasks that run natively and efficiently on Apple Silicon.

### Planned model support

- **SAM 3** — promptable segmentation
- **LocateAnything** — open-vocabulary detection / localization
- …and more detection / grounding architectures over time

## Installation

```bash
pip install mlx-cv
```

> Requires Python 3.9+ and an Apple Silicon Mac (for the eventual MLX runtime).

## Status

| Stage | Status |
|-------|--------|
| Name reserved on PyPI | ✅ |
| Public API | 🚧 in design |
| First model port | 🚧 planned |

## License

[MIT](LICENSE) © AppAutomaton
