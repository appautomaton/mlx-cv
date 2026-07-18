# Project

## One-Liner

- MLX-native, inference-only computer vision for Apple Silicon with typed detection, segmentation, depth, grounding, camera, tracking, and video results. (`README.md`, `src/mlx_cv/core/types.py`)

## Why This Repo Exists

- Provide one consistent MLX architecture and parity discipline for current-generation vision models instead of unrelated one-off ports. (`docs/ARCHITECTURE.md`)
- Keep the runtime weight-agnostic and import-light while making real-checkpoint comparison the standard for strong release claims. (`pyproject.toml`, `src/mlx_cv/parity/`)

## Owned Surfaces

| Surface | Path | Responsibility |
|---|---|---|
| Core spine | `src/mlx_cv/core/` | Typed results, transforms, registries, module/processor/tracker contracts |
| Shared MLX blocks | `src/mlx_cv/backbones/`, `src/mlx_cv/heads/`, `src/mlx_cv/ops/` | ViT/LLM blocks, necks, decoders, sampling, geometry |
| LocateAnything | `src/mlx_cv/models/locateanything/` | MoonViT + Qwen2 grounding/VLM path and conversion |
| Depth Anything V3 | `src/mlx_cv/models/depth_anything_v3/` | Monocular and DA3-SMALL multi-view depth/confidence/camera paths |
| RF-DETR | `src/mlx_cv/models/rfdetr/` | Nano detection architecture, conversion, processor, prediction |
| SAM3 | `src/mlx_cv/models/sam3/` | Faithful image detector and video tracker, streaming memory, Object Multiplex |
| Reference tools | `tools/` | Optional upstream capture, conversion, demos, and required gates |

## Current Evidence

- LocateAnything-3B, RF-DETR Nano, and DA3-SMALL multi-view are `UPSTREAM_PASSED` in the canonical release matrix. (`.agent/work/2026-06-16-release-parity-hardening/parity-status.json`)
- Official SAM 3.1 image and multiplex video are `UPSTREAM_PASSED` on MLX Metal using one final-layout BF16 Safetensors checkpoint. (`src/mlx_cv/models/sam3/`, SAM3 `PLAN.md`)
- SAM 3.0 runtime, NPZ production loading, and Transformers/HF comparison gates were removed at the 3.1 cutover.

## Stack And Commands

- Python 3.13+, `hatchling`, base dependencies `numpy` and `pillow`, optional `[mlx]` runtime, tests via pytest. (`pyproject.toml`)
- Preferred local test command: `.venv/bin/python -m pytest -q` when network-free dependency resolution is required.

## Decision Principles

- One typed `Result`, not per-task result subclasses. (`src/mlx_cv/core/types.py`)
- Spatial outputs retain deterministic mappings to the original input. (`src/mlx_cv/core/geometry.py`)
- Compute modules stay separate from processors, predictors, and sessions. (`src/mlx_cv/core/base.py`)
- Runtime sources do not import Torch, Transformers, or reference repositories. (`tests/test_runtime_dependency_guards.py`)
- A local fixture is useful evidence but never substitutes for a real-checkpoint upstream PASS. (`src/mlx_cv/parity/`, release parity matrix)
