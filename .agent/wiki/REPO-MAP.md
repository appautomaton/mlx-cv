# Repo Map

## One-Sentence Model

- `mlx-cv` is a single Python package that provides an import-light typed vision spine plus MLX-native model families for grounding, depth, detection, segmentation, and video tracking. (`src/mlx_cv/`, `README.md`)

## Runtime Surface

| Surface | Path | Role |
|---|---|---|
| Public Python package | `src/mlx_cv/__init__.py` | MLX-free top-level types, registries, transforms, and contracts |
| MLX model packages | `src/mlx_cv/models/` | LocateAnything, Depth Anything V3, RF-DETR, and SAM3 |
| Shared backbones | `src/mlx_cv/backbones/` | Qwen2, MoonViT, DINOv2, DINOv3, SAM3 ViT, shared transformer layers |
| Shared heads and ops | `src/mlx_cv/heads/`, `src/mlx_cv/ops/` | Detection/segmentation heads, attention/sampling/geometry primitives |
| Upstream tools | `tools/` | Optional reference capture, conversion, demos, and parity gates |
| Tests | `tests/` | Core contracts, fixtures, converters, model paths, and external blocker gates |

There is no CLI server or UI. The supported surface is the Python library plus developer/reference tools.

## Stack

- Python 3.13+, `hatchling`, `src/` layout. (`pyproject.toml`)
- Base dependencies: `numpy`, `pillow`.
- Optional runtime: `[mlx]`; optional reference/test extras remain outside the base import path.
- GitHub Actions runs the Python 3.13 test workflow with an MLX CPU backend configuration. (`.github/workflows/`)

## Commands That Work

- Network-free local tests: `.venv/bin/python -m pytest -q`.
- Last observed 2026-07-16: 615 passed, 13 skipped in about 20 seconds.
- Targeted status tests: `.venv/bin/python -m pytest -q tests/test_runtime_dependency_guards.py tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_upstream_hf.py`.
- Package build backend: `hatchling` through `pyproject.toml`.
- No formatter or linter is configured; use tests and `git diff --check`.

## Model Boundaries

- `models/locateanything/`: tokenizer-backed MoonViT/Qwen2 VLM, PBD decoding, conversion, processor.
- `models/depth_anything_v3/`: monocular and multi-view depth/confidence/camera models and conversion.
- `models/rfdetr/`: DINOv2-based RF-DETR Nano model, conversion, processor, prediction.
- `models/sam3/`: legacy local fixtures plus faithful real image/video architecture, conversion, streaming, memory, and association.

## Current Evidence And Hotspots

- Canonical release status: `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`.
- Active change: `.agent/work/2026-06-18-sam3-real-architecture-port/`; architecture Slices 1–16 are merged, external image/video numeric parity remains.
- Forward model decision: EoMT-DINOv3 real-checkpoint admission after SAM3 closeout. (`.agent/steering/ROADMAP.md`)
- Historical `.agent/work/<change>/` directories are evidence snapshots and are not loaded by default unless referenced by the active change.

## Conventions

- One model family per `models/<family>/` package.
- Runtime reference-framework imports are forbidden.
- External checkpoints stay outside Git.
- Status documents distinguish local fixture coverage, external blockers, and real upstream PASS.
