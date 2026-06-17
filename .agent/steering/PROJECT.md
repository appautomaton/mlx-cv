# Project

## One-Liner

- MLX-native, inference-only computer-vision library for Apple Silicon — typed detection / segmentation / depth / pose / grounding from a single `Result`. (`README.md`)

## Why This Repo Exists

- Give Apple Silicon one consistent, parity-tested way to run current-gen (2025+) vision models natively on MLX, instead of scattered ad-hoc ports. (`docs/ARCHITECTURE.md §1`)
- The repo now contains the task-agnostic **spine** plus early runnable MLX-native model paths: DINOv3, Depth Anything V3 monocular and DA3-SMALL multi-view depth/camera, LocateAnything local integration, RF-DETR detection, SAM 3.1 image-mode segmentation, and SAM 3.1 video/tracker/Object Multiplex local contract coverage. RF-DETR Nano and DA3-SMALL multi-view have passed real-checkpoint upstream parity gates; LocateAnything full-checkpoint parity and SAM 3.1 image-mode upstream tap/checkpoint parity remain blocked in `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, while SAM3 video parity is tracked separately in `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`. (`README.md`, `src/mlx_cv/`)

## Owned Surfaces

| Surface | Path | Responsibility |
|---------|------|----------------|
| Spine core | `src/mlx_cv/core/` | `Result` types, `SpatialTransform`, registries, base contracts |
| Spine support | `src/mlx_cv/{ops,transforms,prompts,parity}/` | pure ops, preprocessing, prompt taxonomy, parity harness |
| LocateAnything | `src/mlx_cv/models/locateanything/` | config + weight remap + tokenizer-backed local VLM integration |
| Depth Anything V3 | `src/mlx_cv/models/depth_anything_v3/` | monocular tiny fixture; DA3-SMALL multi-view depth/confidence/camera load+forward; upstream parity/demo tooling; streaming/nested/metric/3DGS deferred |
| RF-DETR | `src/mlx_cv/models/rfdetr/` | detection model, conversion, processor, predict, tiny fixture gate, real RF-DETR Nano upstream parity gate |
| SAM 3.1 | `src/mlx_cv/models/sam3/` | image-mode text/PCS prompts, mask model, conversion, processor, predict, tiny fixture gate; video frame processor, session/tracker state, Object Multiplex local contract gate |
| Design | `docs/ARCHITECTURE.md` | contracts, package layout, 2025+ model selection |

## Stack and Commands

- Python ≥3.9; `numpy` + `pillow` base, MLX behind the `[mlx]` extra; `hatchling` build. Tests via `uv run pytest`. (`pyproject.toml`; detail in `REPO-MAP.md`)

## Decision Principles Already Visible In The Repo

- One spine, many plug-ins: a model touches `models/<family>/` + one registry line, never the spine. (`core/registry.py`, `§10`)
- Coordinates are sacred: every spatial output inverts back to original-image coords. (`core/geometry.py`, `§5.2`)
- One `Result` for all tasks — optional composable fields, not subclasses. (`core/types.py`, `§5.1`)
- Compute (`Module`) separated from orchestration (`Processor`/`Predictor`). (`core/base.py`, `§5.4`)
- Trust by parity: a model is gated by truthful committed fixtures before shipping; local tiny-oracle gates and env-gated blocker skips must not be described as full upstream reference parity. (`parity/harness.py`, `§11`)
- MIT code, weight-agnostic, inference-only; weight licenses surfaced, not gated. (`LICENSE`, `§14`)

## Evidence Anchors

- `src/mlx_cv/__init__.py`, `src/mlx_cv/core/`, `docs/ARCHITECTURE.md`, `pyproject.toml`
