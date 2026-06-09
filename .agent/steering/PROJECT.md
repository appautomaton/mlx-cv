# Project

## One-Liner

- MLX-native, inference-only computer-vision library for Apple Silicon — typed detection / segmentation / depth / pose / grounding from a single `Result`. (`README.md`)

## Why This Repo Exists

- Give Apple Silicon one consistent, parity-tested way to run current-gen (2025+) vision models natively on MLX, instead of scattered ad-hoc ports. (`docs/ARCHITECTURE.md §1`)
- At `v0.0.2` the deliverable is the task-agnostic **spine** that every future model plugs into — there are no runnable models yet. (`README.md`, `src/mlx_cv/__init__.py`)

## Owned Surfaces

| Surface | Path | Responsibility |
|---------|------|----------------|
| Spine core | `src/mlx_cv/core/` | `Result` types, `SpatialTransform`, registries, base contracts |
| Spine support | `src/mlx_cv/{ops,transforms,prompts,parity}/` | pure ops, preprocessing, prompt taxonomy, parity harness |
| First model (Stage 1) | `src/mlx_cv/models/locateanything/` | config + weight remap + PBD parser (mlx-free) |
| Design | `docs/ARCHITECTURE.md` | contracts, package layout, 2025+ model selection |

## Stack and Commands

- Python ≥3.9; `numpy` + `pillow` base, MLX behind the `[mlx]` extra; `hatchling` build. Tests via `uv run pytest`. (`pyproject.toml`; detail in `REPO-MAP.md`)

## Decision Principles Already Visible In The Repo

- One spine, many plug-ins: a model touches `models/<family>/` + one registry line, never the spine. (`core/registry.py`, `§10`)
- Coordinates are sacred: every spatial output inverts back to original-image coords. (`core/geometry.py`, `§5.2`)
- One `Result` for all tasks — optional composable fields, not subclasses. (`core/types.py`, `§5.1`)
- Compute (`Module`) separated from orchestration (`Processor`/`Predictor`). (`core/base.py`, `§5.4`)
- Trust by parity: a model is gated against its reference before shipping. (`parity/harness.py`, `§11`)
- MIT code, weight-agnostic, inference-only; weight licenses surfaced, not gated. (`LICENSE`, `§14`)

## Evidence Anchors

- `src/mlx_cv/__init__.py`, `src/mlx_cv/core/`, `docs/ARCHITECTURE.md`, `pyproject.toml`
