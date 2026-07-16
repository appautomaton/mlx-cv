# Requirements

## Hard Constraints

- Model execution is MLX-native; MLX remains an optional `[mlx]` extra rather than a base dependency. (`pyproject.toml`)
- Python 3.13+ is the supported package contract. (`pyproject.toml`)
- The project is inference-only; training and fine-tuning are out of scope. (`README.md`, `docs/ARCHITECTURE.md`)
- Code is MIT and weight-agnostic; checkpoints remain outside Git and retain upstream licenses. (`LICENSE`, `.gitignore`)
- `src/mlx_cv/` must not import Torch, Transformers, reference repositories, or checkpoint-download clients as runtime dependencies. (`tests/test_runtime_dependency_guards.py`)

## Invariants

- All tasks use the shared `Result`/`VideoResult` type family. (`src/mlx_cv/core/types.py`)
- Points and boxes map back through `SpatialTransform`; dense outputs use documented deterministic resampling. (`src/mlx_cv/core/geometry.py`)
- MLX compute modules remain separate from preprocessing, orchestration, and session state. (`src/mlx_cv/core/base.py`)
- Converters reject unsupported key families and shape mismatches rather than silently dropping tensors. (`src/mlx_cv/models/*/convert.py`)
- Release claims derive from the canonical parity matrix. A required gate becomes `UPSTREAM_PASSED` only after a real upstream/local comparison. (`.agent/work/2026-06-16-release-parity-hardening/parity-status.json`)

## Current Model Contract

- LocateAnything-3B: real upstream parity passed for parameter conversion, decoded boxes/points, and selected taps.
- RF-DETR Nano: real-checkpoint detection parity passed; segmentation and Plus XL/2XL variants remain out of scope.
- DA3-SMALL: real multi-view depth/confidence/camera parity passed; streaming, nested metric, metric-only, and 3DGS branches remain deferred.
- SAM3 image: faithful 1468-tensor detector and comparison gate implemented; gated external numeric run pending.
- SAM3 video: faithful 1797-tensor detector/tracker/neck, streaming memory, and Object-Multiplex association implemented; gated external numeric run pending.

## Non-Goals

- Non-MLX execution backends.
- Checkpoint redistribution.
- Full training, dataset evaluation, or fine-tuning pipelines.
- Pre-2025 model expansion without a new approved objective.
- EoMT, DEIMv2, Sapiens2, or other new-family implementation inside the active SAM3 closeout.

## Active External Blockers

- SAM3 image parity requires the gated upstream checkpoint and a converted local MLX checkpoint.
- SAM3 video parity requires the gated upstream checkpoint/config and a converted local MLX checkpoint; the resulting run may expose preprocessing/reference reconciliation work.
- These are external gate blockers, not missing architecture or converter blockers. (`.agent/work/2026-06-18-sam3-real-architecture-port/PLAN.md`)
