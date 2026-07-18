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
- SAM 3.1 image: official detector parity passed on MLX Metal BF16 from the combined Safetensors checkpoint.
- SAM 3.1 video: official 457-tensor multiplex tracker, component parity, and real MLX Metal propagation passed.

## Non-Goals

- Non-MLX execution backends.
- Checkpoint redistribution.
- Full training, dataset evaluation, or fine-tuning pipelines.
- Pre-2025 model expansion without a new approved objective.
- EoMT, DEIMv2, Sapiens2, or other new-family implementation inside the active SAM3 closeout.

## Active External Blockers

- None for the SAM 3.1 image/video release gates.
