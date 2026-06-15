# Repo Map

## One-Sentence Model

- `mlx-cv` is an MLX-native, inference-only computer-vision library for Apple Silicon (load weights → run → typed `Result` of boxes/masks/keypoints/points/depth); at `v0.0.2` it is the task-agnostic **spine** with **no runnable models yet**. (`README.md`, `pyproject.toml`, `docs/ARCHITECTURE.md`)

## What This Repository Owns

- The spine: unified `Result` types, invertible `SpatialTransform`, name→builder registries, pure box/coord ops, transforms, prompt taxonomy, parity harness. (`src/mlx_cv/core/`, `ops/`, `transforms/`, `prompts/`, `parity/`)
- Stage-1 (mlx-free) scaffolding of the first model, LocateAnything-3B: config, weight-key remap, PBD output parser. (`src/mlx_cv/models/locateanything/`)
- The architecture blueprint and 2025+ model selection. (`docs/ARCHITECTURE.md`)
- The verified foundation building-block inventory + spine-gap analysis, derived from 10 reference impls. (`docs/BUILDING-BLOCKS.md`, `references/`)

## Runtime Surfaces

| Surface | Path | Role | Entry Points | Notes |
|---------|------|------|--------------|-------|
| Python library | `src/mlx_cv/` | importable spine API | `mlx_cv/__init__.py` | numpy-backed; no CLI / server / UI |

## Stack and Infrastructure

- Python ≥3.9; build backend `hatchling`. (`pyproject.toml`)
- Base deps: `numpy`, `pillow`. Optional extras: `[mlx]` (`mlx>=0.18`, reserved — not yet imported by any code), `[test]` (`pytest`). (`pyproject.toml`)
- Test config: `pythonpath=["src"]`, `testpaths=["tests"]`. (`pyproject.toml`)
- CI: GitHub Actions — `test.yml` (unit+parity, Node-24, no `id-token`) and release-only `workflow.yml` (OIDC `id-token: write` isolated there). (`.github/workflows/`)

## Commands That Work Today

- install (dev): `uv run pytest` auto-builds the editable package
- test: `uv run pytest` → **48 passed** (~0.07s). Bare `python`/`python3` on this machine lack pytest — use `uv run`.
- build: `hatchling` via `pyproject.toml` (not exercised this session)
- lint: none configured (no linter/formatter config present)

## Apps, Packages, and Boundaries

- Single package `mlx_cv` (`src/` layout), 49 tracked files. Subpackages: `core/`, `transforms/`, `ops/`, `prompts/`, `parity/`, `backbones/{vision,llm}/`, `models/locateanything/`.
- `docs/ARCHITECTURE.md §7` also specifies `heads/`, `pipelines/`, `hub/`, `viz/` — these **do not exist in code yet**.

## Existing Conventions

### Observed

- Spine is mlx-free / numpy-backed; MLX is an optional extra, not imported by the spine. (`src/mlx_cv/__init__.py`, `pyproject.toml`)
- One model = one folder `models/<family>/` (`config`/`convert`/`decode`/`processor`/`modeling`) + a registry line. (`models/locateanything/`, `core/registry.py`)
- Spatial outputs routed through invertible `SpatialTransform`. (`core/geometry.py`)
- Parity is first-class: `ParityCase` + `bisect`. (`parity/harness.py`)
- Clean-room: model code carries "verified against reference" docstrings but vendors **no** reference code. (`models/locateanything/{convert,config,decode}.py`)

### Inferred

- Intended three-tier public API (load / compose / raw modules) per `docs/ARCHITECTURE.md §9`; only Tier-3 primitives exist so far.

### Needs Confirmation

- LocateAnything `[0,1000]` coord frame: relative to the resized image vs. the padded grid — resolve when fixtures exist. (`docs/ARCHITECTURE.md §16.7`)

## Verification and Release Surfaces

- Tests: `tests/` (11 files, 383 LOC) cover spine contracts + LA Stage-1 logic against **hand-written** expectations. No reference-parity fixtures exist yet. (`tests/`, `parity/harness.py`)
- Release: PyPI name reserved; release workflow isolated with OIDC. (`pyproject.toml`, `.github/workflows/workflow.yml`)

## Change-Relevant Hotspots

- `models/locateanything/processor.py` and `modeling.py` are still stubs (`__all__=[]`) — preprocessing, VLM assembly, image-token scatter, PBD generation, and end-to-end grounding remain unwritten.
- `backbones/vision/moonvit/` remains a placeholder package; it is the next Phase 4 dependency.
- `backbones/llm/qwen2/` now contains the completed Qwen2.5 LLM backbone: config, modeling, masks, cache, convert/load, tiny fixtures, and parity tests from `2026-06-15-locateanything-qwen2-backbone`.

## Sources Read

- `README.md`, `pyproject.toml`, `docs/ARCHITECTURE.md` — identity, deps, full design + model selection
- `src/mlx_cv/core/{types,geometry,registry,base}.py`, `__init__.py` — spine contracts
- `src/mlx_cv/models/locateanything/{config,convert,decode,processor,modeling}.py` — Stage-1 + stubs
- `src/mlx_cv/{ops,parity,prompts,transforms}/` — supporting spine
- `tests/*` (sampled) + `uv run pytest` — verified 48 passing
- `git ls-files` + machine search — confirmed no vendored/cloned reference; MLX not installed
