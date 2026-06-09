# Requirements

## Hard Constraints

- Model execution is MLX-native on Apple Silicon; MLX is an optional `[mlx]` extra, never a base dependency. (`pyproject.toml`, `docs/ARCHITECTURE.md §1`)
- Inference-only — no training / fine-tuning in scope. (`§7`, `§13`)
- Code stays MIT and weight-agnostic; weights are user-fetched and their licenses are the user's concern, surfaced not gated. (`LICENSE`, `§14`)
- Base install stays import-light: `numpy` + `pillow` only; the spine must not import `mlx`. (`pyproject.toml`, `src/mlx_cv/__init__.py`)
- Python ≥3.9. (`pyproject.toml`)
- Model scope = the verified MVP set: **LocateAnything-3B** (anchor) · **DINOv3** · **RF-DETR** · **Depth Anything V3** · **SAM 3.1**, sharing transformer/ViT-style primitives (DINO-family where applicable; LocateAnything is MoonViT + Qwen2); recency bar **H2 2025 + 2026** only. (`docs/BUILDING-BLOCKS.md`, `.agent/steering/ROADMAP.md`)

## Invariants

- Coordinates (points/boxes) map back **exactly** via `SpatialTransform`; dense outputs (masks/depth/heatmaps) map back via **documented deterministic resampling** (not lossless). (`core/geometry.py`, `§5.2`)
- All tasks return one `Result`; new modalities are optional fields, not subclasses. (`core/types.py`, `§5.1`)
- Adding a model never edits the spine — one `models/<family>/` folder + one registry line. (`core/registry.py`, `§10`)
- Modules stay pure `nn.Module` compute, separate from `Processor`/`Predictor` orchestration. (`core/base.py`, `§5.4`)
- A model must pass reference parity before it ships. Reference corpus now cloned (`references/`); golden fixtures still to be minted. (`parity/harness.py`, `§11`, `§16.6`)
- The spine contracts must be widened to hold the model corpus **before** models are built — the 8 gaps in `docs/BUILDING-BLOCKS.md` Part 2 (VisionBackbone feature contract · SpatialTransform dense-inversion · Head signature · LanguageBackbone cache · ops · Result fields · Tracker · Prompt encoder). A model must never force a spine edit (§10).

## Non-Goals

- Training / fine-tuning. (`§13`)
- Non-MLX / non-Apple-Silicon execution backends. (`§1`)
- Pre-2025 models (OWLv2, ViTPose++, Depth Anything V2, D-FINE, …). (Appendix A)
- API-only models with no released weights (Grounding DINO 1.5/1.6, DINO-X, T-Rex-Omni). (Appendix A)
- Redistributing or relicensing model weights. (`§14`)
- RT-DETRv4 — dropped (redundant with RF-DETR / DEIMv2, least popular). (`docs/BUILDING-BLOCKS.md`)
- YOLO26 — watchlist only, not a target: very popular but AGPL (copyleft) and not flagship-accuracy. (`docs/BUILDING-BLOCKS.md`)

## Planning Blockers

- The full reference corpus (10 repos) is cloned under `references/` (git-ignored, LFS-skipped). Remaining gaps before a first model is verifiable: **(1)** Phase-1 spine-contract hardening (`docs/BUILDING-BLOCKS.md` Part 2); **(2)** MLX runtime not installed; **(3)** no golden fixtures minted. All three front-load the first-model phases. (this session; `§16.6`)

## Evidence Anchors

- `pyproject.toml`, `LICENSE`, `docs/ARCHITECTURE.md` (§1 / §7 / §10 / §11 / §14 / §16 / App. A), `src/mlx_cv/core/`
