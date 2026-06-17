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
- A model must carry a truthful fixture gate before it ships. Some current paths pass upstream-reference fixtures; RF-DETR Nano and DA3-SMALL multi-view now pass real-checkpoint upstream-vs-MLX gates; LocateAnything local integration, SAM 3.1 image-mode, and SAM 3.1 video still use local integration or deterministic contract fixtures plus required blocker gates that record missing checkpoints, tap paths, or comparison components. Blocker gates must not be described as full upstream checkpoint parity. (`parity/harness.py`, `§11`, `§16.6`)
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

- Historical blocker note: the full reference corpus was cloned under `references/` (git-ignored, LFS-skipped) to drive planning. The current repo now has committed fixture gates for the completed phases; RF-DETR Nano and DA3-SMALL multi-view have passed real-checkpoint upstream parity gates. `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` records LocateAnything as blocked by unusable 135-byte LFS stub shards plus the missing decoded-box/point/tap comparison component, and SAM 3.1 image-mode as blocked by the missing image checkpoint plus stable image tap/comparison capture. SAM3 video checkpoint parity is phase-local in `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json` and currently records `BLOCKED:MLX_CV_SAM3_VIDEO_CHECKPOINT is unset`. (`§16.6`)

## Evidence Anchors

- `pyproject.toml`, `LICENSE`, `docs/ARCHITECTURE.md` (§1 / §7 / §10 / §11 / §14 / §16 / App. A), `src/mlx_cv/core/`
