# SPEC: Spine contract-proof slice + parity harness

Change: `2026-06-09-spine-contract-proof` · Stage: frame · Source: `INTAKE.md`, `ROADMAP.md` Phase 1, `docs/BUILDING-BLOCKS.md` Part 2

## Bounded goal
Widen the vision-path spine contracts to the *minimal* shapes DINOv3 needs (`BackboneFeatures`/`FeatureMap`, `SpatialTransform` v2 dense inversion, `HeadInput`/`HeadOutput`) plus a golden-fixture harness, and **prove them by an MLX DINOv3 forward pass matching a committed golden fixture (minted once from the official PyTorch DINOv3) within tolerance — with zero `core/` edits required to carry DINOv3.**

## Broader intent
Phase 1 of the foundation-first roadmap: harden the spine + build-once blocks before stacking the MVP-5 (LocateAnything · DINOv3 · RF-DETR · DA3 · SAM 3.1). This slice de-risks the contracts on a real model before any task model is built.

## Work scale / shape / lenses
- Scale: capability (Phase 1 of a roadmap). Shape: mixed (refactor + parity).
- Lenses: **engineering** (primary — contract shapes, MLX parity), product (light — contract ergonomics for future model authors).

## Constraints & risks (that change implementation)
- **`mlx` is not installed** → first task is `pip install -e ".[mlx]"`. Blocks the parity exit until done.
- **Spine `core/` must stay mlx-free** (numpy + typing only). The new contract *types* live in `core/` mlx-free; the DINOv3 MLX *module* lives in `backbones/vision/` behind the `[mlx]` extra; parity tests import mlx (test-time, behind the extra).
- Parity oracle = a **golden fixture minted once, out-of-band**, from the **official PyTorch DINOv3** (`references/dinov3` / HF `transformers`) in a throwaway `torch` env, then committed. The library depends on **neither `torch` nor `mlx-image`**.
- `SpatialTransform` v2 dense resampling needs a **documented deterministic interpolation policy** (e.g. bilinear for depth, nearest for masks); coordinates stay exact.
- Minting a real fixture needs actual DINOv3 weights (HF) **or** a fixed-seed init for structural-shape parity — **resolved at the early decision checkpoint before the fixture-schema slice, not at mint time** (see Assumptions).

## Required outcome

**Structural change**
- `core/types.py` (or `core/features.py`): add `FeatureMap` / `BackboneFeatures` — mlx-free typed container carrying token/map layout (`B,H,W,C` | `B,N,C` | packed `L,C` | multi-view `B,S,N,C`), strides / grid shape, cls/register/storage token offsets, valid mask, view axis, dtype.
- `core/base.py`: `VisionBackbone` returns `BackboneFeatures` (not bare `list`); add `HeadInput`/`HeadOutput` shapes (feats + grid/size, not bare feats).
- `core/geometry.py`: `SpatialTransform` v2 — deterministic dense resampling for masks/depth/heatmaps alongside exact point/box inversion.
- `parity/`: a golden-fixture **schema** (fixed input + expected output + intermediate taps) and harness wiring for one tiny DINOv3 fixture.
- `backbones/vision/dinov3/`: a DINOv3 MLX backbone **ported from the official PyTorch `references/dinov3`** satisfying the `BackboneFeatures` contract, behind the `[mlx]` extra.

**Behavioral invariants**
- `core/` stays mlx-free; mlx only in `backbones/` + behind the extra.
- Points/boxes invert exactly; dense maps invert via the documented resampling policy.
- The existing 48 tests stay green (no regression).
- Adding DINOv3 required **no model-specific `core/` edit** beyond the new shared contract types (proof the spine generalizes).

**Parity target**
- Reference (oracle): the **official PyTorch DINOv3** (`references/dinov3`) **`forward_features`** output, captured once into a committed golden fixture.
- Conformance: DINOv3 `forward_features` output (`x_norm_clstoken` + `x_norm_patchtokens`, with `x_storage_tokens`) matches the reference within tolerance on one fixed input. NB: the default `forward` returns only `head(x_norm_clstoken)` — both the MLX port and the oracle call `forward_features` (≡ `forward(is_training=True)`); token order is `[cls, storage…, patch…]`.

## Acceptance criteria
1. `uv run` / `pip install -e ".[mlx]"` succeeds; `import mlx` works in the test env.
2. `BackboneFeatures`/`FeatureMap` exist in `core/`, are **mlx-free** (no `import mlx`), and `VisionBackbone` returns them; a unit test asserts the schema fields + an `import mlx_cv.core` stays mlx-free.
3. `SpatialTransform` v2: existing point/box round-trip tests pass; a new test resamples + inverts a dense map (mask + depth) within tolerance under the documented policy.
4. `HeadInput`/`HeadOutput` defined; a trivial identity head consumes `HeadInput` in a unit test.
5. Golden-fixture **schema** defined and one tiny DINOv3 fixture is loadable by `parity/harness.py`.
6. **Headline:** DINOv3 MLX `forward_features` matches the reference within tolerance via `parity/` (`assert_parity`); `bisect` localizes any drift via ordered intermediate taps.
7. Full `pytest` green (prior 48 + new), and a grep confirms `src/mlx_cv/core/` contains no `import mlx`.

## Anti-goals
- Building any task model (DA3 / RF-DETR / SAM / LocateAnything) — Phases 3–6.
- `LanguageBackbone` cache/mask, `Tracker`/video-memory, `Prompt`-encoder contracts — later phases.
- A broad `ops` package (deformable attn, mask ops) — only what a DINOv3 forward needs (none beyond the backbone).
- The full build-once block family (Phase 2) — Phase 1 builds only the minimal DINOv3 path.

## Scope coverage
- Included / Deferred / Anti-goals: per `INTAKE.md` (adopted verbatim; not re-scoped).

## Assumptions
- Parity tolerance: start fp32 `atol≈1e-4`; bf16 looser (finalize when the fixture is minted).
- Fixture source + variant: **resolved at an early decision checkpoint (before the fixture-schema slice), not after minting** — the choice fixes the DINOv3 variant (e.g. ViT-S/16, `n_storage_tokens`), fixture schema, dtype, and convert path that downstream slices assume. Prefer real DINOv3 weights via HF; if gated/heavy, use a fixed-seed deterministic init exported to MLX for **structural** forward-parity, and document the substitution.
- Fixture data is small enough to commit, or generated deterministically at test time (decide in plan).
