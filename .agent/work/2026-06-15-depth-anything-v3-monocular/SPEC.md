# SPEC: Depth Anything V3 — monocular depth (first full task model)

Change: `2026-06-15-depth-anything-v3-monocular` · Stage: frame · Source: `INTAKE.md`, `ROADMAP.md` Phase 3, `docs/BUILDING-BLOCKS.md` Parts 1(#6) & 3

## Bounded goal
Port **Depth Anything V3 (monocular)** onto the spine — complete the DINOv2 backbone (real config + convert rules + weight-load + `forward_features` parity) that Phase 2 built structurally, add a **DPT dense head**, and produce **`Result.depth` + `depth_conf`** for one image with forward parity vs a committed golden fixture — reusing the Phase-2 foundation with **no new spine abstraction** beyond a `DepthMap` confidence field.

## Broader intent
Phase 3 of the foundation-first roadmap: prove the vision spine + a dense head + the load/convert path end-to-end on a real model *before* the heavy VLM (Phase 4). DA3 is the lowest-difficulty full model and the first consumer of the Phase-2 foundation (DINOv2, `ViTBackbone`, `hub/convert.py`, `SpatialTransform` v2).

## Work scale / shape / lenses
- Scale: capability (Phase 3 of a roadmap). Shape: **mixed — parity** (model port vs reference) **+ feature** (DPT head, `Result.depth`/`depth_conf`, processor).
- Lenses: **engineering** (primary — backbone completion, dense head, MLX parity), **product** (light — depth `Result` ergonomics for downstream users).

## Constraints & risks (that change implementation)
- **`core/` stays mlx-free.** Only the `DepthMap`/`Result` *types* may change in `core/` (numpy + typing); the DPT head + DA3 model + DINOv2 weights live in `heads/`/`models/`/`backbones/` behind the `[mlx]` extra.
- **Reuse Phase-2 foundations** — DINOv2 backbone, `ViTBackbone`, the `hub/convert.py` rule engine, `SpatialTransform` v2 dense inversion. A new spine abstraction is a signal to fix the spine, not the model; flag if forced.
- **DINOv2 convert is the first real generality test of the convert engine** (eng-review R2 carried from Phase 2): DINOv2 may need separate-q/k/v→packed-qkv handling the DINOv3 seed lacks.
- **Reference = DA3's *vendored* DINOv2** (`references/Depth-Anything-3/src/depth_anything_3/model/dinov2/`), which may differ from the rf-detr DINOv2 (interpolation/antialias); confirm at plan.
- Parity oracle = a **golden fixture minted once, out-of-band** from the official PyTorch DA3; library depends on **neither `torch` nor the reference**. Same fixed-seed tiny-fixture method as Phase 1 (DINOv3).
- DPT reads **multiple intermediate ViT layers** — first real consumer of `BackboneFeatures.intermediates` (defined Phase 1, unused until now).

## Required outcome

**Structural change**
- `backbones/vision/dinov2/`: real config + convert rules (over `hub/convert.py`) + weight-load; `forward_features` parity (the structural Phase-2 DINOv2 gains real weights + a parity fixture).
- `heads/dense/` (new): a **DPT dense-head family** — multi-scale fusion of ViT intermediate features → dense map (+ a confidence output), parameterized.
- `core/types.py`: `DepthMap` gains `depth_conf: np.ndarray | None` (mlx-free).
- `models/depth_anything_v3/` (new): monocular DA3 `modeling.py` (DINOv2 + DPT → depth + conf), `convert.py` (rules), `processor.py` (preprocess → `(tensor, ctx)`; postprocess → `Result.depth`+`depth_conf` via `SpatialTransform` dense inversion). Register the model name.
- `parity/`: a committed DA3-monocular golden fixture + a DINOv2 fixture, with ordered taps (DINOv2 blocks → DPT fusion stages → depth/conf heads) for `bisect`.

**Behavioral invariants**
- `core/` stays mlx-free; existing **92** tests stay green.
- Depth/conf dense maps invert to original-image coords via `SpatialTransform` v2 (bilinear), documented policy.
- Adding DA3 needs **no new spine abstraction** beyond `DepthMap.depth_conf` (proof the spine carries a dense task model).
- No `torch`/`transformers` in any `pyproject` dependency group.

**Parity target**
- Reference (oracle): official PyTorch DA3 **monocular** forward (`model/da3.py:DepthAnything3Net` + `model/dpt.py:DPT`) → depth + confidence on one fixed image; plus DINOv2 `forward_features`. Captured once into committed fixtures (fixed-seed tiny config; real DA3-BASE mintable on demand, weights not committed).
- Conformance: MLX DA3 depth + `depth_conf` match the fixture within tolerance; DINOv2 `forward_features` matches its fixture; `bisect` localizes drift via ordered taps.

## Acceptance criteria
1. DINOv2 `forward_features` matches a committed golden fixture within tolerance, with real weights loaded via convert rules over `hub/convert.py`.
2. The DPT dense head is unit-tested (multi-scale ViT features → dense map of the expected shape).
3. `DepthMap.depth_conf` exists and is mlx-free (`import mlx_cv.core` stays mlx-free); a `Result` carries depth + confidence.
4. DA3 monocular forward on a fixed image returns `Result.depth` + `depth_conf` matching the golden fixture within tolerance; `bisect` localizes any drift.
5. A depth map round-trips to original-image coordinates through `SpatialTransform` v2 within the documented bilinear policy.
6. Full `pytest` green (92 prior + new); `core/` mlx-free (regex + `sys.modules` smoke); no `torch`/`transformers` in `pyproject`.

## Anti-goals
- Camera pose/intrinsics (`cam_enc`/`cam_dec`); multi-view (`reference_view_selector`, multi-image attention); Gaussian-splatting / 3D reconstruction (`gs_adapter`, `gsdpt`, `dualdpt`); metric-depth variants; DA3-LARGE/GIANT checkpoints.
- Any other task model (RF-DETR / SAM / LocateAnything) — later phases.

## Scope coverage
- Included / Deferred / Anti-goals: per `INTAKE.md` (adopted; not re-scoped).
- Needs-decision (resolved as assumptions below): parity-oracle fidelity; DPT exact structure + DINOv2-real convert keys (plan-stage from the reference).

## Assumptions
- Parity oracle = **fixed-seed tiny fixture** (Phase-1 method); real DA3-BASE (Apache-2.0) mintable on demand. Confirm at plan.
- Tolerance: fp32 `atol≈1e-4` on the CPU stream (as DINOv3); finalize when the fixture is minted.
- DA3's vendored DINOv2 is the convert/parity source of truth.
- Surface DA3's per-checkpoint weight license (BASE Apache-2.0; LARGE/GIANT CC-BY-NC-4.0) in the model card — a §14 note, not a gate.
