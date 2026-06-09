# PLAN: Spine contract-proof slice + parity harness

Change: `2026-06-09-spine-contract-proof` · Stage: plan · Spec: `SPEC.md` (this dir)

## Goal
Prove the minimal vision-path spine contracts on a real model: ship `BackboneFeatures`/`HeadInput-Output` + `SpatialTransform` v2 + a golden-fixture harness, then a DINOv3 MLX backbone that matches a committed official-PyTorch fixture — with `core/` staying mlx-free and the existing 48 tests green. (Full contract: `SPEC.md`.)

## Architecture approach
No new doc needed — shapes are specified in `docs/BUILDING-BLOCKS.md` (Part 1 #1–4, Part 2) and `SPEC.md`. Key invariants threaded through every slice: **`core/` imports no `mlx`** (mlx lives only in `backbones/` behind the `[mlx]` extra); coordinates invert exactly, dense maps via documented resampling; **no `core/` edit may be forced by the model**.

## Ordered slice sequence

### Slice 1: Backbone feature + Head I/O contracts (mlx-free)
**Objective:** Add `FeatureMap`/`BackboneFeatures` and `HeadInput`/`HeadOutput` to `core/`, and have `VisionBackbone` return `BackboneFeatures` — all numpy/typing only.
**Acceptance criteria:**
- `BackboneFeatures`/`FeatureMap` carry: layout, strides/grid, cls + storage (a.k.a. register) offsets, valid mask, view axis, dtype.
- For DINOv3 (single image) only the `B,N,C` + grid `(H,W)` + cls offset 0 + storage offsets `1..R` subset is populated; multi-view (`view_axis`, packed `L,C`) fields stay defined-but-unused for future models (DA3) — foundation-forward, not over-fit to DINOv3.
- `import mlx_cv.core` triggers **no** `import mlx`; a trivial identity head consumes `HeadInput` → `HeadOutput`.
**Verification:** `.venv/bin/python -m pytest tests/test_features.py -q && ! grep -rqE "^[[:space:]]*(import mlx|from mlx)" src/mlx_cv/core && .venv/bin/python -c "import sys, mlx_cv.core; assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules), 'mlx leaked into core'"`
**Touches:** `core/features.py` (new) or `core/types.py`, `core/base.py`, `tests/test_features.py`
**Status:** complete
**Evidence:** added `core/features.py` (`Layout`, `TokenLayout`, `FeatureMap`, `BackboneFeatures`, `HeadInput`, `HeadOutput`, all numpy/typing — mlx-free); `core/base.py` `VisionBackbone`→`BackboneFeatures`, `Head`→`HeadInput`/`HeadOutput`; exported from `core/__init__.py` + top-level `__init__.py`; new `tests/test_features.py`. Verification: `pytest tests/test_features.py` → 6 passed; mlx-free regex clean; `import mlx_cv.core` smoke → mlx absent from `sys.modules`. Full suite: **54 passed** (48 prior + 6 new), no regression.
**Risks / next:** none — `view_axis`/packed-`L,C` schema fields intentionally defined-but-unused (DA3-forward).

### Slice 2: SpatialTransform v2 — dense resampling
**Objective:** Extend `SpatialTransform` with deterministic dense resampling (masks=nearest, depth/heatmap=bilinear) alongside the existing exact point/box inversion.
**Acceptance criteria:**
- Existing point/box round-trip tests still pass exactly.
- A dense map (mask + depth) resamples to model space and inverts back within tolerance under the documented policy.
**Verification:** `.venv/bin/python -m pytest tests/test_geometry.py -q`
**Touches:** `core/geometry.py`, `tests/test_geometry.py`
**Depends on:** none (independent of Slice 1)
**Status:** complete
**Evidence:** added `_sample` (nearest/bilinear inverse-warp, numpy-only) + `apply_dense`/`invert_dense`/`invert_mask`/`invert_depth`/`invert_heatmap` to `SpatialTransform`; policy documented inline (masks=nearest, depth/heatmap=bilinear, out-of-domain→fill). Verification: `pytest tests/test_geometry.py` → 11 passed (5 prior point/box round-trips still exact + 6 new). Key checks: linear-ramp depth round-trips exactly in-domain (`atol 1e-6`); nearest preserves int labels (no interpolated values); letterbox padding fills correctly; identity round-trips exactly. Full suite: **60 passed**, no regression.
**Risks / next:** none.

### Decision checkpoint — Oracle fidelity + DINOv3 variant (before Slice 3)
**Decision:** Choose the parity oracle's fidelity and lock the model config *before* the fixture schema and the MLX port assume any shapes — real gated DINOv3 weights via HF (needs license acceptance / token) **vs** a fixed-seed deterministic init exported to MLX for structural parity; and lock the DINOv3 variant (e.g. ViT-S/16, `n_storage_tokens`), dtype, and convert path.
**Why before Slice 3:** this choice determines the fixed input, fixture schema, `n_storage_tokens` offsets, dtype, and convert path that Slices 3–6 build on; deciding *after* minting (the old Slice-5 placement) deadlocks — minting cannot both produce and depend on the decision.
**Checkpoint:** decision (human)
**Gates:** Slices 3, 4, 5, 6
**Resolved (human, 2026-06-09):** **Fixed-seed → MLX** structural/implementation parity — seed a DINOv3 **ViT-S/16** in PyTorch, run `forward_features`, export those same weights to the MLX port, and compare. Proves the port's RoPE/attention/norm math reproduces exactly; self-contained (no HF-gated weights). The real-checkpoint load/convert path is deferred to Phase 3. Variant locked: ViT-S/16, `patch_size=16`, `n_storage_tokens=4` (DINOv3 default), fp32.

### Slice 3: Golden-fixture schema + harness + fixed input
**Objective:** Define the fixture schema (fixed input + expected output + **ordered intermediate taps**) and load/compare wiring in `parity/`, plus the canonical fixed DINOv3 input.
**Acceptance criteria:**
- A fixture file can be saved/loaded and run through `assert_parity` / `bisect`.
- The fixed input (deterministic image/array) is defined once and reused by mint + parity.
- The schema mandates an **ordered tap list** so `bisect` can localize drift: patch-embed output → RoPE sin/cos → each block output → final norm → cls/storage/patch split.
**Verification:** `.venv/bin/python -m pytest tests/test_parity.py -q`
**Touches:** `parity/`, `tests/test_parity.py`, fixture dir
**Depends on:** Oracle decision (above)
**Status:** complete
**Evidence:** added `save_case`/`load_case` to `parity/harness.py` (npz-native, tap order preserved via `__tap_order__`, no pickle); new `parity/fixtures.py` with `DINOV3_VARIANT` (ViT-S/16), `dinov3_fixed_input` (deterministic `(1,3,64,64)` → 4×4=16 patches), `dinov3_tap_order` (patch_embed → rope_sincos → block_00..11 → norm/cls/storage/patch); exported via `parity/__init__.py`. Verification: `pytest tests/test_parity.py` → 9 passed (4 prior + 5 new): save/load round-trip preserves values + tap order, fixed input deterministic, tap schema ordered, end-to-end fixture runs through `assert_parity`/`bisect`. Full suite: **65 passed**.
**Risks / next:** real fixture not yet minted (Slice 5); the schema is exercised here with a synthetic case.

### Slice 4: DINOv3 MLX backbone (ported from official PyTorch)
**Objective:** Implement DINOv3 ViT in MLX under `backbones/vision/dinov3/` (patch-embed, 2D-RoPE, attention, LayerScale, MLP/SwiGLU, pre-norm block, ViT) exposing a `forward_features`-equivalent that satisfies `BackboneFeatures`, behind the `[mlx]` extra; **self-register via the `register_backbone(..., kind="vision")` decorator inside its own module** (no `core/` edit).
**Acceptance criteria:**
- `forward_features` on the fixed input returns `BackboneFeatures` carrying `x_norm_clstoken` + `x_storage_tokens` + `x_norm_patchtokens` with correct shapes and token order `[cls, storage…, patch…]`; appears in `BACKBONES` once the module is imported.
- `core/` still imports no `mlx`, and **`core/registry.py` is not edited** — registration is decorator-driven (`registry.py` already supports this: §10 "adding a backbone is one registry line, never a spine edit"). This is the concrete proof of the no-core-edit invariant.
**Verification:** `.venv/bin/python -m pytest tests/test_dinov3_forward.py -q`
**Execution:** subagent recommended
**Depends on:** Slice 1, Oracle decision
**Touches:** `backbones/vision/dinov3/`, `tests/test_dinov3_forward.py` (**not** `core/registry.py`)

### Slice 5: Mint the golden fixture (out-of-band, official PyTorch DINOv3)
**Objective:** In a throwaway `torch` env, call the official PyTorch DINOv3 `forward_features` (`references/dinov3`) on the fixed input per the locked variant, capture expected output + ordered taps, commit a small fixture. Mint script lives in `tools/` and is **not** a package dependency.
**Acceptance criteria:**
- A committed fixture holds the official DINOv3 `forward_features` output + ordered taps for the fixed input; the mint step is reproducible from `tools/`.
- `torch`/`transformers` appear in **no** `pyproject` dependency group.
**Verification:** fixture loads via the Slice-3 harness; shapes + tap order match `BackboneFeatures` / the Slice-3 schema.
**Depends on:** Slice 3, Oracle decision (the gated-weights-vs-fixed-seed choice is already made at the checkpoint before Slice 3)

### Slice 6: DINOv3 parity (headline)
**Objective:** Assert our MLX DINOv3 `forward_features` matches the committed fixture within tolerance; `bisect` localizes any drift via the ordered taps.
**Acceptance criteria:**
- Parity passes within the agreed tolerance; **full `pytest` green** (prior 48 + new); the Slice-1 mlx-free check still passes (regex for `import mlx`/`from mlx` + the `sys.modules` import smoke), not the looser bare `grep`.
**Verification:** `.venv/bin/python -m pytest -q && ! grep -rqE "^[[:space:]]*(import mlx|from mlx)" src/mlx_cv/core && .venv/bin/python -c "import sys, mlx_cv.core; assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)"`
**Depends on:** Slice 4, Slice 5

## Execution routing and topology
- Default: **continue** through slices after each verification passes; execution windows are context batches, not stops.
- Checkpoint: **before Slice 3** (`decision` — oracle fidelity + DINOv3 variant lock); gates Slices 3–6. Moved earlier from the old after-Slice-5 placement, which deadlocked (minting cannot both produce and depend on the decision).
- Subagent: **Slice 4** (recommended — substantial cross-module port).
- **Parallel-safe groups:** none (Slices 1 & 2 are independent — and the oracle decision can be made while they run — but kept serial for simplicity).

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The corrected plan closes all four original execution-safety blockers — an early oracle/variant decision gating Slices 3–6, decorator self-registration with zero `core/registry.py` edit, explicit `forward_features` parity targeting, and a mandated ordered-tap schema for `bisect`.
- Concern: Only carry-forward risks remain — the oracle still depends on HF gated weights or a fixed-seed export (now resolved at the pre-Slice-3 checkpoint), and the fp32 `atol≈1e-4` tolerance may need tuning once the real fixture is minted (the ordered taps give a localization path).
- Action: Proceed to `auto-execute`; at the Slice 6 full-suite gate reuse Slice 1's stronger mlx-free check (regex for `import mlx`/`from mlx` + the `sys.modules` import smoke) rather than a bare grep.
- Verified: Codex re-review (resumed session, schema-bound) marked all 4 prior blockers `resolved` with PLAN/SPEC line citations; primary agent confirmed the cited lines and corrected the two flagged stale references (SPEC:20 constraint + this review section). Prior verdict was `needs_correction` (superseded).
