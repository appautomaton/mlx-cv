# PLAN: Depth Anything V3 Monocular

## Goal

Implement the framed DA3 monocular depth change in `SPEC.md`: complete DINOv2 weight/parity, add DPT dense prediction, return `Result.depth.depth_conf`, and prove one-image depth/conf parity with committed golden fixtures.

## Architecture Approach

See `DESIGN.md`. Execution must preserve the existing spine: `core/` stays numpy-only, model code stays behind the `[mlx]` extra, and no new spine abstraction is allowed beyond `DepthMap.depth_conf`.

## Ordered Slice Sequence

### Slice 1: Convert Rules and DINOv2 Weight Load

**Objective:** Extend the shared conversion engine only as needed for DA3/DINOv2/DPT state dict layouts, then add `backbones/vision/dinov2/convert.py` with weight loading.

**Acceptance criteria:**
- `hub.convert` still passes all DINOv3 convert tests.
- DINOv2 rules strip DA3 wrapper prefixes, map `pos_embed -> pos_embed.table`, keep packed qkv packed, and do not add split-q/k/v packing unless fixture keys prove split keys exist.
- The shared engine supports path-specific 4D layout rules so PyTorch `Conv2d` uses OIHW -> OHWI `(0, 2, 3, 1)` and PyTorch `ConvTranspose2d` uses IOHW -> OHWI `(1, 2, 3, 0)`.
- `load_dinov2_weights(model, path)` loads a tiny minted state into `DINOv2ViT` without model-specific imperative key surgery.

**Touches:** `src/mlx_cv/hub/convert.py`, `src/mlx_cv/backbones/vision/dinov2/convert.py`, `src/mlx_cv/backbones/vision/dinov2/__init__.py`, `tests/test_convert.py`, `tests/test_dinov2_convert.py`.

**Produces:** DINOv2 convert/load API and shared layout-rule coverage for later DPT conversion.

**Verification:** `uv run pytest tests/test_convert.py tests/test_dinov2_convert.py`

**Status:** complete
**Evidence:** changed `src/mlx_cv/hub/convert.py`, `src/mlx_cv/backbones/vision/dinov2/convert.py`, `src/mlx_cv/backbones/vision/dinov2/__init__.py`, `tests/test_convert.py`, and `tests/test_dinov2_convert.py`; `uv run pytest tests/test_convert.py tests/test_dinov2_convert.py` passed 8 tests.
**Risks / next:** DPT-specific conversion coverage continues in Slice 4 with shape assertions for transposed-conv weights.

### Slice 2: DINOv2 Intermediate Features and Parity Fixture

**Objective:** Populate DA3-style selected ViT intermediates from DINOv2 and prove `forward_features` parity against a committed tiny fixture.

**Acceptance criteria:**
- `DINOv2ViT.forward_features(..., intermediate_layers=[...], capture_taps=True)` returns final `BackboneFeatures` plus four ordered `intermediates` with cls/register tokens removed.
- The tiny fixture config mirrors DA3 mono where it matters: zero register tokens, disabled rope/alt/cat-token branches, packed qkv, square runtime grid equal to `pretrain_grid`, and selected layers from the DA3 mono config.
- Intermediates are final-normed the same way as DA3 reference `get_intermediate_layers`; if local final-norm eps differs from reference torch default, Slice 2 either matches reference eps or records a tap-level tolerance proof before continuing.
- DINOv2 MLX output and ordered taps match `dinov2_da3_tiny_fixture.npz`; `bisect` is clean and localizes injected drift.

**Touches:** `src/mlx_cv/backbones/vision/vit.py`, `src/mlx_cv/backbones/vision/dinov2/modeling.py`, `src/mlx_cv/parity/fixtures.py`, `tools/mint_da3_fixture.py`, `tests/fixtures/`, `tests/test_dinov2_parity.py`.

**Produces:** DA3-ready DINOv2 feature contract and DINOv2 golden fixture.

**Verification:** `uv run pytest tests/test_dinov2_forward.py tests/test_dinov2_parity.py`

**Depends on:** Slice 1

**Status:** complete
**Evidence:** changed `src/mlx_cv/backbones/vision/vit.py`, `src/mlx_cv/backbones/vision/dinov2/config.py`, `src/mlx_cv/backbones/vision/dinov2/modeling.py`, `src/mlx_cv/parity/fixtures.py`, `src/mlx_cv/parity/__init__.py`, `tools/mint_da3_fixture.py`, `tests/test_dinov2_forward.py`, `tests/test_dinov2_parity.py`, and committed DINOv2 DA3 tiny fixtures; `uv run pytest tests/test_dinov2_forward.py tests/test_dinov2_parity.py` passed 11 tests.
**Risks / next:** DINOv2 final norm is now reference-matched through `final_norm_eps=1e-5`; DPT consumes these four intermediates in Slice 4.

### Slice 3: Depth Confidence Result Contract

**Objective:** Add `DepthMap.depth_conf` while keeping `core/` free of MLX imports.

**Acceptance criteria:**
- `DepthMap(depth=..., depth_conf=...)` stores both arrays as numpy arrays with matching spatial shape.
- `DepthMap(depth=...)` remains valid and sets `depth_conf=None`.
- `Result.to_dict()` includes depth and confidence when present.
- Importing `mlx_cv.core` does not import `mlx`.

**Touches:** `src/mlx_cv/core/types.py`, `tests/test_types.py`, `tests/test_version.py`.

**Produces:** Public result type support for confidence.

**Verification:** `uv run pytest tests/test_types.py tests/test_version.py`

**Status:** complete
**Evidence:** changed `src/mlx_cv/core/types.py`, `tests/test_types.py`, and `tests/test_version.py`; `uv run pytest tests/test_types.py tests/test_version.py` passed 11 tests.
**Risks / next:** none.

### Slice 4: DPT Dense Head Family

**Objective:** Add a parameterized MLX DPT head under `heads/dense/` that consumes four ViT intermediates and emits model-space depth plus optional confidence.

**Acceptance criteria:**
- The head accepts `HeadInput` with four BNC intermediates and produces `HeadOutput` keys `depth` and, for `output_dim=2`, `depth_conf`.
- The tiny DPT fixture config pins `output_dim=2`, `use_sky_head=False`, `pos_embed=False`, `down_ratio=1`, and `norm_type="idt"`; `output_dim=1` still returns depth without confidence.
- Stage projection, `ConvTranspose2d` resize, fusion, activation, and output shapes match the DA3 DPT contract for a tiny config.
- DPT conversion rules transpose `Conv2d` and `ConvTranspose2d` weights with distinct path-specific layouts, including the resize layers.
- Bilinear upsample behavior matches reference `align_corners=True`.
- Ordered DPT taps are exposed for `bisect`.
- The head registers in `HEADS` without touching `core/`.

**Touches:** `src/mlx_cv/heads/dense/`, `src/mlx_cv/heads/__init__.py`, `tests/test_dpt_head.py`, `tests/test_dpt_convert.py`, `tests/test_registry.py`.

**Produces:** Reusable dense DPT head family.

**Verification:** `uv run pytest tests/test_dpt_head.py tests/test_dpt_convert.py tests/test_registry.py`

**Depends on:** Slice 2

**Status:** complete
**Evidence:** added `src/mlx_cv/heads/dense/`, `src/mlx_cv/heads/__init__.py`, `tests/test_dpt_head.py`, `tests/test_dpt_convert.py`, and updated `tests/test_registry.py`; `uv run pytest tests/test_dpt_head.py tests/test_dpt_convert.py tests/test_registry.py` passed 13 tests.
**Risks / next:** DPT parity is structurally covered here; Slice 7 still needs end-to-end fixture parity against the DA3 reference.

### Slice 5: DA3 Monocular Model Assembly

**Objective:** Add `models/depth_anything_v3/` that composes DINOv2 and DPT into a monocular depth model.

**Acceptance criteria:**
- `DepthAnythingV3Monocular` runs a tiny fixed-seed config from tensor input to `HeadOutput`.
- The model uses DINOv2 selected layers and DPT with the pinned mono fixture config; it does not instantiate camera encoders, multi-view attention, `DualDPT`, sky head, or Gaussian components.
- Model conversion delegates to DINOv2 and DPT convert rules.
- The model registers under a DA3 monocular name.

**Touches:** `src/mlx_cv/models/depth_anything_v3/{__init__.py,config.py,modeling.py,convert.py}`, `src/mlx_cv/models/__init__.py`, `tests/test_da3_model.py`, `tests/test_da3_convert.py`.

**Produces:** DA3 monocular MLX model and convert entry points.

**Verification:** `uv run pytest tests/test_da3_model.py tests/test_da3_convert.py`

**Depends on:** Slices 1, 2, 4

**Status:** complete
**Evidence:** added `src/mlx_cv/models/depth_anything_v3/{__init__.py,config.py,modeling.py,convert.py}`, updated `src/mlx_cv/models/__init__.py`, and added `tests/test_da3_model.py` plus `tests/test_da3_convert.py`; `uv run pytest tests/test_da3_model.py tests/test_da3_convert.py` passed 4 tests.
**Risks / next:** processor postprocess must map model-space depth/confidence to original coordinates in Slice 6.

### Slice 6: DA3 Processor and Spatial Inversion

**Objective:** Add DA3 preprocess/postprocess that returns `Result.depth` in original-image coordinates with optional confidence.

**Acceptance criteria:**
- Preprocess loads RGB numpy/PIL/path input, applies ImageNet normalization, produces NCHW MLX input, and records a `SpatialTransform`.
- The default resize policy is aspect-preserving, patch-size 14 divisible, and implemented without OpenCV.
- Postprocess converts model-space depth and confidence with `SpatialTransform.invert_depth`/`invert_dense(mode="bilinear")`.
- `Result.depth.depth` and `Result.depth.depth_conf` have original image shape.
- A model-card or docs note records DA3 checkpoint license differences without adding checkpoint dependencies.

**Touches:** `src/mlx_cv/models/depth_anything_v3/processor.py`, `docs/`, `tests/test_da3_processor.py`, `tests/test_geometry.py`.

**Produces:** One-image user-facing DA3 result path.

**Verification:** `uv run pytest tests/test_da3_processor.py tests/test_geometry.py`

**Depends on:** Slices 3, 5

**Status:** complete
**Evidence:** added `src/mlx_cv/models/depth_anything_v3/processor.py`, `tests/test_da3_processor.py`, and `docs/depth-anything-v3.md`; `uv run pytest tests/test_da3_processor.py tests/test_geometry.py` passed 15 tests.
**Risks / next:** final parity still needs committed DA3 model/head fixture and dependency guards.

### Slice 7: End-to-End DA3 Parity and Package Guardrails

**Objective:** Commit the DA3 monocular tiny golden fixture and prove MLX depth/conf parity end to end.

**Acceptance criteria:**
- `tools/mint_da3_fixture.py` mints DINOv2 and DA3 tiny fixtures from the official PyTorch reference out of band.
- The mint path disables DPT sky handling (`use_sky_head=False`) and avoids `DepthAnything3Net` mono sky postprocessing, either by calling the DPT path directly or by constructing the full model with sky disabled.
- The DA3 tiny config pins `n_register_tokens=0`, `pretrain_grid == runtime_grid`, `output_dim=2`, `use_sky_head=False`, `pos_embed=False`, `down_ratio=1`, and `norm_type="idt"`.
- `tests/fixtures/da3_monocular_tiny_fixture.npz` and weights are committed at tiny size.
- MLX DA3 on the fixed one-image fixture returns `Result.depth.depth` and `Result.depth.depth_conf` within tolerance.
- `bisect` is clean on ordered DINOv2 and DPT taps and localizes injected drift.
- Full test suite passes; `pyproject.toml` contains no `torch` or `transformers`; `core/` remains MLX-free.

**Touches:** `tools/mint_da3_fixture.py`, `src/mlx_cv/parity/fixtures.py`, `tests/fixtures/`, `tests/test_da3_parity.py`, package guard tests.

**Produces:** Terminal parity proof and package guardrails.

**Verification:** `uv run pytest`

**Depends on:** Slices 1-6

**Status:** complete
**Evidence:** extended `tools/mint_da3_fixture.py`, `src/mlx_cv/parity/fixtures.py`, and `src/mlx_cv/parity/__init__.py`; added `tests/fixtures/da3_monocular_tiny_fixture.npz`, `tests/fixtures/da3_monocular_tiny_fixture_weights.npz`, and `tests/test_da3_parity.py`; `uv run pytest` passed 127 tests, dependency guard passed, and core MLX-free guard passed.
**Risks / next:** none.

## Execution Routing and Topology

Default: direct, serial, continuation after each slice verification passes.

Overrides:
- Slice 7: subagent recommended if fixture minting or parity drift diagnosis expands beyond the planned DA3/DINOv2/DPT files.

Parallel-safe groups: none.

Checkpoints: none.

## Requirement Traceability

| SPEC acceptance | Satisfying slices |
| --- | --- |
| AC1 DINOv2 parity and real weight load | Slices 1, 2 |
| AC2 DPT dense head unit-tested | Slice 4 |
| AC3 `DepthMap.depth_conf` and MLX-free core | Slices 3, 6, 7 |
| AC4 DA3 depth/conf fixture parity and bisect | Slices 5, 7 |
| AC5 Depth/conf original-coordinate inversion | Slice 6 |
| AC6 Full pytest, no torch/transformers deps | Slice 7 |

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Convert and DINOv2 | `uv run pytest tests/test_convert.py tests/test_dinov2_convert.py tests/test_dinov2_forward.py tests/test_dinov2_parity.py` |
| Core/result contract | `uv run pytest tests/test_types.py tests/test_version.py` |
| DPT and DA3 model | `uv run pytest tests/test_dpt_head.py tests/test_dpt_convert.py tests/test_da3_model.py tests/test_da3_convert.py` |
| Processor and geometry | `uv run pytest tests/test_da3_processor.py tests/test_geometry.py` |
| Final suite | `uv run pytest` |
| Dependency guard | `uv run python -c "from pathlib import Path; s=Path('pyproject.toml').read_text(); assert 'torch' not in s and 'transformers' not in s"` |
| Core MLX-free guard | `uv run python -c "import sys, mlx_cv.core; assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)"` |

## Risks

- The approved spec asks for DPT plus confidence, while stock `da3mono-large.yaml` has `DPT(output_dim=1)` and no confidence channel. The plan resolves this by supporting both output shapes and minting the tiny parity fixture with `output_dim=2`; this proves library confidence plumbing without claiming stock DA3Mono-Large checkpoint confidence.
- Reference DPT defaults `use_sky_head=True`, and `DepthAnything3Net` can alter depth through mono sky postprocessing. Fixture minting and model assembly must pin `use_sky_head=False` so parity covers only depth/conf.
- The real conversion risk is DPT layout, not qkv packing: `Conv2d` and `ConvTranspose2d` require distinct transpose rules and path-specific tests.
- DINOv2 final-norm eps and absolute-position interpolation can cause parity drift. Slice 2 must match or prove final-norm eps tolerance, and the tiny fixture must use `pretrain_grid == runtime_grid` to keep interpolation out of the parity target.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The corrected plan explicitly addresses the prior DA3/DPT/DINOv2 blockers while preserving the existing mlx-free core boundary, staged convert/parity order, and tiny-fixture proof strategy.
- Concern: The remaining risk is implementation-time parity drift in the new path-keyed conversion rules, MLX ConvTranspose2d target-shape assumptions, final-norm eps handling, and repeated bilinear `align_corners=True` upsampling.
- Action: Proceed to `auto-execute`, adding the planned DPT convert tests and shape assertions before relying on end-to-end parity.
- Verified: Read SPEC/DESIGN/PLAN, checked local source and DA3 reference code for convert, DINOv2 features, DPT sky/conf behavior, processor resize, core result boundaries, and resumed Claude Code Opus session `cd94e5e3-88b6-4ac4-97c7-6e14b963ea3f` for an outside engineering review.

## Verification

### Summary

**Overall:** PASS
**Passed:** 31 of 31 criteria
**Remaining gaps:** none

### Slice Rollup

- Slice 1 Convert Rules and DINOv2 Weight Load: PASS, 4 criteria. Evidence: `uv run pytest tests/test_convert.py tests/test_dinov2_convert.py tests/test_dinov2_forward.py tests/test_dinov2_parity.py` passed 19 tests.
- Slice 2 DINOv2 Intermediate Features and Parity Fixture: PASS, 4 criteria. Evidence: DINOv2 parity fixture present and the same command passed DINOv2 conversion/forward/parity tests.
- Slice 3 Depth Confidence Result Contract: PASS, 4 criteria. Evidence: `uv run pytest tests/test_types.py tests/test_version.py` passed 11 tests and the core MLX-free guard passed.
- Slice 4 DPT Dense Head Family: PASS, 7 criteria. Evidence: `uv run pytest tests/test_dpt_head.py tests/test_dpt_convert.py tests/test_da3_model.py tests/test_da3_convert.py` passed 11 tests, including distinct Conv2d/ConvTranspose2d layout checks and shape assertions.
- Slice 5 DA3 Monocular Model Assembly: PASS, 4 criteria. Evidence: the DPT/DA3 command above passed model assembly and delegated conversion tests.
- Slice 6 DA3 Processor and Spatial Inversion: PASS, 5 criteria. Evidence: `uv run pytest tests/test_da3_processor.py tests/test_geometry.py` passed 15 tests and `docs/depth-anything-v3.md` records checkpoint licensing.
- Slice 7 End-to-End DA3 Parity and Package Guardrails: PASS, 7 criteria. Evidence: `uv run pytest` passed 127 tests; dependency guard and core MLX-free guard passed; DA3 and DINOv2 tiny fixtures are committed-sized artifacts under `tests/fixtures/`.

### Commands Run

- `uv run pytest tests/test_convert.py tests/test_dinov2_convert.py tests/test_dinov2_forward.py tests/test_dinov2_parity.py`
- `uv run pytest tests/test_types.py tests/test_version.py`
- `uv run pytest tests/test_dpt_head.py tests/test_dpt_convert.py tests/test_da3_model.py tests/test_da3_convert.py`
- `uv run pytest tests/test_da3_processor.py tests/test_geometry.py`
- `uv run pytest`
- `uv run python -c "from pathlib import Path; s=Path('pyproject.toml').read_text(); assert 'torch' not in s and 'transformers' not in s"`
- `uv run python -c "import sys, mlx_cv.core; assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)"`

**Change status:** complete
**New objective:** use `auto-office-hours` to shape the next objective when you are ready.
