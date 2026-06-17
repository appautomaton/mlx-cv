# PLAN: Depth Anything 3 Multi-View Real Checkpoint

Change: `2026-06-16-depth-anything-v3-multiview-checkpoint` - Stage: plan - Spec: `SPEC.md`

## Current Truth

This is the same approved DA3 multi-view checkpoint spec, reopened for corrective execution. Slices 1-8 implemented the checkpoint resolver, upstream capture, architecture contract, multi-view result contract, local DA3 any-view backbone/head/camera path, strict real checkpoint load, synthetic fixed-input upstream parity, and demo artifact writing. That baseline is useful but no longer sufficient for verification.

Real-image follow-up evidence exposed remaining parity drift:

| Case | Failing fields | Evidence |
| --- | --- | --- |
| SOH 2-image demo | `confidence` max abs `0.2812455893` > `0.05` | `/tmp/mlx-cv-da3-real-demo/parity_summary.json` |
| Robot video 3-frame demo | `confidence` max abs `0.1753456593` > `0.05`; `intrinsics` max abs `12.3303222656` > `12.0` | `/tmp/mlx-cv-da3-real-video-demo/parity_summary.json` |

Until the corrective slices pass, DA3-SMALL multi-view must not be treated as verified or documented as a clean upstream parity pass. The original `SPEC.md` remains valid: AC6 requires real upstream-vs-MLX parity for depth, confidence, extrinsics, intrinsics, and selected taps.

## Goal

Finish the existing DA3 spec by correcting the real-image/video parity drift, keeping the plan compact enough for execution agents to reload.

## Architecture Approach

Keep the existing runtime boundary: upstream DA3, Torch, OpenCV, torchvision, and Hugging Face Hub remain in `tools/` or env-gated tests. Runtime fixes belong in MLX/NumPy/Pillow-compatible modules under `src/mlx_cv`.

The corrective focus is narrow:

- Match upstream DINOv2 absolute positional embedding interpolation for DA3 any-view grids.
- Match upstream DualDPT auxiliary `LayerNorm` behavior, including default-initialized LayerNorms whose weights are missing from the upstream checkpoint because PyTorch loads non-strictly.
- Promote real-image/video parity evidence into the required gate before restoring verified status.

## Corrective Review Input

Antigravity review used `Gemini 3.5 Flash (High)` in read-only mode, session `c22e1d64-fb0f-46c3-9188-34437b60b837`, and returned `CHANGES_NEEDED`.

Key findings to carry into execution:

- Upstream PyTorch DINOv2 uses `interpolate_offset=0.1` and PyTorch bicubic `align_corners=False` coordinate semantics for learned absolute positional embeddings. Current MLX `LearnedAbsPosEmb` uses `nn.Upsample(mode="cubic")` with direct `th / gh`, `tw / gw` scale factors.
- Upstream DualDPT constructs `LayerNorm(32)` for all `output_conv2_aux` levels. The DA3-SMALL checkpoint only stores level-0 LayerNorm weights; PyTorch non-strict load leaves levels 1-3 at default `weight=1`, `bias=0`. Current MLX uses `Identity()` for levels 1-3, skipping normalization.
- Agy's scratch in-memory patch reportedly reduced confidence error from `0.2812` to `0.0019`, intrinsics error from `12.33` to `2.0904`, and Block-0 input error from `0.1358` to about `0.00001`.
- Follow-up plan review with `Gemini 3.5 Flash (High)`, session `26cb5bc3-9477-4a02-8c3f-7ccd1a7cd785`, returned `APPROVED_WITH_RISK`. Carry forward its execution risks: Slice 10 will change the local parameter count from `437` to `443` unless default aux LayerNorm keys are injected or explicitly handled; Slice 9 needs a custom PyTorch-coordinate bicubic helper rather than MLX native cubic upsample; Slice 11 should demote stale `da3_multiview` parity status before re-promoting it.

## Baseline Slice Summary

The completed baseline remains implementation context, not terminal verification.

| Slice | Status | Outcome | Corrective relevance |
| --- | --- | --- | --- |
| 1. Checkpoint resolver | complete | DA3-SMALL config/weights resolve from out-of-git cache with provenance | keep |
| 2. Upstream capture | complete | upstream fixed multi-view capture and taps exist | keep |
| 3. Architecture contract | complete | DA3-SMALL tensor groups and unsupported branches are audited | update for default aux LayerNorm keys |
| 4. Result/processor contract | complete | `Result.depth_views` and camera geometry output exist | keep |
| 5. Any-view backbone | complete | DA3 any-view DINOv2 path exists | correct learned-abs positional interpolation |
| 6. DualDPT/camera path | complete | local depth/confidence/ray/camera path exists | correct auxiliary LayerNorm behavior |
| 7. Real checkpoint load/forward | complete | converted DA3-SMALL strict-loads current parameter tree | update strict/default-key handling |
| 8. Synthetic parity/demo | complete but superseded | fixed synthetic upstream parity passed; demo artifacts written | extend gate to real SOH/video evidence |

Detailed historical evidence is intentionally not repeated here; use orchestration summaries under `orchestration/` and git history when needed.

## Requirement Traceability

| Requirement / AC | Corrective slices |
| --- | --- |
| R3 / R5 / AC5 architecture and strict load fidelity | 10 |
| R6 / AC6 upstream-vs-MLX depth/confidence/camera parity | 9, 10, 11 |
| R7 / AC7 regression safety | 9, 10, 11 |
| R7 / AC8 truthful docs/status | 11 |
| AC9 visible demo evidence | 11 |

## Execution Routing And Topology

- Default path: serial execution through Slices 9-11 after approval; continue after each verified slice.
- Subagent route: recommended for all corrective slices because they involve cross-checking upstream/reference numerics and checkpoint semantics.
- Checkpoints: none.
- Parallel-safe groups: none.

## Ordered Slice Sequence

### Slice 9: DA3 Positional Embedding Interpolation Parity

**Objective:** Make local DA3 any-view learned absolute positional embeddings match upstream PyTorch DINOv2 interpolation on runtime patch grids.

**Acceptance criteria:**
- `LearnedAbsPosEmb` or its DA3 caller supports upstream-compatible `interpolate_offset=0.1` without changing non-DA3 defaults unless existing DINOv2 parity proves that should also change.
- MLX interpolation matches PyTorch `interpolate(..., mode="bicubic", align_corners=False, scale_factor=((H+0.1)/Gh, (W+0.1)/Gw))` closely enough for DA3 `37x37 -> 8x8`.
- The implementation uses explicit PyTorch-compatible coordinate mapping, not raw MLX `nn.Upsample(mode="cubic")`; use the local MoonViT bicubic helper pattern as a reference if useful.
- DA3 any-view backbone tests cover the offset/interpolation behavior and do not regress existing DINOv2 positional-embedding behavior.
- A diagnostic or test proves the Block-0 input drift is removed or materially reduced before the head/camera fixes are evaluated.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache PYTHONPATH=references/Depth-Anything-3/src uv run --extra test --extra mlx --extra da3-reference pytest tests/test_layers.py tests/test_dinov2_parity.py tests/test_da3_multiview_backbone.py`

**Execution:** subagent recommended

**Touches:** `src/mlx_cv/backbones/layers/position.py`, `src/mlx_cv/backbones/vision/dinov2/`, `tests/test_layers.py`, `tests/test_da3_multiview_backbone.py`, parity diagnostics as needed.

**Produces:** Upstream-compatible DA3 absolute positional interpolation and a focused regression test.

**Evidence:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache PYTHONPATH=references/Depth-Anything-3/src uv run --extra test --extra mlx --extra da3-reference pytest tests/test_layers.py tests/test_dinov2_parity.py tests/test_da3_multiview_backbone.py` passed outside the sandbox with Metal access on 2026-06-17: 30 passed.

### Slice 10: DualDPT Auxiliary LayerNorm And Default-Key Load Semantics

**Objective:** Match upstream DualDPT auxiliary output normalization while preserving strict real-checkpoint admission.

**Acceptance criteria:**
- `output_conv2_aux` uses `LayerNorm(32)` for all auxiliary levels, matching the upstream module behavior.
- Missing upstream checkpoint LayerNorm keys for aux levels 1-3 are handled deliberately: either conversion injects default `weight=1`, `bias=0`, or strict load allows only those documented default-initialized keys while continuing to fail on any other missing inference tensor.
- Parameter-tree and strict-load assertions are updated for the expected local model parameter count change from `437` to `443`, while the architecture contract still distinguishes actual upstream checkpoint keys from local default-initialized keys.
- Architecture contract, converter tests, parameter-tree assertions, and strict-load tests distinguish required checkpoint tensors from expected default-initialized tensors.
- Local real DA3-SMALL load and forward remain runtime-clean and do not import upstream/Torch from `src/mlx_cv`.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL uv run --extra test --extra mlx --extra da3-reference pytest tests/test_da3_real_architecture_contract.py tests/test_da3_real_checkpoint_load.py tests/test_da3_real_forward.py tests/test_da3_multiview_model.py tests/test_da3_convert.py tests/test_runtime_dependency_guards.py`

**Execution:** subagent recommended

**Depends on:** Slice 9

**Touches:** `src/mlx_cv/heads/dense/dualdpt.py`, `src/mlx_cv/models/depth_anything_v3/convert.py`, `tools/da3_real_architecture_contract.py`, `tests/test_da3_real_architecture_contract.py`, `tests/test_da3_real_checkpoint_load.py`, `tests/test_da3_multiview_model.py`, `tests/test_da3_convert.py`.

**Produces:** Strict checkpoint loading that matches upstream non-strict defaults without silently dropping real inference tensors.

### Slice 11: Real-Image Parity Gate And Truthful Status

**Objective:** Replace the synthetic-only completion claim with real-image/video DA3 parity evidence and corrected docs/status.

**Acceptance criteria:**
- Required DA3 parity covers the original fixed synthetic 3-view fixture and the real sample paths that exposed the bug: SOH images and robot video-derived 3-frame still input.
- Demo artifacts include saved input images, upstream/local depth visualizations, absdiff depth visualizations, contact sheet, `camera_summary.json`, `parity_summary.json`, and README labels.
- Depth, confidence, extrinsics, intrinsics, and selected taps pass explicit tolerances on the required real-image/video cases.
- Stale `da3_multiview` `UPSTREAM_PASSED` status in `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` is demoted before execution evidence is regenerated, then promoted only after corrected real-image/video gates pass.
- Docs, roadmap, parity status JSON, and Automaton state no longer claim DA3 is verified until the corrected gates pass.
- Full normal regression remains green.

**Verification:**
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL PYTHONPATH=references/Depth-Anything-3/src uv run --extra test --extra mlx --extra da3-reference pytest tests/test_da3_upstream_parity.py tests/test_da3_real_forward.py tests/test_da3_real_checkpoint_load.py tests/test_da3_multiview_model.py tests/test_da3_multiview_processor.py`
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest`
- `git diff --check`

**Execution:** subagent recommended

**Depends on:** Slice 10

**Touches:** `tools/da3_demo.py`, `tools/da3_upstream.py`, `tests/test_da3_upstream_parity.py`, DA3 docs/status files, `.agent/steering/ROADMAP.md`, this plan's terminal verification section.

**Produces:** Real DA3 parity evidence that can safely restore `verified` status when it passes.

## Aggregate Verification Commands

| Gate | Command |
| --- | --- |
| Corrective numeric/unit gate | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache PYTHONPATH=references/Depth-Anything-3/src uv run --extra test --extra mlx --extra da3-reference pytest tests/test_layers.py tests/test_dinov2_parity.py tests/test_da3_multiview_backbone.py tests/test_da3_real_architecture_contract.py tests/test_da3_real_checkpoint_load.py tests/test_da3_real_forward.py tests/test_da3_multiview_model.py tests/test_da3_convert.py` |
| Required DA3 parity gate | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL PYTHONPATH=references/Depth-Anything-3/src uv run --extra test --extra mlx --extra da3-reference pytest tests/test_da3_upstream_parity.py tests/test_da3_real_forward.py tests/test_da3_real_checkpoint_load.py tests/test_da3_multiview_model.py tests/test_da3_multiview_processor.py` |
| Full regression | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest` |

## Risks

- The custom PyTorch-compatible bicubic path must be tested carefully; a close-looking interpolation change can shift all downstream DA3 tensors.
- Strict-load semantics must not become a general missing-key escape hatch. Only documented upstream-default aux LayerNorm keys may be tolerated or synthesized.
- Real-image/video parity commands require MLX Metal access outside the Codex sandbox and upstream DA3 reference dependencies.
