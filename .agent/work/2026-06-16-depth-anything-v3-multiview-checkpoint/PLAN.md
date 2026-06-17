# PLAN: Depth Anything 3 Multi-View Real Checkpoint

Change: `2026-06-16-depth-anything-v3-multiview-checkpoint` - Stage: plan - Spec: `SPEC.md`

## Goal

Implement `SPEC.md`: DA3 any-view multi-view depth/camera inference with a real Small/Base checkpoint resolver, local MLX conversion/load, upstream-vs-MLX parity gate, truthful docs/status, and visible demo evidence.

## Architecture Approach

Use `DESIGN.md` as the architecture contract. Keep upstream DA3, Torch, OpenCV, torchvision, and Hugging Face Hub in `tools/` and env-gated tests. Keep `src/mlx_cv` runtime-clean and focused on MLX/NumPy/Pillow-compatible loading, preprocessing, model execution, and typed outputs. Target `depth-anything/DA3-SMALL` first; fallback to `depth-anything/DA3-BASE` only with recorded reason.

## Execution Routing And Topology

- Default path: serial execution through all slices after approval.
- Subagent recommended: Slices 2, 3, 5, 6, and 8 because they cross upstream DA3, checkpoint architecture, converter coverage, local MLX numerics, and parity.
- Checkpoints: none. Download/network and MLX execution may require sandbox escalation, but they are execution mechanics, not product decisions.
- Parallel-safe groups: none.

## Requirement Traceability

| Requirement | Slices |
| --- | --- |
| R1 / AC1 / AC2 checkpoint resolver and required gate | 1 |
| R2 / AC3 upstream fixed multi-view capture | 2 |
| R3 real architecture contract | 3 |
| R4 / AC4 runtime multi-view output contract | 4, 5, 6 |
| R5 / AC5 conversion and strict real load | 7 |
| R6 / AC6 upstream-vs-MLX parity | 8 |
| R7 / AC7 / AC8 docs, status, regression | 8 |
| AC9 visible demo artifact | 8 |

## Ordered Slice Sequence

### Slice 1: DA3 Checkpoint Resolver And Provenance Gate

**Objective:** Add an explicit DA3 Small/Base checkpoint preparation path that resolves `config.json` and `model.safetensors` from env/cache or opt-in download and records provenance.

**Acceptance criteria:**
- Resolver supports `MLX_CV_DA3_MODEL_ID`, `MLX_CV_DA3_CHECKPOINT`, `MLX_CV_DA3_CONFIG`, cache root, `--download`, and `--required`.
- Primary target defaults to `depth-anything/DA3-SMALL`; fallback to `depth-anything/DA3-BASE` requires an explicit message.
- Successful resolution prints model id, local paths, license/provenance, revision or commit, and hash/Xet/SHA evidence.
- Normal no-checkpoint mode skips cleanly; `MLX_CV_REQUIRE_DA3_GATE=1` fails if any required artifact is unavailable.
- Raw and converted weights remain outside git.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_da3_checkpoint_gate.py tests/test_runtime_dependency_guards.py`

**Execution:** direct

**Touches:** `tools/da3_checkpoint.py`, `tests/test_da3_checkpoint_gate.py`, `.gitignore` only if a new local cache pattern needs protection.

**Produces:** Reproducible DA3 checkpoint/config/provenance contract for later slices.

**Status:** complete
**Evidence:** added `tools/da3_checkpoint.py` and `tests/test_da3_checkpoint_gate.py`; resolver supports explicit env paths, model id selection, out-of-git cache layout, required-vs-optional behavior, opt-in download, SHA-256 provenance, license evidence, and evidence printing; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run pytest tests/test_da3_checkpoint_gate.py tests/test_runtime_dependency_guards.py` passed with 14 tests.
**Risks / next:** none; proceed to Slice 2 upstream fixed three-view capture.

### Slice 2: Upstream DA3 Multi-View Capture

**Objective:** Add an env-gated upstream DA3 runner that captures fixed three-view reference outputs and comparable taps from the selected real checkpoint.

**Acceptance criteria:**
- Runner uses the Slice 1 resolver and imports upstream DA3 only under `tools/` or env-gated tests.
- Fixed input contains exactly three deterministic same-size still views, preserves view order, and exercises upstream reference-view selection; choose images that force non-first reference selection when feasible.
- Capture prefers CPU/float32 and disables or records upstream autocast behavior to avoid unmeasured mixed-precision drift.
- Capture records processed image shape, depth `(V,H,W)`, confidence `(V,H,W)`, extrinsics `(V,3,4)` or `(V,4,4)`, intrinsics `(V,3,3)`, and selected taps.
- Required mode fails on missing checkpoint/config/reference dependencies or skipped capture.
- Normal no-checkpoint CI skips without importing upstream DA3 into runtime.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL PYTHONPATH=references/Depth-Anything-3/src uv run --extra test --extra da3-reference pytest tests/test_da3_upstream_capture.py tests/test_da3_checkpoint_gate.py`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tools/da3_upstream.py`, `tests/test_da3_upstream_capture.py`, `src/mlx_cv/parity/fixtures.py` if a fixed multi-view helper belongs there.

**Produces:** Real upstream oracle for DA3 multi-view depth/camera output.

**Status:** complete
**Evidence:** added `tools/da3_upstream.py`, `tests/test_da3_upstream_capture.py`, and `da3_multiview_fixed_images()`; declared clean-env `test` and `da3-reference` extras; required DA3-SMALL gate passed with `17 passed, 1 warning`; normal no-checkpoint gate passed with `16 passed, 1 skipped`; CLI capture saved `/tmp/mlx-cv-da3-upstream-capture.npz` with depth/confidence `(3,112,112)`, extrinsics `(3,3,4)`, intrinsics `(3,3,3)`, taps `feat_layer_5/7/9/11`, CPU float32 autocast disabled, and recorded reference selector call `[[1]]`; spec and quality re-reviews approved.
**Risks / next:** real-checkpoint capture still depends on the external DA3 reference checkout and cached/resolved checkpoint; proceed to Slice 3 architecture contract.

### Slice 3: Real DA3 Architecture Contract

**Objective:** Turn the selected checkpoint/config into an executable architecture contract before adding local MLX modules.

**Acceptance criteria:**
- Contract names selected model id, DINOv2 variant (`vits` or `vitb`), out layers `[5,7,9,11]`, `alt_start=4`, `qknorm_start=4`, `rope_start=4`, `cat_token=True`, doubled `embed_dim*2` head input dimensions, DualDPT dimensions, camera encoder/decoder dimensions, camera pose utility dependencies, and unsupported branches.
- Contract groups checkpoint tensor keys by backbone, DualDPT, camera encoder, camera decoder, and excluded branches.
- Contract proves the existing monocular-only path cannot satisfy multi-view checkpoint loading because DA3 needs any-view DINOv2 behavior, DualDPT, camera tokens, camera geometry modules, and pose conversion utilities.
- Normal no-checkpoint CI skips; required mode fails on missing checkpoint/config/provenance.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL uv run --extra test --extra da3-reference pytest tests/test_da3_real_architecture_contract.py tests/test_runtime_dependency_guards.py`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tools/da3_real_architecture_contract.py`, `tests/test_da3_real_architecture_contract.py`, DA3 config helpers if needed.

**Produces:** Executable map of the real DA3 Small/Base inference contract.

**Status:** complete
**Evidence:** added `tools/da3_real_architecture_contract.py` and `tests/test_da3_real_architecture_contract.py`; contract validates DA3-SMALL config/provenance, names DINOv2 `vits`, `out_layers [5,7,9,11]`, `alt/qknorm/rope_start=4`, `cat_token=True`, DualDPT/camera dimensions, pose utility dependencies, unsupported branches, complete grouped tensor coverage (`437/437` required), and monocular-path gaps; clean required gate `UV_PROJECT_ENVIRONMENT=/tmp/mlx-cv-slice3-clean-reference-venv4 UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL uv run --extra test --extra da3-reference pytest tests/test_da3_real_architecture_contract.py tests/test_runtime_dependency_guards.py` passed with 10 tests; spec and quality reviews approved.
**Risks / next:** DA3-BASE is table-supported but not live-checkpoint exercised here; proceed to Slice 4 public multi-view result/processor contract.

### Slice 4: Multi-View Result And Processor Contract

**Objective:** Extend the typed result and DA3 processor surface for still-image multi-view depth/confidence and camera geometry while preserving monocular behavior.

**Acceptance criteria:**
- Public preprocessing accepts a list of still images and optional per-view extrinsics/intrinsics.
- Postprocessing returns view-ordered depth/confidence and camera geometry through explicit `Result.depth_views: list[DepthMap] | None` and `Result.camera_geometry: CameraGeometry | None` fields while preserving existing `Result.depth` semantics.
- Existing `DA3Processor` monocular tests and `Result.to_dict()` behavior still pass.
- Invalid view-axis, camera shape, or mixed-size cases fail with clear errors unless represented as explicit per-view `DepthMap` entries.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_da3_processor.py tests/test_da3_multiview_processor.py tests/test_types.py tests/test_da3_parity.py`

**Execution:** direct

**Depends on:** Slice 3

**Touches:** `src/mlx_cv/core/types.py`, `src/mlx_cv/models/depth_anything_v3/processor.py`, `tests/test_da3_multiview_processor.py`, `tests/test_types.py`.

**Produces:** Stable public output contract for DA3 image-set geometry.

### Slice 5: DA3 Any-View Backbone Admission

**Objective:** Add the DA3 Small/Base any-view DINOv2 feature path needed by the real checkpoint.

**Acceptance criteria:**
- Shared `Attention` supports optional per-head `qk_norm` for Q/K without changing default behavior.
- Shared `TransformerBlock` or a DA3-specific wrapper supports per-block `qk_norm` and DA3 RoPE conditioned by layer index.
- A DA3 any-view DINOv2 class or adapter accepts `(B,V,3,H,W)` and preserves `(B,V,N,C)` token layout.
- The backbone implements alternating local `(B*V,N,C)` and global `(B,V*N,C)` attention dispatch from `alt_start`.
- The backbone implements camera-token injection, reference-view selection with reorder/restore for three or more views, and `cat_token` output concatenation.
- The backbone emits features shaped `(B,V,N,embed_dim*2)` when `cat_token=True`, with upstream-compatible split normalization.
- Existing DINOv2 and RF-DETR DINOv2 tests remain green.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_dinov2_forward.py tests/test_dinov2_parity.py tests/test_dinov2_convert.py tests/test_da3_multiview_backbone.py tests/test_rfdetr_nano_backbone_projector.py`

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `src/mlx_cv/backbones/vision/dinov2/`, `src/mlx_cv/core/features.py`, `tests/test_da3_multiview_backbone.py`, existing DINOv2 tests.

**Produces:** View-aware DINOv2 feature contract for DA3 Small/Base.

### Slice 6: DualDPT And Camera Geometry Heads

**Objective:** Add the DA3 any-view depth/confidence and camera decoder path needed for multi-view inference.

**Acceptance criteria:**
- Local head path implements DA3 DualDPT as a new module with main depth/confidence and auxiliary ray/ray-confidence branches, UV positional embeddings, and multi-view feature reshaping.
- Camera encoder/decoder modules include pose encoding/decoding utilities, scalar-last quaternion conversion, FOV/intrinsics conversion, and affine inversion.
- Camera decoder produces 9D pose encoding and final extrinsics/intrinsics in the same convention and shape as upstream for the selected fixed input.
- Optional pose-conditioned input path is represented enough to validate shapes, even if final parity uses unconditioned camera prediction.
- Existing monocular DA3 model and DPT tests remain green.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_da3_model.py tests/test_da3_multiview_model.py tests/test_da3_convert.py tests/test_da3_parity.py`

**Execution:** subagent recommended

**Depends on:** Slices 4, 5

**Touches:** `src/mlx_cv/heads/dense/`, `src/mlx_cv/models/depth_anything_v3/`, `tests/test_da3_multiview_model.py`, `tests/test_da3_convert.py`.

**Produces:** Local MLX DA3 multi-view forward path with depth/confidence/camera tensors.

### Slice 7: Real Checkpoint Conversion, Strict Load, And Local Forward

**Objective:** Convert the selected real DA3 checkpoint into a runtime-clean local representation and prove strict local load plus real MLX forward.

**Acceptance criteria:**
- Converter maps all required backbone, DualDPT, camera encoder, and camera decoder tensors.
- Loader rejects unsupported Gaussian/nested/metric-only branches unless they are absent or explicitly excluded.
- Required real-load test converts or locates converted weights outside git and strict-loads the local model.
- Local real-forward test runs the fixed multi-view input and returns depth, confidence, extrinsics, and intrinsics with expected shapes.
- Normal no-checkpoint CI skips; required mode fails on missing checkpoint, missing conversion output, or skipped local forward.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL uv run --extra test pytest tests/test_da3_real_checkpoint_load.py tests/test_da3_real_forward.py tests/test_da3_convert.py tests/test_runtime_dependency_guards.py`

**Execution:** direct

**Depends on:** Slice 6

**Touches:** `src/mlx_cv/models/depth_anything_v3/convert.py`, `tools/da3_convert_checkpoint.py`, `src/mlx_cv/parity/da3_real.py`, `tests/test_da3_real_checkpoint_load.py`, `tests/test_da3_real_forward.py`.

**Produces:** Real DA3 Small/Base checkpoint load and local MLX inference path.

### Slice 8: Upstream Parity, Demo Evidence, And Status Truthfulness

**Objective:** Compare real upstream and local MLX DA3 outputs, produce visible demo evidence, and update docs/status without overstating unsupported DA3 branches.

**Acceptance criteria:**
- Required parity gate compares fixed-input depth, confidence, extrinsics, intrinsics, and selected taps with explicit tolerances and printed checkpoint evidence.
- Required gate fails on skip, missing upstream capture, missing local load, or drift beyond tolerance.
- Demo artifacts under `/tmp/mlx-cv-da3-demo/` include per-view depth visualizations and JSON camera/parity summaries.
- README, architecture docs, DA3 docs, roadmap/status files, and the release parity status JSON distinguish real DA3 multi-view pass from deferred DA3 streaming/nested/metric/3DGS work.
- Full normal suite passes without real checkpoint env vars.

**Verification:** Required gate: `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL PYTHONPATH=references/Depth-Anything-3/src uv run --extra test --extra da3-reference pytest tests/test_da3_upstream_parity.py tests/test_da3_real_forward.py tests/test_da3_real_checkpoint_load.py tests/test_da3_multiview_model.py tests/test_da3_multiview_processor.py`; full regression: `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest`

**Execution:** subagent recommended

**Depends on:** Slice 7

**Touches:** `tests/test_da3_upstream_parity.py`, `tools/da3_demo.py`, README/docs/status files, `.agent/steering/ROADMAP.md`.

**Produces:** Real DA3 upstream parity result, visible local evidence, and truthful project status.

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Normal suite | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest` |
| Required DA3 gate | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL PYTHONPATH=references/Depth-Anything-3/src uv run --extra test --extra da3-reference pytest tests/test_da3_checkpoint_gate.py tests/test_da3_upstream_capture.py tests/test_da3_real_architecture_contract.py tests/test_da3_multiview_processor.py tests/test_da3_multiview_backbone.py tests/test_da3_multiview_model.py tests/test_da3_real_checkpoint_load.py tests/test_da3_real_forward.py tests/test_da3_upstream_parity.py` |
| Status/docs sanity | `python -m json.tool .agent/work/2026-06-16-release-parity-hardening/parity-status.json && git diff --check` |

## Risks

- DA3 any-view is effectively a new DINOv2 forward variant, not a simple config swap over the existing local ViT path; Slice 3 must name missing primitives before Slices 5-7 proceed.
- DualDPT is a new multi-view/ray head, not a direct extension of the current local `DPTHead`.
- Upstream CPU reference may be slow even for Small. If it is too slow, execution should reduce fixed input resolution/views before widening parity tolerance.
- Camera pose parity may reveal convention differences (`w2c` vs `c2w`, `(3,4)` vs `(4,4)`). The processor contract must encode the convention rather than hiding the transform in tests.
- If `DA3-SMALL` checkpoint availability or license changes, fall back to `DA3-BASE` only with recorded evidence; do not silently move to Large/Giant/Nested.
