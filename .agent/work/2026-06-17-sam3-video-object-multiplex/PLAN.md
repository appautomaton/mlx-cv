# PLAN: SAM 3.1 Video / Object Multiplex

Change: `2026-06-17-sam3-video-object-multiplex` - Stage: plan - Spec: `SPEC.md` - Design: `DESIGN.md`

## Goal

Execute the entire Phase 3 spec: add SAM3 Video, Sam3Tracker, Object Multiplex-aware tracking state, deterministic short-clip verification, and a real video-checkpoint gate or precise blocker.

## Architecture Approach

See `DESIGN.md`. Keep SAM3 image-mode conversion and prediction stable. Add video/tracker state as a separate SAM3 path, represent video outputs as typed per-frame `Result` objects, and keep upstream/reference execution out of runtime code.

## Execution Routing And Topology

- Default execution: direct, serial, continue through all slices when verification passes.
- Subagent recommended: Slices 1, 4, 5, and 6 because they cross upstream reference behavior, state semantics, or checkpoint-gate correctness.
- Checkpoints: none. If checkpoint prerequisites are missing, record a precise blocker and continue through docs/status work without claiming parity.
- Parallel-safe groups: none. Slices touch shared SAM3 exports, result types, tests, and docs.
- Phase 3 status artifact: `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`. Do not add `sam3_video` to `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`; that older matrix is intentionally bounded by `tests/test_runtime_dependency_guards.py::test_release_parity_status_matrix_is_bounded`.

## Ordered Slice Sequence

### Slice 1: SAM3 Video Reference Contract And Gate Skeleton

**Objective:** Capture the Phase 3 upstream contract and create the video checkpoint admission skeleton without changing runtime behavior.

**Acceptance criteria:**
- A local contract artifact names the relevant upstream surfaces: `build_sam3_video_predictor`, `build_sam3_multiplex_video_predictor`, `Sam3TrackerPredictor`, `SimpleMaskEncoder`, `MultiplexController`, `VideoTrackingDynamicMultiplex`, and `start_session` / `add_prompt` / `propagate_in_video`.
- The contract distinguishes SAM3 Video text/concept tracking from Sam3Tracker visual-prompt tracking and Object Multiplex grouping.
- `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json` records the SAM3 video gate status, checkpoint/config env names, reference path, and initial `BLOCKED:<reason>` when checkpoint access is absent.
- `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` is not modified or expanded for `sam3_video`.
- Existing SAM3 image-mode upstream blocker tests remain unchanged.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_upstream_parity.py tests/test_runtime_dependency_guards.py`

**Execution:** subagent recommended

**Touches:** `.agent/work/2026-06-17-sam3-video-object-multiplex/`, `tools/`, `tests/test_sam3_video_checkpoint_gate.py`, SAM3 status docs if needed.

**Produces:** A reloadable SAM3 video reference/gate contract and a testable initial blocker status.

**Status:** complete
**Evidence:** added `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-contract.md`, `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`, and `tests/test_sam3_video_checkpoint_gate.py`; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_upstream_parity.py tests/test_runtime_dependency_guards.py` passed with 8 passed, 1 skipped.
**Risks / next:** none.

### Slice 2: Typed Video Results, Tracks, And Multiplex State

**Objective:** Add the core typed surfaces needed to represent per-frame video tracking outputs and multiplex state.

**Acceptance criteria:**
- `Tracks` can represent stable object IDs, frame index, optional labels/scores, and per-object metadata without breaking existing callers.
- A typed video result collection represents ordered per-frame `Result` outputs and serializes to dictionaries.
- Object Multiplex state records fixed-capacity buckets, object-to-bucket assignment, active object IDs, and frame/memory metadata.
- Validation catches mismatched mask/detection/track lengths and invalid bucket assignments.
- Existing result serialization tests for detections, masks, depth, and image-mode SAM3 still pass.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_types.py tests/test_tracking.py tests/test_sam3_processor.py tests/test_runtime_dependency_guards.py`

**Touches:** `src/mlx_cv/core/types.py`, optional `src/mlx_cv/core/tracking.py`, `src/mlx_cv/core/__init__.py`, `src/mlx_cv/__init__.py`, `tests/test_types.py`, `tests/test_tracking.py`.

**Produces:** Typed output and state primitives for video tracking.

**Status:** complete
**Evidence:** added `VideoResult`, extended `Tracks`, added `src/mlx_cv/core/tracking.py`, and covered result/tracking validation in `tests/test_types.py` and `tests/test_tracking.py`; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_types.py tests/test_tracking.py tests/test_sam3_processor.py tests/test_runtime_dependency_guards.py` passed outside the sandbox with Metal access: 32 passed.
**Risks / next:** none.

### Slice 3: Frame Sequence Processor And Session API

**Objective:** Add the SAM3 video input/session surface for deterministic frame sequences and prompt admission.

**Acceptance criteria:**
- A SAM3 video processor accepts an ordered frame sequence from arrays, image paths, or a frame directory and records per-frame image sizes/transforms.
- Video-file decoding is optional and does not add a hard runtime dependency.
- A local session API supports `start_session`, `add_prompt`, and `propagate_in_video` method names with typed request/response behavior.
- Text/concept prompts route through the SAM3 Video boundary; visual prompts route through the Sam3Tracker boundary where supported.
- Unsupported point, mask, or video prompt branches fail with precise errors until implemented by later slices.
- Existing one-image `SAM3Processor` behavior and deferred prompt-state errors remain intact.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_prompts.py tests/test_sam3_processor.py`

**Depends on:** Slice 2

**Touches:** `src/mlx_cv/models/sam3/`, SAM3 exports, `tests/test_sam3_video_processor.py`, `tests/test_sam3_video_session.py`, fixture frame helpers.

**Produces:** A typed SAM3 video session and deterministic frame-sequence preprocessing path.

**Status:** complete
**Evidence:** added `src/mlx_cv/models/sam3/video.py` with frame-sequence preprocessing, prompt classification, session state, and upstream-style request handling; exported video types from `src/mlx_cv/models/sam3/__init__.py`; added `tests/test_sam3_video_processor.py` and `tests/test_sam3_video_session.py`; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_prompts.py tests/test_sam3_processor.py` passed outside the sandbox with Metal access: 18 passed.
**Risks / next:** `propagate_in_video` intentionally raises until Slice 4 adds tracker memory propagation.

### Slice 4: Deterministic Tracker Memory And Short-Clip Propagation

**Objective:** Implement the local tracker/memory path that produces stable IDs and masks over a fixed short clip.

**Acceptance criteria:**
- Session propagation updates per-object memory records across frames and returns ordered per-frame results.
- A deterministic short-clip fixture covers text/concept initialization and produces stable track IDs and masks over at least three frames.
- Visual-prompt tracking through the Sam3Tracker boundary is implemented for the supported prompt type, or fails with a precise upstream-grounded blocker and test.
- Propagation tests prove that `Result.masks`, `Result.tracks`, and optional `Result.detections.track_ids` agree for each frame.
- The deterministic tracker fixture is labeled as local contract coverage, not upstream parity.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_tracking.py tests/test_sam3_video_session.py tests/test_types.py`

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `src/mlx_cv/models/sam3/`, optional `src/mlx_cv/core/tracking.py`, `tests/fixtures/`, `tests/test_sam3_video_tracking.py`.

**Produces:** Runnable short-clip SAM3 video tracking with stable IDs and memory updates.

**Status:** complete
**Evidence:** extended `src/mlx_cv/models/sam3/video.py` so sessions propagate deterministic local text and visual prompts into per-frame `VideoResult` outputs with aligned `Masks`, `Detections.track_ids`, `Tracks`, and `TrackMemoryRecord` entries; added `tests/test_sam3_video_tracking.py` and updated video session tests; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_tracking.py tests/test_sam3_video_session.py tests/test_types.py` passed outside the sandbox with Metal access: 27 passed.
**Risks / next:** deterministic tracking remains local contract coverage, not upstream parity.

### Slice 5: Object Multiplex Grouping And Multi-Object Propagation

**Objective:** Integrate Object Multiplex-aware bucket assignment into multi-object video tracking.

**Acceptance criteria:**
- Multiple tracked objects are assigned into fixed-capacity multiplex buckets with deterministic metadata.
- Bucket state updates when objects are added, removed, or propagated.
- Multi-object propagation exercises the multiplex path rather than a hidden single-object-only loop.
- Tests cover at least two objects, at least two buckets or a full-bucket boundary, and stable object IDs through propagation.
- No local docs or comments claim upstream Object Multiplex speedups from this shape-level coverage.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_object_multiplex.py tests/test_sam3_video_tracking.py tests/test_types.py`

**Execution:** subagent recommended

**Depends on:** Slice 4

**Touches:** multiplex state/controller modules, SAM3 video tracking modules, `tests/test_sam3_object_multiplex.py`, fixtures.

**Produces:** Object Multiplex-aware local state and multi-object tracking coverage.

**Status:** complete
**Evidence:** wired `ObjectMultiplexState` into `SAM3VideoSessionState`, assigned prompt objects to fixed-capacity buckets, recorded bucket metadata in per-frame `Tracks`, supported `remove_object`, and added `tests/test_sam3_object_multiplex.py`; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_object_multiplex.py tests/test_sam3_video_tracking.py tests/test_types.py` passed outside the sandbox with Metal access: 25 passed.
**Risks / next:** Object Multiplex coverage is shape/state coverage only, not speed or upstream numeric parity.

### Slice 6: Real SAM3 Video Checkpoint Gate

**Objective:** Add the required real-checkpoint admission gate for SAM3 video/tracker/multiplex checkpoints.

**Acceptance criteria:**
- The gate defines env/config inputs for video checkpoint, optional config/model ID, and required-mode execution.
- The image-mode loader still rejects video/tracker checkpoints through the existing `_VIDEO_KEY_PARTS` image-loader rejection, while the video gate recognizes expected video/tracker/multiplex key families through a separate admission path.
- Default tests assert a precise `BLOCKED:<reason>` when checkpoint/config access is absent.
- Required-mode tests fail with a precise blocker when prerequisites are absent and run the upstream-vs-local comparison path when prerequisites are present.
- Optional upstream reference execution stays in `tools/` or tests and does not add Torch/OpenCV/xformers/CUDA dependencies to runtime imports.
- Gate output updates `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json` without advertising parity unless the real comparison passes.
- The older release-parity status file remains bounded to its existing model set: `da3_multiview`, `locateanything`, `rfdetr`, and `sam3_image`.

**Verification:**
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_convert.py tests/test_runtime_dependency_guards.py`
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py`

**Execution:** subagent recommended

**Depends on:** Slice 5

**Touches:** `src/mlx_cv/models/sam3/convert.py`, SAM3 video gate helpers, `tools/sam3_video_upstream.py`, `tests/test_sam3_video_checkpoint_gate.py`, `tests/test_sam3_video_upstream_parity.py`, `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`.

**Produces:** Truthful SAM3 video checkpoint gate with pass-or-blocker semantics.

**Status:** complete
**Evidence:** added separate `inspect_sam3_video_state_dict` video checkpoint inspection while keeping image-mode conversion rejection intact, added `tools/sam3_video_upstream.py`, updated the Phase 3 status artifact, and added default/required gate tests; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_convert.py tests/test_runtime_dependency_guards.py` passed outside the sandbox with Metal access: 15 passed; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py` passed: 2 passed.
**Risks / next:** real upstream-vs-local video parity remains externally blocked until `MLX_CV_SAM3_VIDEO_CHECKPOINT` points to a usable checkpoint and the numeric comparison path exists.

### Slice 7: Docs, Roadmap, And Final Regression

**Objective:** Document the claim level reached and close Phase 3 without overstating checkpoint parity.

**Acceptance criteria:**
- SAM3 docs describe supported video inputs, session API, output shape, Object Multiplex state, checkpoint env variables, and current claim level.
- `.agent/steering/ROADMAP.md` marks Phase 3 complete only if the local short-clip contract passes and the video checkpoint gate has either a real pass or precise external blocker.
- `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json` distinguishes local deterministic fixture coverage from upstream video parity.
- The release-parity hardening matrix remains unchanged and bounded to its existing model set.
- Full SAM3 image and video tests pass, plus the normal regression suite.
- `git diff --check` passes.

**Verification:**
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_tokenizer.py tests/test_sam3_prompts.py tests/test_sam3_processor.py tests/test_sam3_predict.py tests/test_sam3_parity.py tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_video_tracking.py tests/test_sam3_object_multiplex.py tests/test_sam3_video_checkpoint_gate.py`
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest`
- `git diff --check`

**Depends on:** Slice 6

**Touches:** `docs/`, `.agent/steering/ROADMAP.md`, `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`, README snippets if needed.

**Produces:** Final Phase 3 documentation, roadmap state, and regression evidence.

**Status:** complete
**Evidence:** added `docs/sam3-video.md`, updated `README.md`, `.agent/steering/ROADMAP.md`, `.agent/steering/PROJECT.md`, and `.agent/steering/REQUIREMENTS.md` to distinguish local SAM3 video/Object Multiplex contract coverage from blocked upstream video parity; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_tokenizer.py tests/test_sam3_prompts.py tests/test_sam3_processor.py tests/test_sam3_predict.py tests/test_sam3_parity.py tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_video_tracking.py tests/test_sam3_object_multiplex.py tests/test_sam3_video_checkpoint_gate.py` passed outside the sandbox with Metal access: 40 passed; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest` passed outside the sandbox with Metal access: 435 passed, 10 skipped; `git diff --check` passed.
**Risks / next:** upstream SAM3 video parity remains blocked by the external checkpoint/comparison path, recorded in `sam3-video-status.json`.

## Aggregate Verification Commands

| Gate | Command |
| --- | --- |
| Local SAM3 video contract | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_video_tracking.py tests/test_sam3_object_multiplex.py tests/test_types.py` |
| SAM3 video checkpoint gate | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_convert.py tests/test_runtime_dependency_guards.py` |
| Required upstream gate | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py` |
| Full regression | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest` |

## Risks

- Upstream SAM 3.1 video/multiplex checkpoint access may remain unavailable. That is acceptable only if the required gate records a precise blocker and docs avoid parity claims.
- The deterministic tracker path proves local state and output contracts, not upstream model quality.
- Extending `core.types` must preserve import-light behavior and existing serialization compatibility.
- Object Multiplex shape tests must not be mistaken for measured performance parity.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: Additive design that keeps image-mode untouched, reuses the established per-model env-gated real-checkpoint pattern, and is backed by machine-enforced runtime guards (`tests/test_runtime_dependency_guards.py` blocks torch/references imports in `src/mlx_cv`).
- Concern: Slice 6 is the riskiest because the video gate must recognize the exact `video/tracker/memory/temporal` key families that `convert.py:_reject_unsupported_variant` still hard-rejects for image-mode, so building the gate by loosening that rejection instead of adding a parallel path would silently regress image-mode conversion.
- Action: In Slice 6 add the video gate as a separate admission path, keep `_VIDEO_KEY_PARTS` rejection intact for the image loader, and route the SAM3 video status into this change's own status artifact only — do not register `sam3_video` in `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, whose exact model set is asserted by `test_release_parity_status_matrix_is_bounded`.
- Verified: Upstream surfaces in Slice 1 confirmed present (`build_sam3_video_predictor`/`build_sam3_multiplex_video_predictor` in `model_builder.py`, `Sam3TrackerPredictor`/`SimpleMaskEncoder`/`MultiplexController`/`VideoTrackingDynamicMultiplex`, and `start_session`/`add_prompt`/`propagate_in_video` in `sam3_base_predictor.py`); fixture clip `references/sam3/assets/videos/0001/` exists (270 frames); `core/types.py` `Tracks` is minimal and `Result.to_dict()` has no `tracks` branch (Slice 2 is additive, not a regression on serialization); image-mode rejection in `prompts.py`/`processor.py`/`convert.py` confirmed; data flow traced frame-seq→session→memory→per-frame `Result`→typed collection; rollback safety (all additive) and test-before-code strategy reviewed.
