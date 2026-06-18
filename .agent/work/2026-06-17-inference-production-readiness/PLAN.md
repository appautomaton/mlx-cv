# PLAN: Inference Production Readiness For Remaining Model Families

Change: `2026-06-17-inference-production-readiness` - Stage: plan - Spec: `SPEC.md`

## Goal

Bring LocateAnything, SAM3 image, and SAM3 video to checkpoint-ready production status (real MLX inference + honest upstream-vs-MLX parity harness) per `SPEC.md`. Verified PASS is gated on user-supplied weights; everything reachable without weights is delivered here.

## Architecture Approach

- **Comparison harnesses (LA-CMP, SAM3I-CMP, SAM3V-CMP)** mirror the proven working pattern in `tools/da3_upstream.py` / `tests/test_da3_upstream_parity.py`: torch reference capture (tools/tests only) + local capture + numeric compare over stable taps, with a required-mode gate that returns a precise component-specific blocker (never a fake pass). Committed tests exercise the comparison logic with tiny fixtures / mocked reference and the honest-blocker default branch — no real weights.
- **SAM3 video neural port (SAM3V-NN)** replaces the deterministic `_propagate_frame` stand-in with the real memory-encoder + tracker/mask-decoder inference path ported from `references/sam3/sam3/model/` (`sam3_video_base.py`, `sam3_tracker_base.py`, `memory.py`, `multiplex_mask_decoder.py`). Inference path only — no training, no reference imports in `src/`. The audit slice produces the concrete module map and the porting sub-decomposition.
- All new tensor compute stays under `models/`/`backbones/`/`heads/`; `core/` and top-level runtime stay MLX-native and import-light.

## Requirement Traceability

| SPEC gap ID / AC | Satisfying slices |
|---|---|
| LA-CMP (AC1) | Slice 1 |
| SAM3I-CMP (AC2) | Slice 2 |
| SAM3V-NN (AC3) | Slices 3, 4, 5 |
| SAM3V-CMP (AC4) | Slice 6 |
| Honest parity matrix (AC5) | Slice 7 |
| Runtime hygiene (AC6) | all slices; final gate in Slice 7 |
| Checkpoint-ready commands (AC7) | each harness slice; consolidated in Slice 7 |

## Ordered Slice Sequence

### Slice 1: LA-CMP — Complete LocateAnything Upstream Comparison Harness

**Objective:** Implement real upstream-vs-MLX comparison in `tools/locateanything_upstream.py` for decoded boxes/points and stable taps, with an honest blocker when capture/checkpoint is unavailable.
**Acceptance criteria:**
- `evaluate_locateanything_comparison_gate` performs real reference capture + local capture + numeric compare (documented tolerances) when a checkpoint and reference runtime are present; otherwise returns a precise component-specific blocker (not the current "component missing" stub, not a fake pass).
- A committed test drives the comparison logic on a tiny fixture / mocked reference (no real weights) and asserts the honest-blocker default branch.
- No torch/transformers/`references/` imports added to `src/mlx_cv/`.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_runtime_dependency_guards.py -q`
**Execution:** subagent recommended
**Touches:** `tools/locateanything_upstream.py`, `tests/test_la_upstream_parity.py`, possibly `tools/mint_locateanything_fixture.py`.

**Status:** complete
**Evidence:** changed `tools/locateanything_upstream.py` and `tests/test_la_upstream_parity.py`; subagent spec review APPROVED after fixing admitted-only metadata; quality review APPROVED after making mocked gate tests independent of the ignored reference checkout; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_runtime_dependency_guards.py -q` passed with 10 passed, 1 skipped.
**Risks / next:** none for Slice 1.

### Slice 2: SAM3I-CMP — Complete SAM3 Image Upstream Comparison Harness

**Objective:** Implement stable image tap capture + numeric comparison in `tools/sam3_image_upstream.py` for masks, paired detections, and token/text evidence, with an honest blocker when unavailable.
**Acceptance criteria:**
- `evaluate_sam3_image_comparison_gate` performs real reference + local capture + compare over masks/detections/token-text taps with documented tolerances when checkpoint/runtime present; otherwise a precise component-specific blocker (replaces the current stub).
- Committed tiny-fixture/mock test of the comparison logic plus an honest-blocker default test.
- SAM3 image-vs-video checkpoint rejection unchanged.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_upstream_parity.py tests/test_sam3_predict.py tests/test_runtime_dependency_guards.py -q`
**Execution:** subagent recommended
**Touches:** `tools/sam3_image_upstream.py`, `tests/test_sam3_upstream_parity.py`, possibly `tools/mint_sam3_fixture.py`.

**Status:** complete
**Evidence:** changed `tools/sam3_image_upstream.py` and `tests/test_sam3_upstream_parity.py`; subagent spec review APPROVED after adding separate upstream/local checkpoint resolution; quality review APPROVED after normalizing upstream mask shape and canonicalizing detection comparison; sandboxed verification hit MLX Metal collection failure, then escalated verification `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_upstream_parity.py tests/test_sam3_predict.py tests/test_runtime_dependency_guards.py -q` passed with 18 passed, 1 skipped.
**Risks / next:** none for Slice 2.

### Slice 3: SAM3V-NN-audit — SAM3 Video Real-Inference Architecture Audit + Port Decomposition

**Objective:** Map the reference SAM3.1 video inference path (memory encoder, tracker/mask decoder, video forward) to a local MLX module plan, with a fixture/tap plan and an explicit porting sub-decomposition.
**Acceptance criteria:**
- An audit artifact (`.agent/work/2026-06-17-inference-production-readiness/sam3-video-port-map.md`) maps each reference surface (`sam3_video_base.py`, `sam3_tracker_base.py`, `memory.py`, `multiplex_mask_decoder.py`) to a planned local module/owner, names the inference-only boundary (no training), the tiny-fixture shape, the comparison taps, and an ordered sub-slice list for Slices 4–5.
- No `src/mlx_cv/` behavior change in this slice; release parity matrix still bounded.
**Verification:** `test -f .agent/work/2026-06-17-inference-production-readiness/sam3-video-port-map.md && rg -n "memory|tracker|mask decoder|tap|fixture|inference-only|sub-slice" .agent/work/2026-06-17-inference-production-readiness/sam3-video-port-map.md && git diff --name-only HEAD -- src/mlx_cv | (! grep .)`
**Execution:** subagent recommended
**Produces:** Port map + refined Slice 4–5 sub-decomposition (plan may be refined from this evidence).

**Status:** complete
**Evidence:** created `.agent/work/2026-06-17-inference-production-readiness/sam3-video-port-map.md`; subagent spec review APPROVED; quality review APPROVED after adding the Object Multiplex runtime layer (`multiplex_utils.py`, `video_tracking_multiplex.py`, demo/runtime state, multiplex base/tracking wrappers), bucket mux/demux fixture shapes, and object/bucket-space taps; `test -f .agent/work/2026-06-17-inference-production-readiness/sam3-video-port-map.md && rg -n "memory|tracker|mask decoder|tap|fixture|inference-only|sub-slice" .agent/work/2026-06-17-inference-production-readiness/sam3-video-port-map.md && git diff --name-only HEAD -- src/mlx_cv | (! grep .)` passed.
**Risks / next:** Slices 4–5 must implement the multiplex-first tracker path, not the non-multiplex ancestor alone.

### Slice 4: SAM3V-NN-modules — Port Memory Encoder + Tracker/Mask Decoder Neural Modules

**Objective:** Implement the MLX memory-encoder and tracker/mask-decoder modules (inference path) per the Slice 3 map, with tiny-fixture forward tests using random weights.
**Acceptance criteria:**
- New MLX modules produce correct output shapes on a tiny fixture (random weights, no real checkpoint); focused unit tests pass.
- Modules live under `models/sam3/` (or `heads/`/`backbones/` as the map dictates); no torch/`references/` imports in `src/`.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/ -k "sam3 and (memory or tracker or video_model)" tests/test_runtime_dependency_guards.py -q`
**Execution:** subagent recommended
**Depends on:** Slice 3
**Detail:** `slices/` breakdown may be produced by Slice 3 if the port exceeds one session.

**Status:** complete
**Evidence:** added isolated MLX SAM3 video/Object Multiplex modules in `src/mlx_cv/models/sam3/{multiplex_state.py,video_memory.py,multiplex_decoder.py,video_tracking.py,video_model.py}` plus config/export updates and `tests/test_sam3_video_model_modules.py`; subagent spec review APPROVED; quality review APPROVED after fixing reduced eval-capacity bucket packing and making `SAM3VideoModel` an `nn.Module` with a parameter tree containing `tracker`; `git diff --check` passed; sandboxed Slice 4 selector hit MLX Metal access failure, then escalated `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/ -k "sam3 and (memory or tracker or video_model)" tests/test_runtime_dependency_guards.py -q` passed with 16 passed, 472 deselected; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_runtime_dependency_guards.py -q` passed with 5 passed.
**Risks / next:** Slice 5 must wire these modules into the session/tracker runtime and add focused converter/load tests before propagation verification.

### Slice 5: SAM3V-NN-wire — Replace Deterministic Propagation With Real Neural Forward + Converter/Load

**Objective:** Wire the real modules into the video propagation path (replacing `_deterministic_box`/`_box_mask`), add converter/load for the real checkpoint keys, and keep the streaming `SAM3VideoTracker` and session API working.
**Acceptance criteria:**
- The real video path produces model-derived masks; `claim_level` for the real path is no longer `local_contract_fixture`.
- `SAM3VideoSessionManager` (`propagate_in_video`/`handle_request`) and `SAM3VideoTracker` (`init`/`step`) tests pass against the real path on a tiny fixture; image-vs-video rejection preserved.
- Converter/load maps reference keys explicitly; unsupported variants fail with clear errors.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_tracking.py tests/test_sam3_video_tracker.py tests/test_sam3_object_multiplex.py -q`
**Execution:** subagent recommended
**Depends on:** Slice 4

### Slice 6: SAM3V-CMP — Complete SAM3 Video Upstream Comparison Harness

**Objective:** Complete `tools/sam3_video_upstream.py` to compare the real local neural outputs against the upstream reference within documented tolerances, with an honest component-specific blocker when checkpoint/runtime is missing.
**Acceptance criteria:**
- Comparison runs over real local neural outputs (from Slice 5) vs upstream taps with documented tolerances when checkpoint/runtime present; otherwise a precise blocker.
- Committed tiny-fixture/mock test plus honest-blocker default test.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_runtime_dependency_guards.py -q`
**Execution:** subagent recommended
**Depends on:** Slice 5

### Slice 7: Parity-Matrix Truthfulness + Final Hygiene

**Objective:** Update `parity-status.json` to state each family's literal remaining blocker, decide `sam3_video` matrix membership explicitly, document checkpoint-ready commands, and run the full regression gate.
**Acceptance criteria:**
- `parity-status.json` status strings reflect the post-change truth (LA/SAM3-image blocked only on external checkpoint; SAM3 video membership decided with a stated reason); no family claims `UPSTREAM_PASSED` without a real reference comparison.
- Each family's required-mode checkpoint-ready command is documented (status artifact or docs).
- Full `uv run pytest` green; `test_runtime_dependency_guards.py` passes; `git diff --check` clean.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest -q && python -c "import json; s=json.load(open('.agent/work/2026-06-16-release-parity-hardening/parity-status.json')); print('models:', sorted(s['models']))" && git diff --check`
**Depends on:** Slices 1, 2, 6

## Execution Routing And Topology

- **Default execution:** direct, serial; continue through all approved slices once each slice's verification passes. Execution windows are context batches, not planned stops.
- **Subagent routes:** Slices 1, 2, 3, 4 recommended for subagent (broad, cross-subsystem reference-capture and neural-port work that benefits from isolated context). Slices 5–7 direct unless context pressure warrants delegation.
- **Checkpoints:** none. Slice 3's audit may refine the Slice 4–5 sub-decomposition; that is a plan refinement recorded as evidence, not a human checkpoint.
- **Parallel-safe groups:** none. Run serially per the project's non-aggressive-concurrency preference (other agents may touch the repo).
- **External access:** real checkpoints are user-supplied and out-of-git; verified PASS runs outside the sandbox. No weights are downloaded or committed in this change.

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| LA harness | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_la_upstream_parity.py -q` |
| SAM3 image harness | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_upstream_parity.py -q` |
| SAM3 video real path | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_tracking.py tests/test_sam3_video_tracker.py -q` |
| SAM3 video harness | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_upstream_parity.py -q` |
| Runtime hygiene | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_runtime_dependency_guards.py -q` |
| Full regression | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest -q` |
| Diff hygiene | `git diff --check` |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The slice order keeps tools-only upstream reference capture separate from MLX runtime code and inserts an audit slice before the SAM3 video neural port.
- Concern: Slice 5 is the riskiest slice because checkpoint key conversion/load and neural propagation wiring can fail without being fully covered by the listed slice verification command.
- Action: In Slice 5, add focused converter/load tests for accepted reference key maps, unsupported variants, shape mismatches, and session/tracker metadata before running the listed propagation tests.
- Verified: canonical plan read; no DESIGN.md configured; DA3 parity pattern, LocateAnything/SAM3 stubs, SAM3 video session/tracker runtime, checkpoint converter guard, runtime dependency guard, and upstream SAM3 reference surfaces checked.
