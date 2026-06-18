# PLAN: SAM 3.1 Video Real Checkpoint Admission

Change: `2026-06-17-sam3-video-real-checkpoint-admission` - Stage: plan - Spec: `SPEC.md` - Design: `DESIGN.md`

## Goal

Execute roadmap Phase 2 end to end: turn SAM 3.1 video/Object Multiplex from local deterministic contract coverage into a real-checkpoint admission gate that either passes against a usable checkpoint/config/reference path or records a precise blocker.

## Architecture Approach

See `DESIGN.md`. Add a new Phase 2 checkpoint-admission status artifact, keep the prior SAM 3.1 video local-contract status as historical evidence, keep `sam3_video` out of the release parity matrix, and keep all upstream/Torch/Hugging Face work in tools/tests. Required mode must never silently skip: it produces `UPSTREAM_PASSED` only after a real pass, otherwise a specific `BLOCKED:<reason>`.

## Requirement Traceability

| SPEC criteria | Satisfying slices |
|---|---|
| AC1, AC2, AC3, AC9 | Slice 1 |
| AC4, AC5 | Slice 2 |
| AC6 | Slice 3 |
| AC7, AC8, AC9 | Slice 4 |
| AC10, docs/status truth | Slice 5 |

## Execution Routing And Topology

- Default execution: direct, serial, continue through all slices when verification passes.
- Subagent recommended: Slices 2, 3, and 4 because they cross checkpoint source/cache, upstream reference execution, and local comparison blockers.
- Checkpoints: none committed. If checkpoint download or Hugging Face access is needed, request execution-time approval and write only to an out-of-git cache.
- Parallel-safe groups: none. Slices share `tools/sam3_video_upstream.py`, the new status artifact, and SAM 3.1 video tests.
- External access: optional only. Default tests must simulate admission states without network.

## Ordered Slice Sequence

### Slice 1: Phase Boundary, Status Contract, And Official Source

**Objective:** Make the Phase 2 boundary and SAM 3.1 video checkpoint-admission status contract executable before touching checkpoint/download logic.

**Acceptance criteria:**
- Roadmap Phase 2 is active with change `2026-06-17-sam3-video-real-checkpoint-admission`; Phase 1 stays done and Phase 3 stays pending.
- New status artifact path is `.agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json`.
- Status records official source metadata for `facebook/sam3.1`, `sam3.1_multiplex.pt`, `config.json`, terms/auth expectations, and env names.
- `tools/sam3_video_upstream.py` writes/reads the new Phase 2 status artifact and preserves the prior local-contract artifact as historical evidence only.
- `sam3_video` remains absent from `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`.
- Runtime dependency guards still prevent upstream/reference dependencies from entering package runtime.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_runtime_dependency_guards.py`

**Touches:** `.agent/steering/ROADMAP.md`, `tools/sam3_video_upstream.py`, `tests/test_sam3_video_upstream_parity.py`, `tests/test_sam3_video_checkpoint_gate.py`, `.agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json`.

**Produces:** A Phase 2 status contract with official source metadata and required-gate semantics.

**Status:** complete
**Evidence:** added the new Phase 2 checkpoint-admission status artifact path, official `facebook/sam3.1` / `sam3.1_multiplex.pt` / `config.json` source metadata, required/cache env fields, and local-contract status back-reference in `tools/sam3_video_upstream.py`; updated `tests/test_sam3_video_checkpoint_gate.py` to assert the new status contract; generated `.agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json`. Sandbox verification hit the known Metal-device limitation, then the same command passed outside the sandbox: `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_runtime_dependency_guards.py` passed with 11 tests.
**Risks / next:** Slice 2 still needs detailed checkpoint/config/cache admission states; current status is still the expected unset-checkpoint blocker.

### Slice 2: Checkpoint Download, Cache, And Admission

**Objective:** Turn SAM 3.1 video checkpoint/config availability into a precise, non-network-default admissibility decision.

**Acceptance criteria:**
- The gate distinguishes unset env, missing checkpoint, missing config, path shape errors, tiny/LFS-stub checkpoint, unsupported model ID, missing auth/download access, and usable checkpoint/config.
- Optional cache/download helper supports `facebook/sam3.1`, `sam3.1_multiplex.pt`, and `config.json`, writes outside git, and records provenance/checksum when files exist.
- Default tests use fake files and local fixtures only; no network or checkpoint access is required.
- If a user-provided cache is present, the gate records its path, source metadata, checksum/provenance status, and admission result.
- No upstream weights or derived full-checkpoint artifacts are committed.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tools/sam3_video_upstream.py`, `tests/test_sam3_video_upstream_parity.py`, `tests/test_sam3_video_checkpoint_gate.py`, `.agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json`.

**Produces:** SAM 3.1 video checkpoint/config admission evidence or a precise source/cache blocker.

**Status:** complete
**Evidence:** expanded `tools/sam3_video_upstream.py` with official model ID validation, explicit checkpoint/config admission, optional `MLX_CV_SAM3_VIDEO_CACHE_DIR` lookup under `facebook--sam3.1`, tiny/unusable checkpoint and config blockers, download/auth cache-miss wording, admitted checkpoint/config SHA256 recording, and `checkpoint_admitted` status metadata; updated `tests/test_sam3_video_upstream_parity.py` with fake-file coverage for missing config, missing path, unsupported model ID, uncached gated source, explicit admission, and cache admission. `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py` passed outside the sandbox with Metal access: 12 tests.
**Risks / next:** admitted checkpoint/config pairs still need to flow into the reference/runtime and local comparison blockers in Slices 3 and 4.

### Slice 3: Upstream Reference Capture Or Reference Blocker

**Objective:** Use the official SAM 3.1 reference path when possible and otherwise name the exact reference-side blocker.

**Acceptance criteria:**
- The gate audits `references/sam3/` for SAM 3.1 video/Object Multiplex surfaces: `build_sam3_predictor(version="sam3.1")`, `build_sam3_multiplex_video_predictor`, session start, prompt add, and video propagation.
- With admitted checkpoint/config and required dependencies, the gate attempts upstream reference execution on fixed tiny video inputs and stable prompt/session operations.
- Missing `references/sam3/`, missing Torch/reference dependencies, auth/runtime failure, upstream builder failure, and upstream output capture failure are distinct blockers.
- Reference code is imported only from tools/tests and never from `src/mlx_cv/`.
- Default tests can force the admitted-checkpoint path to a precise reference or comparison blocker without requiring real weights.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_runtime_dependency_guards.py`

**Execution:** subagent recommended

**Depends on:** Slice 2

**Touches:** `tools/sam3_video_upstream.py`, `tests/test_sam3_video_upstream_parity.py`, optional tool-only helpers under `tools/`.

**Produces:** Upstream SAM 3.1 video/Object Multiplex reference pass evidence or a precise reference-runtime blocker.

**Status:** complete
**Evidence:** added `evaluate_sam3_video_reference_gate` in `tools/sam3_video_upstream.py`, preserving checkpoint/config SHA evidence after admission and separating missing reference path, missing Object Multiplex surfaces, missing Torch/reference runtime, and incomplete upstream output capture blockers; added tests in `tests/test_sam3_video_upstream_parity.py` that force each branch with fake admitted checkpoint/config files. `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_runtime_dependency_guards.py` passed outside the sandbox with Metal access: 21 tests.
**Risks / next:** reference capture remains a precise blocker unless real SAM3.1 checkpoint access and upstream runtime execution are available; Slice 4 must route admitted/reference-ready paths to local converter/tap/comparator blockers instead of generic wording.

### Slice 4: Local Comparison Boundary And Component Blockers

**Objective:** Replace the current checkpoint-present "comparison not implemented" branch with either the smallest credible comparison or a component-specific blocker.

**Acceptance criteria:**
- When upstream outputs are available, the gate compares the smallest stable local outputs available from the current SAM 3.1 video implementation.
- Missing MLX checkpoint conversion, missing local tap capture, missing output mapper, unsupported checkpoint branch, numeric mismatch, and pass are distinct outcomes.
- A default-mode unit test creates a fake admitted checkpoint/config branch and asserts the exact component-specific blocker.
- No checkpoint-present path contains a bare `pytest.fail("comparison is not implemented")` or equivalent generic failure.
- Existing SAM 3.1 video session/tracking/Object Multiplex tests continue to pass.
- Existing SAM 3.1 image-mode tests continue to reject video/tracker checkpoints.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_video_tracking.py tests/test_sam3_object_multiplex.py tests/test_sam3_upstream_parity.py tests/test_sam3_convert.py`

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `tools/sam3_video_upstream.py`, `tests/test_sam3_video_upstream_parity.py`, `tests/test_sam3_video_checkpoint_gate.py`, `src/mlx_cv/models/sam3/video.py` only if a small comparison hook is necessary and consistent with the existing API.

**Produces:** SAM 3.1 video real-checkpoint comparison pass or a precise local converter/tap/comparator blocker.

**Status:** complete
**Evidence:** added `evaluate_sam3_video_comparison_gate` in `tools/sam3_video_upstream.py`, made the CLI/status writer use the comparison gate, and added a default fake-admitted checkpoint/config test in `tests/test_sam3_video_upstream_parity.py` that asserts the precise local blocker for missing MLX checkpoint conversion, stable video tap capture, and output mapper/comparator. Corrected this slice's verification command from nonexistent `tests/test_sam3_video.py` to the actual split video tests (`processor`, `session`, `tracking`, and Object Multiplex). `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_video_tracking.py tests/test_sam3_object_multiplex.py tests/test_sam3_upstream_parity.py tests/test_sam3_convert.py` passed outside the sandbox with Metal access: 40 passed, 1 skipped; `rg -n "comparison is not implemented|not implemented in this workspace|pytest\\.fail" tools/sam3_video_upstream.py tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py` found no matches.
**Risks / next:** no generic checkpoint-present blocker remains; docs/status still need to publish the final precise Phase 2 truth and full regression.

### Slice 5: Docs, Roadmap, And Full Regression

**Objective:** Publish the exact Phase 2 result and verify that the repo remains truthful and clean.

**Acceptance criteria:**
- `docs/sam3-video.md`, README/architecture/steering docs if applicable, and the new status artifact describe the exact result: upstream passed or precise blocker.
- Docs include official source, env knobs, out-of-git cache/download boundary, auth/terms warning, and how to run the required gate.
- The prior local-contract status is described as historical local coverage, not as checkpoint admission.
- Roadmap Phase 2 remains active until verify; Phase 3 remains pending.
- `sam3_video` remains absent from the release parity matrix.
- Targeted tests, status JSON validation, runtime dependency guards, full regression, and `git diff --check` pass.

**Verification:** Run `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_video_tracking.py tests/test_sam3_object_multiplex.py tests/test_sam3_upstream_parity.py tests/test_sam3_convert.py tests/test_runtime_dependency_guards.py`, then `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest`, then `python -m json.tool .agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json >/tmp/mlx-cv-sam3-video-checkpoint-status.json`, then `git diff --check`.

**Depends on:** Slices 1 through 4

**Touches:** `docs/sam3-video.md`, `README.md` if status table text needs updating, `.agent/steering/ROADMAP.md`, `.agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json`, `PLAN.md`.

**Produces:** Truthful Phase 2 closeout evidence and final regression result.

**Status:** complete
**Evidence:** updated `docs/sam3-video.md`, `README.md`, `.agent/steering/PROJECT.md`, `.agent/steering/REQUIREMENTS.md`, and `.agent/steering/ROADMAP.md` so the new Phase 2 checkpoint-admission status is the live SAM3 video checkpoint source and the prior `sam3-video-object-multiplex` status is historical local-contract evidence. Corrected this slice's verification command to the actual split SAM3 video tests. Targeted verification `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_video_tracking.py tests/test_sam3_object_multiplex.py tests/test_sam3_upstream_parity.py tests/test_sam3_convert.py tests/test_runtime_dependency_guards.py` passed outside the sandbox with Metal access: 45 passed, 1 skipped. Full regression `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest` passed outside the sandbox with Metal access: 450 passed, 10 skipped. `python -m json.tool .agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json >/tmp/mlx-cv-sam3-video-checkpoint-status.json` passed; direct release-matrix check confirmed models remain bounded to `da3_multiview`, `locateanything`, `rfdetr`, and `sam3_image`; `rg -n "comparison is not implemented|not implemented in this workspace|pytest\\.fail" tools/sam3_video_upstream.py tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py` found no matches; `git diff --check` passed.
**Risks / next:** no implementation gaps remain for the planned Phase 2 scope; final verify can mark the phase closed if the aggregate checks remain stable.

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Required status/gate contract | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_runtime_dependency_guards.py` |
| Upstream reference boundary | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_runtime_dependency_guards.py` |
| Local SAM 3.1 video/image regression | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_video_tracking.py tests/test_sam3_object_multiplex.py tests/test_sam3_upstream_parity.py tests/test_sam3_convert.py` |
| Full test suite | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest` |
| Status JSON | `python -m json.tool .agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json >/tmp/mlx-cv-sam3-video-checkpoint-status.json` |
| Diff hygiene | `git diff --check` |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: Reuses the just-verified Phase 1 closeout template — required-mode gate, `tools/sam3_video_upstream.py` admission/comparison helpers, and a forced-branch default-mode test — against files that already exist, with checkpoint source metadata (`facebook/sam3.1`, `sam3.1_multiplex.pt`, `config.json`) that matches upstream's own builder constants.
- Concern: The headline real path cannot execute in this workspace because `facebook/sam3.1` is a gated Hugging Face repo requiring accepted terms/auth and no MLX converter for the `.pt` multiplex checkpoint exists yet, so the Slice 3 reference-execution and Slice 4 numeric-comparison branches will ship validated only at the blocker level.
- Action: Keep Slice 4's mitigation concrete — turn the existing generic `sam3_video_upstream.py:84` blocker into component-specific reasons (missing converter, tap, or comparator) and add the planned default-mode test that forces a fake-admitted checkpoint to that precise blocker, so the untestable real path can never regress to generic wording.
- Verified: PLAN/DESIGN/SPEC read; confirmed `tools/sam3_video_upstream.py` exists with the current generic comparison blocker (line 84) writing the prior change's status path; SAM3 video tests present (`test_sam3_video_checkpoint_gate.py`, `test_sam3_video_upstream_parity.py`, session/tracking/object-multiplex); upstream surfaces `build_sam3_predictor`/`build_sam3_multiplex_video_predictor` and the `version="sam3.1"` dispatch confirmed in `references/sam3/sam3/model_builder.py`; source constants confirmed against `model_builder.py:664-666` with the gated-repo auth note in `README.md`; additive blast radius and the package-runtime guard boundary reviewed.

## Verification

### Summary

**Overall:** PASS
**Passed:** 5 of 5 slices
**Remaining gaps:** none
**Change status:** complete
**New objective:** use `auto-office-hours` to shape the next objective when you are ready.

### Slice Rollup

- Slice 1, Phase Boundary, Status Contract, And Official Source: **PASS**. Evidence: fresh verify-stage `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_runtime_dependency_guards.py` passed outside the sandbox with Metal access: 22 passed.
- Slice 2, Checkpoint Download, Cache, And Admission: **PASS**. Evidence: the same fresh required gate/status command covered fake checkpoint/config admission states, official model ID validation, cache admission, uncached gated-source blocker, and status schema assertions: 22 passed.
- Slice 3, Upstream Reference Capture Or Reference Blocker: **PASS**. Evidence: fresh verify-stage `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_runtime_dependency_guards.py` passed outside the sandbox with Metal access: 22 passed.
- Slice 4, Local Comparison Boundary And Component Blockers: **PASS**. Evidence: fresh verify-stage `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_video_processor.py tests/test_sam3_video_session.py tests/test_sam3_video_tracking.py tests/test_sam3_object_multiplex.py tests/test_sam3_upstream_parity.py tests/test_sam3_convert.py` passed outside the sandbox with Metal access: 40 passed, 1 skipped; direct scan found no `comparison is not implemented`, `not implemented in this workspace`, or `pytest.fail` fail-stub in the SAM3 video gate/test files.
- Slice 5, Docs, Roadmap, And Full Regression: **PASS**. Evidence: fresh verify-stage full regression `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest` passed outside the sandbox with Metal access: 450 passed, 10 skipped; `python -m json.tool .agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json >/tmp/mlx-cv-sam3-video-checkpoint-status.json` passed; direct release-matrix check confirmed models remain bounded to `da3_multiview`, `locateanything`, `rfdetr`, and `sam3_image`; `git diff --check` passed.

### Skipped Checks

- Full regression has 10 expected env-gated skips for external checkpoint/reference gates. This change's required SAM3 video gate was exercised in required mode above, so default-mode skips are not treated as checkpoint parity passes.
