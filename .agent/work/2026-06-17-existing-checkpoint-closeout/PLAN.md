# PLAN: Existing Checkpoint Closeout

Change: `2026-06-17-existing-checkpoint-closeout` - Stage: plan - Spec: `SPEC.md` - Design: `DESIGN.md`

## Goal

Execute roadmap Phase 1: close out LocateAnything-3B and SAM 3.1 image-mode checkpoint blockers by producing real upstream/reference parity passes or precise blocker records.

## Architecture Approach

See `DESIGN.md`. Keep release parity status in `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, update only `locateanything` and `sam3_image`, keep checkpoints outside git, and keep reference/Torch/Hugging Face execution out of package runtime. Required gates may pass with a blocker only after asserting the exact `BLOCKED:<reason>` status.

## Requirement Traceability

| SPEC criteria | Satisfying slices |
|---|---|
| AC1, AC2, AC9, AC10 | Slice 1 and Slice 5 |
| AC3, AC4, AC5 | Slice 2 and Slice 3 |
| AC6, AC7, AC8 | Slice 4 |

## Execution Routing And Topology

- Default execution: direct, serial, continue through all slices when verification passes.
- Subagent recommended: Slices 2, 3, and 4 because they cross upstream references, checkpoint shape, and parity blockers.
- Checkpoints: none. If checkpoints or network/model access are unavailable, record precise blockers and continue to status/docs verification.
- Parallel-safe groups: none. Slices share `parity-status.json`, upstream gate tests, and status docs.
- External access: checkpoint download attempts require execution-time approval and must write only to out-of-git cache paths.

## Ordered Slice Sequence

### Slice 1: Phase Boundary, Status Contract, And Required Gates

**Objective:** Make the Phase 1 boundary and pass-or-blocker gate semantics executable before model-specific work starts.

**Acceptance criteria:**
- Roadmap Phase 1 remains `active` with change `2026-06-17-existing-checkpoint-closeout`; Phase 2 and Phase 3 remain pending and unmodified.
- `parity-status.json` remains bounded to `da3_multiview`, `locateanything`, `rfdetr`, and `sam3_image`.
- LocateAnything and SAM 3.1 image entries include explicit source/provenance fields or blocker fields needed by later slices.
- `MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1` and `MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1` required modes are defined so gates cannot silently skip in required verification.
- Existing runtime dependency guards continue to reject reference/runtime dependency leakage into `src/mlx_cv/`.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1 MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1 uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_sam3_upstream_parity.py tests/test_runtime_dependency_guards.py`

**Touches:** `.agent/steering/ROADMAP.md`, `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, `tests/test_la_upstream_parity.py`, `tests/test_sam3_upstream_parity.py`, `tests/test_runtime_dependency_guards.py`.

**Produces:** A required-gate contract that distinguishes blocker pass from silent skip.

**Status:** complete
**Evidence:** added required-mode handling for `MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1` and `MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1`, and extended the `locateanything` / `sam3_image` status entries with required gate, source, cache, and provenance fields; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1 MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1 uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_sam3_upstream_parity.py tests/test_runtime_dependency_guards.py` passed with 7 tests.
**Risks / next:** real checkpoint-present comparison branches still need Slice 3/4 component-specific blockers to address the engineering review risk.

### Slice 2: LocateAnything Checkpoint Admission And Provenance

**Objective:** Turn LocateAnything checkpoint availability into a precise admissibility decision.

**Acceptance criteria:**
- The gate recognizes unset env, nonexistent path, LFS stub shards, missing index, incomplete shard set, unsupported format, and usable checkpoint directory.
- The current `references/LocateAnything-3B` directory is explicitly classified as blocked because both safetensors shards are 135-byte stubs.
- Status records the official source/model ID, NVIDIA non-commercial license note, expected out-of-git cache behavior, and checksum/provenance status when known.
- If a usable cache is present or downloaded with approval, the gate records its path/provenance without committing weights.
- Local conversion/load smoke coverage remains intact for supported `.npz`, `.safetensors`, and sharded directory inputs.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1 MLX_CV_LOCATEANYTHING_CHECKPOINT=references/LocateAnything-3B PYTHONPATH=references/LocateAnything-3B uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_la_convert.py tests/test_la_integration_fixture.py tests/test_runtime_dependency_guards.py`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tests/test_la_upstream_parity.py`, `src/mlx_cv/models/locateanything/convert.py` if admission gaps are found, optional `tools/locateanything_upstream.py`, `parity-status.json`.

**Produces:** LocateAnything checkpoint admission evidence or a precise checkpoint blocker.

**Status:** complete
**Evidence:** added `tools/locateanything_upstream.py` with LocateAnything checkpoint admission classification for unset env, missing path, missing index, incomplete shards, LFS stubs, unsupported files, and usable full directories; updated `tests/test_la_upstream_parity.py` and `parity-status.json` with model ID/source/cache/provenance fields. Sandbox verification hit the known Metal access limitation while collecting MLX tests, then the same command passed outside the sandbox: `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1 MLX_CV_LOCATEANYTHING_CHECKPOINT=references/LocateAnything-3B PYTHONPATH=references/LocateAnything-3B uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_la_convert.py tests/test_la_integration_fixture.py tests/test_runtime_dependency_guards.py` passed with 18 tests.
**Risks / next:** Slice 3 must route an admitted checkpoint with absent comparison/taps to a precise component blocker instead of a bare fail-stub.

### Slice 3: LocateAnything Reference Comparison Or Component Blocker

**Objective:** Replace the LocateAnything "checkpoint prerequisites are present but comparison is not implemented" fail-stub with a real comparison attempt or component-specific blocker.

**Acceptance criteria:**
- With a usable checkpoint and reference deps, the gate attempts upstream/reference-vs-MLX comparison on fixed grounding inputs.
- The comparison covers decoded boxes/points and stable taps where upstream exposes them; missing stable taps are named as blockers rather than ignored.
- Dependency absence, upstream runtime failure, local converter/load failure, missing comparator, numeric mismatch, and pass are separated in status or test output.
- Existing local LocateAnything tests continue to prove tokenizer-backed local integration without being labeled upstream parity.
- If no usable checkpoint is available, the required gate still passes by asserting the precise blocker from Slice 2.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1 MLX_CV_LOCATEANYTHING_CHECKPOINT=references/LocateAnything-3B PYTHONPATH=references/LocateAnything-3B uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_la_parity.py tests/test_la_predict.py tests/test_la_integration_fixture.py`

**Execution:** subagent recommended

**Depends on:** Slice 2

**Touches:** `tests/test_la_upstream_parity.py`, `tools/`, `src/mlx_cv/models/locateanything/`, `parity-status.json`.

**Produces:** LocateAnything upstream parity pass or precise comparison/load blocker.

**Status:** complete
**Evidence:** replaced the LocateAnything checkpoint-present fail-stub with `evaluate_locateanything_comparison_gate`, which routes admitted checkpoints to dependency, reference-path, or missing-comparison-component blockers; added a default unit test that creates a fake admitted checkpoint and asserts the component-specific blocker for decoded boxes/points and stable taps. `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1 MLX_CV_LOCATEANYTHING_CHECKPOINT=references/LocateAnything-3B PYTHONPATH=references/LocateAnything-3B uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_la_parity.py tests/test_la_predict.py tests/test_la_integration_fixture.py` passed outside the sandbox with Metal access: 10 tests.
**Risks / next:** no real LocateAnything checkpoint is available here; upstream numeric parity remains blocked until a complete checkpoint and comparator are available.

### Slice 4: SAM 3.1 Image Checkpoint And Tap Closeout

**Objective:** Replace the SAM 3.1 image fail-stub with a real image checkpoint/tap comparison attempt or precise blocker.

**Acceptance criteria:**
- The gate recognizes unset env, nonexistent path, unusable checkpoint, dependency absence, video/tracker checkpoint mismatch, missing stable image tap capture, local converter/load failure, numeric mismatch, and pass.
- The gate uses or audits upstream image-mode entry points under `references/sam3/` without importing them from package runtime.
- With a usable checkpoint and reference deps, the gate attempts text and PCS-style image prompt comparison for masks, paired detections, token/text evidence, and stable taps where available.
- Existing image-mode converter rejection of video/tracker state remains covered.
- If no usable image checkpoint or stable tap path is available, the required gate passes only by asserting a precise `BLOCKED:<reason>` status.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_upstream_parity.py tests/test_sam3_convert.py tests/test_sam3_parity.py tests/test_sam3_predict.py tests/test_sam3_processor.py`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tests/test_sam3_upstream_parity.py`, `src/mlx_cv/models/sam3/convert.py` if admission gaps are found, optional `tools/sam3_image_upstream.py`, `parity-status.json`.

**Produces:** SAM 3.1 image upstream parity pass or precise checkpoint/tap/comparison blocker.

**Status:** complete
**Evidence:** added `tools/sam3_image_upstream.py` with SAM3 image checkpoint admission and comparison blockers for unset env, missing path, directory path, tiny/unusable file, local-converter-incompatible format, video/tracker checkpoint mismatch, missing torch, and missing stable tap/comparison component; updated `tests/test_sam3_upstream_parity.py` with default-mode tests for admitted-checkpoint missing tap comparison and video/tracker checkpoint blocking; kept `tests/test_sam3_convert.py` image-loader rejection coverage intact. `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_upstream_parity.py tests/test_sam3_convert.py tests/test_sam3_parity.py tests/test_sam3_predict.py tests/test_sam3_processor.py` passed outside the sandbox with Metal access: 22 tests.
**Risks / next:** no real SAM3 image checkpoint or stable image tap path is configured here; upstream numeric parity remains a precise external/component blocker.

### Slice 5: Status Docs, Roadmap, And Regression

**Objective:** Publish only the truth established by the two closeout gates and run the final regression.

**Acceptance criteria:**
- README, architecture docs, and steering docs report LocateAnything and SAM 3.1 image as `UPSTREAM_PASSED` only if their required gates actually pass against real checkpoints.
- Blocked entries name the exact missing external artifact, runtime, tap path, conversion/load component, or comparison component.
- Roadmap Phase 1 remains active until verify; Phase 2/3 stay pending.
- `sam3_video` remains outside `parity-status.json` and outside this phase.
- `git diff --check`, targeted parity/local tests, and full regression pass.

**Verification:** Run `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_sam3_upstream_parity.py tests/test_la_parity.py tests/test_sam3_parity.py tests/test_runtime_dependency_guards.py`, then `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest`, then `git diff --check`.

**Depends on:** Slices 3 and 4

**Touches:** `README.md`, `docs/ARCHITECTURE.md`, `.agent/steering/PROJECT.md`, `.agent/steering/REQUIREMENTS.md`, `.agent/steering/ROADMAP.md`, `PLAN.md`.

**Produces:** Truthful Phase 1 status and final regression evidence.

**Status:** complete
**Evidence:** updated `README.md`, `docs/ARCHITECTURE.md`, `.agent/steering/PROJECT.md`, `.agent/steering/REQUIREMENTS.md`, and `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` so LocateAnything and SAM 3.1 image remain precise blockers rather than upstream-passed claims. Targeted verification `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_sam3_upstream_parity.py tests/test_la_parity.py tests/test_sam3_parity.py tests/test_runtime_dependency_guards.py` passed outside the sandbox with Metal access: 15 passed, 2 skipped. Full regression `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest` passed outside the sandbox with Metal access: 439 passed, 10 skipped. `python -m json.tool .agent/work/2026-06-16-release-parity-hardening/parity-status.json >/tmp/mlx-cv-parity-status.json` passed; `git diff --check` passed; direct scans found no old `comparison is not implemented` fail-stub, no `sam3_video` row in the release parity matrix, and no roadmap status drift.
**Risks / next:** none for planned scope; both remaining models are precise blockers, not upstream-passed real-checkpoint parity.

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Required blocker gates | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1 MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1 uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_sam3_upstream_parity.py tests/test_runtime_dependency_guards.py` |
| LocateAnything closeout | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1 MLX_CV_LOCATEANYTHING_CHECKPOINT=references/LocateAnything-3B PYTHONPATH=references/LocateAnything-3B uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_la_convert.py tests/test_la_parity.py tests/test_la_predict.py tests/test_la_integration_fixture.py` |
| SAM 3.1 image closeout | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_upstream_parity.py tests/test_sam3_convert.py tests/test_sam3_parity.py tests/test_sam3_predict.py tests/test_sam3_processor.py` |
| Full test suite | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest` |
| Diff hygiene | `git diff --check` |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan mirrors an already-proven shape — the `MLX_CV_REQUIRE_*_GATE` required-mode pattern plus `tools/*_upstream.py` that landed DA3 and RF-DETR at `UPSTREAM_PASSED` — so Slice 1's gate contract and Slices 3/4's comparison harness have a working template rather than a new design.
- Concern: The real upstream-vs-MLX comparison branches in Slices 3 and 4 cannot be exercised end-to-end in this workspace (LocateAnything shards are confirmed 135-byte LFS stubs and the SAM 3.1 image checkpoint plus stable tap path are unconfigured), so without an approved download those branches ship validated only at the blocker level and could silently regress to the old `pytest.fail("comparison is not implemented")` stub.
- Action: In Slices 3 and 4, route the checkpoint-present-but-comparison-absent case to a component-specific `BLOCKED:<reason>` (naming the missing converter, tap capture, or comparator — mirroring `tools/da3_checkpoint.py`), and add a default-mode unit test that forces that branch to a precise blocker so the untestable real path can never fall back to a bare `pytest.fail`.
- Verified: Current fail-stub gates read (`tests/test_la_upstream_parity.py`, `tests/test_sam3_upstream_parity.py`) — both `pytest.skip` on unset checkpoint and `pytest.fail("...not implemented")` when prereqs present, and neither has a required-mode path yet; `MLX_CV_REQUIRE_LOCATEANYTHING_GATE`/`MLX_CV_REQUIRE_SAM3_IMAGE_GATE` confirmed absent today while the analogous DA3/RF-DETR/sam3_video required-gate envs exist in `tools/`; `parity-status.json` confirmed bounded to `{da3_multiview, locateanything, rfdetr, sam3_image}` with `locateanything`/`sam3_image` already `BLOCKED:` and carrying the fields the runtime guard asserts; LocateAnything `*.safetensors` confirmed 135-byte stubs with `model.safetensors.index.json` present; ROADMAP Phase 1 bound to this change with Phases 2/3 pending; runtime-import boundary machine-enforced and status-matrix kept additive (no new rows). Data flow traced env→admission→(blocker|comparison)→status→docs; rollback safety reviewed (additive, no committed weights).

## Verification

### Summary

**Overall:** PASS
**Passed:** 5 of 5 slices
**Remaining gaps:** none for planned scope
**Change status:** complete
**New objective:** use `auto-office-hours` to shape the next objective when you are ready.

### Slice Rollup

- Slice 1, Phase Boundary, Status Contract, And Required Gates: **PASS**. Evidence: fresh verify-stage `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1 MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1 uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_sam3_upstream_parity.py tests/test_runtime_dependency_guards.py` passed outside the sandbox with Metal access: 11 passed.
- Slice 2, LocateAnything Checkpoint Admission And Provenance: **PASS**. Evidence: fresh verify-stage `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_LOCATEANYTHING_GATE=1 MLX_CV_LOCATEANYTHING_CHECKPOINT=references/LocateAnything-3B PYTHONPATH=references/LocateAnything-3B uv run --extra test pytest tests/test_la_upstream_parity.py tests/test_la_convert.py tests/test_la_parity.py tests/test_la_predict.py tests/test_la_integration_fixture.py` passed outside the sandbox with Metal access: 19 passed.
- Slice 3, LocateAnything Reference Comparison Or Component Blocker: **PASS**. Evidence: the same fresh LocateAnything closeout command covered the required local fixture, predict, admission, and component-blocker paths: 19 passed; direct scan found no old `comparison is not implemented` / `pytest.fail` fail-stub in `tests/test_la_upstream_parity.py` or `tools/locateanything_upstream.py`.
- Slice 4, SAM 3.1 Image Checkpoint And Tap Closeout: **PASS**. Evidence: fresh verify-stage `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1 PYTHONPATH=references/sam3 uv run --extra test pytest tests/test_sam3_upstream_parity.py tests/test_sam3_convert.py tests/test_sam3_parity.py tests/test_sam3_predict.py tests/test_sam3_processor.py` passed outside the sandbox with Metal access: 22 passed; direct scan found no old `comparison is not implemented` / `pytest.fail` fail-stub in `tests/test_sam3_upstream_parity.py` or `tools/sam3_image_upstream.py`.
- Slice 5, Status Docs, Roadmap, And Regression: **PASS**. Evidence: fresh verify-stage full regression `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest` passed outside the sandbox with Metal access: 439 passed, 10 skipped; `python -m json.tool .agent/work/2026-06-16-release-parity-hardening/parity-status.json >/tmp/mlx-cv-parity-status.json` passed; `git diff --check` passed; direct matrix check confirmed models remain bounded to `da3_multiview`, `locateanything`, `rfdetr`, and `sam3_image`, with LocateAnything and SAM 3.1 image both precise `BLOCKED:` records.

### Skipped Checks

- Full regression has 10 expected env-gated skips for external checkpoint/reference gates. The closeout-specific required commands above exercised LocateAnything and SAM 3.1 image blockers in required mode, so these default-mode skips are not treated as upstream parity passes.
