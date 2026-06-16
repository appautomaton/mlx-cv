# PLAN: Release Parity Hardening

Change: `2026-06-16-release-parity-hardening` - Stage: plan - Spec: `SPEC.md`

## Goal

Execute `SPEC.md`: harden LocateAnything, RF-DETR, and SAM 3.1 image-mode parity claims without adding reference runtime dependencies or expanding beyond roadmap Phase 1.

## Architecture Approach

Keep existing local tiny/integration fixtures as fast CI coverage and add separate env-gated upstream-reference/full-checkpoint parity paths. Reference code, Torch, Transformers, and external checkpoints may run only in opt-in tools/tests outside package runtime. Release status depends on each env-gated parity command passing or recording a precise blocker; skipped tests and local-only fixtures do not create stronger claims.

`parity-status.json` is the phase-local status source for docs and execution evidence. Allowed status values are `LOCAL_FIXTURE_ONLY`, `UPSTREAM_PASSED`, and `BLOCKED:<reason>`. Default parity tolerance is `atol=1e-4, rtol=1e-4`; any model-specific tolerance override must be recorded with a reason, and loosening beyond `1e-3` requires returning to planning.

## Execution Routing And Topology

- Default execution: direct, serial, continue through all slices when verification passes.
- Subagent recommended: Slices 2-4, because each model family crosses reference execution, checkpoint conversion, and parity drift diagnosis.
- Parallel-safe groups: none by default. The three hardening paths touch shared fixture tooling, parity harness conventions, docs/status wording, and dependency guards.
- Checkpoints: none. If required checkpoints or reference environments cannot be obtained during execution, record blockers and keep status wording conservative.

## Ordered Slice Sequence

### Slice 1: Phase Boundary And Shared Parity Metadata

**Objective:** Make the Phase 1 hardening boundary testable across LocateAnything, RF-DETR, and SAM 3.1 image-mode.

**Acceptance criteria:**
- A shared metadata or documentation surface lists the three Phase 1 hardening targets, external checkpoint/reference prerequisites, license caveats, local fixture status, and default tolerance policy.
- `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` is created with one entry per model and status values limited to `LOCAL_FIXTURE_ONLY`, `UPSTREAM_PASSED`, or `BLOCKED:<reason>`.
- Out-of-scope items remain explicit: SAM video/tracker/Object Multiplex, DA3 multi-view, RF-DETR segmentation/Plus variants, and new model families.
- Existing unsupported-variant guards for RF-DETR and SAM remain covered.
- Runtime dependency guards still reject Torch/reference dependencies from package runtime and assert that `src/mlx_cv/` neither imports from `references` nor injects `references` onto `sys.path`.

**Touches:** `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, `.agent/steering/`, `docs/`, `src/mlx_cv/parity/fixtures.py`, model metadata/config tests, `tests/test_runtime_dependency_guards.py`.

**Produces:** A single Phase 1 hardening target matrix.

**Verification:** `uv run pytest tests/test_runtime_dependency_guards.py tests/test_rfdetr_convert.py tests/test_sam3_convert.py tests/test_la_convert.py`

**Status:** complete
**Evidence:** added `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` with the Phase 1 model matrix, default tolerance, status enum, checkpoint env vars, reference paths, license notes, and scoped exclusions; extended `tests/test_runtime_dependency_guards.py` to reject `references` imports/sys.path injection under `src/mlx_cv/` and validate the status matrix; `uv run pytest tests/test_runtime_dependency_guards.py tests/test_rfdetr_convert.py tests/test_sam3_convert.py tests/test_la_convert.py` passed with 26 tests.
**Risks / next:** upstream parity slices must update `parity-status.json` instead of treating skipped env-gated tests as success.

### Slice 2: LocateAnything Full-Checkpoint Parity Gate

**Objective:** Add or tighten the env-gated LocateAnything reference parity path for fixed grounding prompts.

**Acceptance criteria:**
- Tooling/test command accepts explicit LocateAnything checkpoint/reference paths and runs out of package runtime.
- Local MLX outputs are compared to upstream/reference outputs for decoded boxes/points and stable intermediate taps where available.
- Tokenizer, coordinate-token decoding, image-token scatter/projector, and PBD generation drift are separately diagnosable where stable taps exist.
- If the upstream checkpoint/reference runtime is unavailable, `parity-status.json` records `BLOCKED:<reason>` and no stronger claim is made.
- If the env var is unset, the pytest gate skips with a precise reason and does not change status to `UPSTREAM_PASSED`.
- Existing LocateAnything local integration/parity tests continue to pass.

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tools/mint_locateanything_fixture.py`, `tests/test_la_parity.py`, `tests/test_la_integration_fixture.py`, `tests/test_la_predict.py`, `src/mlx_cv/models/locateanything/`.

**Produces:** LocateAnything full-checkpoint parity evidence or blocker.

**Verification:** `MLX_CV_LOCATEANYTHING_CHECKPOINT=/path/to/LocateAnything-3B PYTHONPATH=references/LocateAnything-3B uv run pytest tests/test_la_upstream_parity.py tests/test_la_parity.py tests/test_la_integration_fixture.py`

**Status:** complete
**Evidence:** added `tests/test_la_upstream_parity.py` as an env-gated LocateAnything upstream parity gate and updated `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` to `BLOCKED:MLX_CV_LOCATEANYTHING_CHECKPOINT is unset or points to incomplete local safetensor stubs`; `MLX_CV_LOCATEANYTHING_CHECKPOINT=references/LocateAnything-3B PYTHONPATH=references/LocateAnything-3B uv run pytest tests/test_la_upstream_parity.py tests/test_la_parity.py tests/test_la_integration_fixture.py` passed with 3 tests and 1 expected blocker skip.
**Risks / next:** full LocateAnything upstream comparison remains blocked until a usable full checkpoint and reference runtime are available; the skip is not counted as upstream parity success.

### Slice 3: RF-DETR Nano Upstream Parity Gate

**Objective:** Add RF-DETR Nano COCO upstream checkpoint parity while preserving local tiny detector fixture coverage.

**Acceptance criteria:**
- RF-DETR Nano metadata is documented: alias/class, upstream filename, URL/source identifier, expected MD5, Apache 2.0 status, and model input size.
- Tooling captures upstream RF-DETR detector outputs/taps from `references/rf-detr/` instead of the local MLX model.
- The same checkpoint can be converted/loaded into the local RF-DETR model, with real-checkpoint key/config fixes covered by focused tests.
- Env-gated parity compares final boxes, scores, class IDs, raw logits/boxes, and stable taps; injected drift is localized by `bisect`.
- If the checkpoint/reference runtime is unavailable, `parity-status.json` records `BLOCKED:<reason>`; skipped pytest gates do not count as upstream parity.
- Existing local RF-DETR tiny parity and predict tests continue to pass.

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tools/mint_rfdetr_fixture.py`, `src/mlx_cv/models/rfdetr/`, `tests/test_rfdetr_upstream_parity.py`, `tests/test_rfdetr_parity.py`, `tests/test_rfdetr_predict.py`.

**Produces:** RF-DETR Nano upstream parity evidence or blocker.

**Verification:** `MLX_CV_RFDETR_NANO_CHECKPOINT=/path/to/rf-detr-nano.pth PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py`

**Status:** complete
**Evidence:** added `tests/test_rfdetr_upstream_parity.py` as an env-gated RF-DETR Nano upstream checkpoint gate and updated `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` to `BLOCKED:MLX_CV_RFDETR_NANO_CHECKPOINT is unset or checkpoint file is unavailable`; `MLX_CV_RFDETR_NANO_CHECKPOINT=/path/to/rf-detr-nano.pth PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` passed with 6 tests and 1 expected blocker skip.
**Risks / next:** full RF-DETR upstream comparison remains blocked until the Nano checkpoint is available and matches the expected MD5.

### Slice 4: SAM 3.1 Image Upstream Parity Gate

**Objective:** Add SAM 3.1 image-mode upstream reference parity for text and PCS box/exemplar prompts without entering video/tracker scope.

**Acceptance criteria:**
- The SAM 3.1 image-mode reference target, checkpoint prerequisites, and license caveat are documented.
- Tooling/test command captures upstream image-mode outputs for text prompts and PCS box/exemplar prompts where stable reference APIs/taps exist.
- Local outputs are compared for masks, paired boxes/scores, token/text path evidence, prompt embeddings, and stable decoder/mask taps where available.
- Video/tracker/Object Multiplex imports and state remain out of runtime scope.
- If stable upstream image-mode taps, checkpoint, or reference runtime are unavailable, `parity-status.json` records `BLOCKED:<reason>` and local tiny image fixtures stay labeled as local coverage.
- If the existing SAM3 fixture-mint comment says no reference checkout is available, reconcile that wording with the current `references/sam3/` source checkout so the blocker, if any, names the real missing prerequisite.
- Existing SAM 3.1 local image parity and predict tests continue to pass.

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tools/mint_sam3_fixture.py`, `src/mlx_cv/models/sam3/`, `tests/test_sam3_upstream_parity.py`, `tests/test_sam3_parity.py`, `tests/test_sam3_predict.py`.

**Produces:** SAM 3.1 image-mode upstream parity evidence or blocker.

**Verification:** `MLX_CV_SAM3_IMAGE_CHECKPOINT=/path/to/sam3-image-checkpoint PYTHONPATH=references/sam3 uv run pytest tests/test_sam3_upstream_parity.py tests/test_sam3_parity.py tests/test_sam3_predict.py`

### Slice 5: Status Truthfulness And Full Regression

**Objective:** Update release/status wording only after model-specific parity gates pass or record blockers, then run full regression.

**Acceptance criteria:**
- README, steering docs, and architecture docs accurately report each model's hardening status: upstream-reference/full-checkpoint parity passed, blocked with reason, or still local-fixture-only.
- Status wording is derived from `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, not from pytest skip counts or local fixture success alone.
- Docs distinguish local tiny/integration fixtures from upstream checkpoint parity for LocateAnything, RF-DETR, and SAM 3.1 image-mode.
- No out-of-scope roadmap items are marked complete.
- Runtime dependency guard and full suite pass.
- `PLAN.md` records evidence for each env-gated parity command, blocker, and full regression.

**Depends on:** Slices 2, 3, 4

**Touches:** `README.md`, `.agent/steering/`, `docs/`, `PLAN.md`.

**Produces:** Truthful Phase 1 hardening status.

**Verification:** `uv run pytest`

## Requirement Traceability

| SPEC acceptance | Satisfying slices |
| --- | --- |
| AC1 LocateAnything scope | Slices 1, 2 |
| AC2 LocateAnything parity | Slices 1, 2 |
| AC3 RF-DETR scope | Slices 1, 3 |
| AC4 RF-DETR parity | Slices 1, 3 |
| AC5 SAM 3.1 image scope | Slices 1, 4 |
| AC6 SAM 3.1 image parity | Slices 1, 4 |
| AC7 Fast CI remains useful | Slices 1, 2, 3, 4, 5 |
| AC8 Runtime hygiene | Slices 1, 5 |
| AC9 Status truthfulness | Slices 1, 5 |
| AC10 Regression | Slice 5 |

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Shared guards | `uv run pytest tests/test_runtime_dependency_guards.py tests/test_rfdetr_convert.py tests/test_sam3_convert.py tests/test_la_convert.py` |
| LocateAnything local and upstream parity | `MLX_CV_LOCATEANYTHING_CHECKPOINT=/path/to/LocateAnything-3B PYTHONPATH=references/LocateAnything-3B uv run pytest tests/test_la_upstream_parity.py tests/test_la_parity.py tests/test_la_integration_fixture.py` |
| RF-DETR local and upstream parity | `MLX_CV_RFDETR_NANO_CHECKPOINT=/path/to/rf-detr-nano.pth PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` |
| SAM 3.1 image local and upstream parity | `MLX_CV_SAM3_IMAGE_CHECKPOINT=/path/to/sam3-image-checkpoint PYTHONPATH=references/sam3 uv run pytest tests/test_sam3_upstream_parity.py tests/test_sam3_parity.py tests/test_sam3_predict.py` |
| Full regression | `uv run pytest` |

## Risks

- External checkpoints and Torch/reference environments may require approval or may be unavailable. A blocked parity gate must remain a blocker, not a skipped success.
- Reference APIs may not expose stable intermediate taps for every model. Final outputs and the most stable available taps are required; unstable tap gaps must be documented.
- Converter fixes discovered by real checkpoints can tempt broader model support. Keep fixes limited to the three Phase 1 hardening targets unless a shared safety fix is required.
- Status updates are part of the plan, not cleanup. They prevent release claims from outrunning evidence.
