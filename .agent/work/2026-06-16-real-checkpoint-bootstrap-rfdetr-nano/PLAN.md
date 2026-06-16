# PLAN: Real Checkpoint Bootstrap - RF-DETR Nano

Change: `2026-06-16-real-checkpoint-bootstrap-rfdetr-nano` - Stage: plan - Spec: `SPEC.md`

## Goal

Execute `SPEC.md`: finish roadmap Phase 1 by proving RF-DETR Nano real-checkpoint load, upstream/reference comparison, MLX comparison, and truthful `UPSTREAM_PASSED` status.

## Architecture Approach

Keep all network, Torch, checkpoint extraction, and upstream RF-DETR execution in `tools/` and env-gated tests. `src/mlx_cv/` remains runtime-clean and loads only local `.npz`/safetensors-style converted arrays. Use the existing release parity status matrix as the current model status source so RF-DETR does not have competing blocker/pass records.

## Execution Routing And Topology

- Default execution: direct, serial, continue through all slices when verification passes.
- Subagent recommended: Slices 3 and 4, because they cross upstream RF-DETR, checkpoint format inspection, converter behavior, and numerical parity.
- Parallel-safe groups: none.
- Checkpoints: none. Network/checkpoint fetch may require sandbox escalation during execution, but it is not a product decision. This phase must not close with a missing-checkpoint blocker or skipped upstream gate.

## Ordered Slice Sequence

### Slice 1: Checkpoint Cache And MD5 Gate

**Objective:** Add an explicit RF-DETR Nano checkpoint preparation path that resolves an out-of-git checkpoint and verifies the expected MD5 before any parity run.

**Acceptance criteria:**
- A tool or helper resolves RF-DETR Nano from `MLX_CV_RFDETR_NANO_CHECKPOINT` or an out-of-git cache root such as `$MLX_CV_CACHE`.
- The helper knows the recorded URL, filename `rf-detr-nano.pth`, and expected MD5 `fb6504cce7fbdc783f7a46991f07639f`.
- If the source URL basename differs from the canonical filename, the helper stores or aliases it as `rf-detr-nano.pth` and verifies the final file bytes.
- Checksum mismatch is a hard failure, not a skip.
- Download behavior is opt-in and does not run in normal CI by accident.
- The helper prints the resolved checkpoint path and MD5 after a successful verification.
- Raw checkpoints and converted weights remain outside git.
- Runtime dependency guards still pass.

**Touches:** `tools/`, `tests/test_rfdetr_checkpoint_gate.py`, `.gitignore` only if a new local cache path needs protection.

**Produces:** Reproducible checkpoint path contract for later slices.

**Verification:** `uv run pytest tests/test_runtime_dependency_guards.py tests/test_rfdetr_checkpoint_gate.py`

**Status:** complete
**Evidence:** added `tools/rfdetr_checkpoint.py` and `tests/test_rfdetr_checkpoint_gate.py`; `uv run pytest tests/test_runtime_dependency_guards.py tests/test_rfdetr_checkpoint_gate.py` passed with 12 tests.
**Risks / next:** none.

### Slice 2: Upstream RF-DETR Nano Capture Path

**Objective:** Replace ad hoc reference assumptions with an env-gated upstream RF-DETR Nano runner that captures comparable reference outputs for one fixed input.

**Acceptance criteria:**
- Reference execution is isolated to tools/tests and may use `PYTHONPATH=references/rf-detr/src`.
- The runner uses the verified RF-DETR Nano checkpoint path from Slice 1.
- The fixed input and preprocessing/postprocessing are explicit and deterministic.
- Captured reference data includes final boxes, scores, class IDs, raw logits/boxes when available, and stable taps where reference APIs expose them.
- Any unavailable intermediate tap is recorded as a stable-tap gap rather than invented.
- In normal CI mode without checkpoint configuration, this dedicated test cleanly skips; with `MLX_CV_REQUIRE_RFDETR_GATE=1`, missing checkpoint, bad checksum, missing reference dependency, or skipped capture fails.
- Required-vs-normal behavior uses the shared RF-DETR gate helper so capture, load, and parity tests cannot drift.
- No reference import enters `src/mlx_cv/`.
- The capture verification uses a dedicated `upstream_capture` test target and does not accidentally select the legacy placeholder gate.

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tools/`, `tests/test_rfdetr_upstream_capture.py`, `tests/test_rfdetr_upstream_parity.py`, `src/mlx_cv/parity/fixtures.py` if a fixed real-gate input helper is needed.

**Produces:** Upstream capture function or tool usable by the real parity test.

**Verification:** `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_capture.py`

### Slice 3: Local Real-Checkpoint Conversion And Load

**Objective:** Make the same verified RF-DETR Nano checkpoint load into the local MLX RF-DETR detection path without adding Torch to package runtime.

**Acceptance criteria:**
- Torch `.pth` checkpoint extraction, if needed, happens in tools/tests outside `src/mlx_cv/`.
- The local MLX loader consumes a runtime-clean converted representation or supported array format.
- Real checkpoint key/config mismatches are fixed for RF-DETR Nano detection only.
- Existing segmentation/PML/unsupported-variant rejection tests remain intact.
- The local model runs the fixed input with `capture_taps=True` and returns raw logits/boxes, typed detections, and local taps.
- A dedicated real-checkpoint load test actually reads the verified checkpoint path or converted cache output; tiny fixture tests remain regression coverage only and do not satisfy this slice by themselves.
- In normal CI mode without checkpoint configuration, this dedicated test cleanly skips; with `MLX_CV_REQUIRE_RFDETR_GATE=1`, missing checkpoint, bad checksum, missing conversion output, or skipped load fails.
- Required-vs-normal behavior uses the shared RF-DETR gate helper so capture, load, and parity tests cannot drift.

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tools/`, `src/mlx_cv/models/rfdetr/convert.py`, `src/mlx_cv/models/rfdetr/config.py`, `src/mlx_cv/models/rfdetr/modeling.py`, `tests/test_rfdetr_real_checkpoint_load.py`, `tests/test_rfdetr_convert.py`, `tests/test_rfdetr_parity.py`, `tests/test_rfdetr_predict.py`.

**Produces:** Real RF-DETR Nano checkpoint load path for MLX.

**Verification:** `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> uv run pytest tests/test_rfdetr_real_checkpoint_load.py tests/test_rfdetr_convert.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py`

### Slice 4: Real Upstream-vs-MLX Parity Gate

**Objective:** Replace the current success-path `pytest.fail(...)` placeholder with a real RF-DETR Nano upstream-vs-MLX comparison that must pass to close Phase 1.

**Acceptance criteria:**
- `tests/test_rfdetr_upstream_parity.py` no longer contains a success-path placeholder failure.
- With a checksum-matching checkpoint and `MLX_CV_REQUIRE_RFDETR_GATE=1`, the test runs upstream/reference and local MLX on the same fixed input.
- Final boxes, scores, class IDs, raw logits/boxes, and stable taps compare within default tolerance unless a justified model-specific tolerance is recorded.
- Drift remains localizable through ordered taps or an explicitly documented stable-tap gap.
- Missing checkpoint may skip only outside the phase-closing verification; the phase-closing command must run with a real checkpoint and pass.
- After RF-DETR is `UPSTREAM_PASSED`, the no-checkpoint branch skips without asserting `BLOCKED:` so normal checkpoint-less CI remains green.
- The successful gate output includes the resolved checkpoint path and MD5.

**Execution:** subagent recommended

**Depends on:** Slices 2, 3

**Touches:** `tests/test_rfdetr_upstream_parity.py`, `tools/`, `src/mlx_cv/parity/`, `tests/fixtures/` only for small derived parity cases.

**Produces:** Passing real RF-DETR Nano upstream parity gate.

**Verification:** `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py`

### Slice 5: Status Truthfulness And Full Regression

**Objective:** Promote RF-DETR to real-checkpoint-passed status only after Slice 4 passes, then run the full regression suite.

**Acceptance criteria:**
- `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` records RF-DETR as `UPSTREAM_PASSED` only after the real gate passes.
- LocateAnything and SAM 3.1 image statuses remain blocked and are not silently promoted.
- README, architecture, and steering docs distinguish RF-DETR real-checkpoint parity from remaining local-fixture/blocker states, including current RF-DETR status lines in `README.md`, `docs/ARCHITECTURE.md`, `.agent/steering/PROJECT.md`, and `.agent/steering/REQUIREMENTS.md`.
- With RF-DETR marked `UPSTREAM_PASSED` and no checkpoint env set, normal `uv run pytest` still passes because the env-gated upstream test skips without a stale `BLOCKED:` assertion.
- Runtime dependency guards pass.
- Full suite passes.
- PLAN.md records command evidence for the checkpoint MD5, real upstream gate, status update, and full regression.

**Depends on:** Slice 4

**Touches:** `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, `README.md`, `.agent/steering/`, `docs/`, `PLAN.md`.

**Produces:** Truthful Phase 1 closure with RF-DETR `UPSTREAM_PASSED`.

**Verification:** `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py && uv run pytest`

## Requirement Traceability

| SPEC acceptance | Satisfying slices |
| --- | --- |
| AC1 Checkpoint acquisition | Slice 1 |
| AC2 Runtime hygiene | Slices 1, 2, 5 |
| AC3 Upstream reference run | Slice 2 |
| AC4 Local real-checkpoint run | Slice 3 |
| AC5 Real parity comparison | Slice 4 |
| AC6 Drift diagnosis | Slices 2, 4 |
| AC7 Status truthfulness | Slices 4, 5 |
| AC8 Unsupported variants | Slices 3, 5 |
| AC9 Regression | Slice 5 |

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Checkpoint/cache guard | `uv run pytest tests/test_runtime_dependency_guards.py tests/test_rfdetr_checkpoint_gate.py` |
| Reference capture | `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_capture.py` |
| Local checkpoint load | `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> uv run pytest tests/test_rfdetr_real_checkpoint_load.py tests/test_rfdetr_convert.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` |
| Real RF-DETR parity | `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` |
| Full regression | `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py && uv run pytest` |

## Risks

- Downloading the checkpoint may require explicit network approval during execution.
- The reference package may require optional dependencies not present in the base project environment.
- The real checkpoint may expose converter/config gaps not covered by tiny fixtures.
- Reference APIs may not expose every desired tap; final output parity remains mandatory, and tap gaps must be named.
- The status matrix currently lives under the prior parity-hardening change; this plan intentionally updates it as the current user-facing status source rather than creating a competing status file.
- The RF-DETR upstream test must intentionally diverge from the LocateAnything/SAM blocker-gate shape after RF-DETR passes: checkpoint-less normal CI skips are allowed, but required-gate verification fails when the checkpoint is absent.
- The capture, real-load, and parity tests must share one gate helper so normal no-checkpoint CI skips stay green while required phase-closing runs fail on missing prerequisites.

## Engineering Review

- Reviewer: Claude Code Opus 4.8, max effort, read-only plan mode, session `80eeb71e-2198-446f-9073-5c67bc598c0f`.
- Initial verdict: `NEEDS_CORRECTION`.
- Accepted corrections: require a non-skipped real gate with checkpoint path/MD5 evidence before status promotion; make the RF-DETR no-checkpoint branch compatible with future `UPSTREAM_PASSED` normal CI; replace broad `-k` selectors with dedicated checkpoint, upstream-capture, and real-load test targets; make Slice 3 prove real checkpoint load instead of relying on tiny-fixture tests.
- Final re-review verdict: `APPROVED_WITH_RISKS`.
- Incorporated final review guidance: require shared gate semantics for upstream capture, local real-load, and parity tests; make new env-gated tests cleanly skip in normal no-checkpoint CI but fail in required phase-closing mode; map AC7 to both Slice 4 and Slice 5; preserve canonical `rf-detr-nano.pth` naming even when the download URL basename differs.
