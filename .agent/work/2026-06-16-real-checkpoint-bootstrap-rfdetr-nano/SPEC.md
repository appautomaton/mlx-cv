# SPEC: Real Checkpoint Bootstrap - RF-DETR Nano

Change: `2026-06-16-real-checkpoint-bootstrap-rfdetr-nano` - Stage: frame - Source: `.agent/steering/ROADMAP.md` Phase 1, `.agent/work/2026-06-16-real-checkpoint-bootstrap-rfdetr-nano/INTAKE.md`, current RF-DETR implementation, current blocker matrix.

## Bounded Goal

Finish the active roadmap Phase 1 by proving one real pretrained checkpoint path end to end: RF-DETR Nano must be fetched or supplied outside git, checksum-verified, loaded into both upstream/reference and local MLX paths, and compared successfully.

## Broader Intent

Stop treating local tiny fixtures as evidence that real pretrained models work. This phase establishes the reusable checkpoint discipline that later phases will apply to LocateAnything, SAM 3.1 image/video, DA3 multi-view, and any new model family.

## Target Stakeholder

Library maintainers and users deciding whether `mlx-cv` can honestly claim real checkpoint-backed RF-DETR support rather than only architecture-plumbing coverage.

## Work Scale And Shape

- Scale: phase-sized checkpoint bootstrap for one model family.
- Shape: parity, runtime/tooling, checkpoint acquisition, converter hardening, status truthfulness.
- Selected lenses: product, engineering, runtime, security.

## Required Outcome

- RF-DETR Nano COCO is the only model target for this phase.
- The checkpoint is obtained outside git, either by an explicit approved download from the recorded URL or by a user-supplied local path.
- The checkpoint file is verified against expected MD5 `fb6504cce7fbdc783f7a46991f07639f` before any parity claim is made.
- The existing upstream parity fail-stub in `tests/test_rfdetr_upstream_parity.py` is replaced with a real comparison path.
- The upstream/reference RF-DETR detector and local MLX RF-DETR model run on the same fixed input with aligned preprocessing/postprocessing.
- Local MLX outputs are compared to upstream/reference outputs for final boxes, scores, class IDs, raw logits/boxes, and stable intermediate taps where available.
- Drift diagnosis remains bisectable through the existing parity harness or an equivalent per-tap comparison.
- The phase-closing gate runs in a required mode where a missing checkpoint is a failure, not a skip, and successful output prints the verified checkpoint path and MD5.
- RF-DETR status changes to `UPSTREAM_PASSED` only after the real checkpoint comparison passes; blocker skips and local fixtures still do not count as upstream parity.
- After RF-DETR is `UPSTREAM_PASSED`, normal checkpoint-less CI still passes by skipping the env-gated upstream test without asserting a `BLOCKED:` status.
- Runtime package imports remain clean: Torch, upstream RF-DETR code, checkpoint download code, and `references/` imports stay in tools/tests, not `src/mlx_cv/` runtime.

## Constraints And Risks

- Raw upstream checkpoints and converted model weights must not be committed.
- Normal CI must not depend on network access. The real-checkpoint gate may be env-gated, but phase verification requires running it with a real verified checkpoint.
- Checkpoint cache/download tooling must be explicit about destination, checksum, and license/provenance. A default such as `$MLX_CV_CACHE` or `~/.cache/mlx-cv/` is acceptable.
- RF-DETR segmentation checkpoints and RF-DETR Plus XL/2XL PML variants remain out of scope and must still be rejected by the detection loader.
- Any converter/config changes must be limited to RF-DETR Nano detection unless a shared safety fix is required.
- Reference/Torch execution may require an isolated environment or `PYTHONPATH=references/rf-detr/src`; this must stay out of package runtime.
- If the public download is unavailable, the phase may use a user-supplied checkpoint, but the phase does not close until a checksum-matching checkpoint actually runs and passes.

## Source Evidence

- Current roadmap Phase 1 requires RF-DETR Nano real checkpoint bootstrap and explicitly disallows `BLOCKED:<reason>` as this phase's exit.
- Current status matrix records RF-DETR as `BLOCKED:MLX_CV_RFDETR_NANO_CHECKPOINT is unset or checkpoint file is unavailable` with URL `https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth` and MD5 `fb6504cce7fbdc783f7a46991f07639f`.
- `tests/test_rfdetr_upstream_parity.py` currently skips when the checkpoint is absent or wrong and then `pytest.fail(...)` when prerequisites exist; the real comparison is not implemented yet.
- Local MLX RF-DETR implementation exists under `src/mlx_cv/models/rfdetr/`, with model, processor, converter, and `predict` path.
- Local RF-DETR tiny fixture coverage exists in `tests/test_rfdetr_parity.py`, `tests/test_rfdetr_predict.py`, and `tests/fixtures/rfdetr_tiny_fixture*.npz`.
- The parity harness already provides `ParityCase`, `assert_parity`, `bisect`, `save_case`, and `load_case` under `src/mlx_cv/parity/harness.py`.

## Acceptance Criteria

1. Checkpoint acquisition: there is an explicit tool or documented command path to place RF-DETR Nano outside git, verify the expected MD5, and expose the verified path to tests through a clear environment variable or cache convention.
2. Runtime hygiene: `src/mlx_cv/` does not import Torch, upstream RF-DETR, download clients, or `references/`, and does not inject `references` onto `sys.path`.
3. Upstream reference run: a tool/test path runs upstream RF-DETR Nano on a fixed deterministic input and captures final boxes, scores, class IDs, raw logits/boxes, and stable taps where available.
4. Local real-checkpoint run: the same verified checkpoint converts/loads into the local MLX RF-DETR Nano detection path, runs on the aligned fixed input, and produces comparable raw outputs and typed `Result.detections`.
5. Real parity comparison: `tests/test_rfdetr_upstream_parity.py` no longer contains a success-path `pytest.fail(...)` stub; with a verified checkpoint and required-gate flag it compares upstream/reference and MLX outputs within the default parity tolerance or a recorded justified model-specific tolerance not looser than `1e-3`, reports `1 passed` with no skip for the gate test, and prints the verified checkpoint path plus MD5.
6. Drift diagnosis: the parity result includes ordered tap comparison or a documented stable-tap gap, and injected drift remains localizable with `bisect` or equivalent targeted assertion.
7. Status truthfulness: the current status source and user-facing docs mark RF-DETR `UPSTREAM_PASSED` only after a non-skipped real gate pass; skipped env-gated tests, missing checkpoints, or local tiny fixtures never produce that status, and the no-checkpoint branch remains compatible with `UPSTREAM_PASSED` for normal CI.
8. Unsupported variants: existing tests still reject segmentation checkpoints and unsupported RF-DETR variants.
9. Regression: existing RF-DETR conversion, parity, predict, processor, runtime dependency guard, and full-suite tests pass after the real-checkpoint changes.

## Scope Coverage Decisions

- Included: RF-DETR Nano checkpoint cache/download or user-supplied path convention, MD5 verification, upstream capture, local MLX load/run, real parity comparison, converter/config fixes required by real Nano weights, status/docs truthfulness, runtime dependency guards.
- Deferred: LocateAnything checkpoint closeout, SAM 3.1 image checkpoint closeout, DA3 multi-view, SAM video/Object Multiplex, DEIMv2, EoMT-DINOv3, Sapiens2, YOLO26, RT-DETRv4.
- Anti-goals: do not commit raw or converted weights; do not make network access part of ordinary CI; do not claim RF-DETR is upstream-passed from a skip, blocker, local tiny fixture, or synthetic checkpoint; do not broaden scope to RF-DETR segmentation or PML variants.

## Assumptions

- RF-DETR Nano checkpoint remains available from the recorded public URL or can be supplied manually.
- The recorded MD5 remains the expected identity for the target checkpoint.
- A Torch/upstream reference environment can be installed or activated outside package runtime when the env-gated parity command runs.

## Anti-Goals

- Do not port or admit a new model family.
- Do not implement DA3 multi-view or SAM video/tracker work.
- Do not redistribute model weights.
- Do not add Torch, upstream RF-DETR, or download clients as base package dependencies.
- Do not weaken current local fixture coverage or unsupported-variant guards.
