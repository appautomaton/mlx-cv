# DESIGN: Real Checkpoint Bootstrap - RF-DETR Nano

## Goal

Create the first real pretrained checkpoint validation path in `mlx-cv` without adding reference or download dependencies to package runtime.

## Boundary

- Runtime package code stays under `src/mlx_cv/` and remains import-light.
- Checkpoint download, Torch checkpoint extraction, and upstream reference execution live under `tools/` and env-gated tests.
- Raw upstream checkpoints and converted real weights stay outside git.
- Small derived parity cases may be committed only when they contain fixed inputs, reference outputs, and taps, not model weights.

## Flow

1. Resolve RF-DETR Nano checkpoint path.
   - Input: `MLX_CV_RFDETR_NANO_CHECKPOINT`, or cache root such as `$MLX_CV_CACHE`.
   - If download is requested, fetch `https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth`.
   - Verify MD5 `fb6504cce7fbdc783f7a46991f07639f`.

2. Run upstream reference.
   - Tool/test adds `references/rf-detr/src` only outside runtime.
   - Upstream RF-DETR Nano runs on the fixed parity image/input.
   - Capture final boxes, scores, class IDs, raw logits/boxes, and stable taps where reference APIs expose them.

3. Run local MLX.
   - Convert/extract the same checkpoint into a local-loadable representation outside git.
   - Load the MLX RF-DETR model.
   - Use aligned preprocessing/postprocessing and capture matching local taps.

4. Compare and update status.
   - Use default tolerance `atol=1e-4, rtol=1e-4` unless a tighter/looser model-specific tolerance is justified in status evidence; any tolerance looser than `1e-3` returns to planning.
   - `tests/test_rfdetr_upstream_parity.py` must compare when prerequisites exist, not fail with a placeholder.
   - Mark RF-DETR `UPSTREAM_PASSED` only after the real comparison passes.

## Gate Modes

- All env-gated RF-DETR real-checkpoint tests share the same gate semantics, preferably through one helper: normal mode may cleanly skip when no checkpoint is configured, while required mode fails instead of skipping.
- Normal CI mode may skip the upstream capture, local real-load, and upstream-vs-MLX parity gates when no checkpoint is configured. This skip must not assert RF-DETR is still `BLOCKED:` after the model has passed.
- Phase-closing mode sets an explicit required-gate flag, for example `MLX_CV_REQUIRE_RFDETR_GATE=1`. In this mode, missing checkpoint, bad checksum, missing reference dependency, or skipped capture/load/comparison is a failure.
- A successful required-gate run prints the resolved checkpoint path and MD5 so verification evidence proves the run used the real checkpoint.
- Status promotion to `UPSTREAM_PASSED` requires evidence from a required-gate run with the gate test collected and passed, not skipped.

## Status Source

Use the existing release parity status matrix as the current user-facing model status source for this bootstrap:

`.agent/work/2026-06-16-release-parity-hardening/parity-status.json`

This avoids competing status files. RF-DETR may move to `UPSTREAM_PASSED`; LocateAnything and SAM image remain blocked until their own closeout phase.

## Failure Handling

- Missing checkpoint before execution starts is not a successful phase exit.
- Bad checksum fails the phase.
- Reference dependency setup failures are execution blockers unless a user-supplied equivalent reference run is provided.
- Missing stable reference taps do not block final-output parity, but the gap must be recorded and drift diagnosis must remain as granular as available.
