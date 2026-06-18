# SPEC: SAM 3.1 Video / Object Multiplex

## Bounded Goal

Frame and implement the entire roadmap Phase 3: add the deferred SAM 3.1 video and tracker memory path, using the upstream names and boundaries precisely:

- SAM3 Video for concept/text video detection and tracking.
- Sam3Tracker for visual-prompt segmentation where it applies.
- SAM 3.1 Object Multiplex for multi-object video tracking state and batching shape.

The phase is complete when a short fixed video clip can produce stable tracked object IDs and masks through the shared `Result` surface, image-mode SAM 3.1 behavior does not regress, memory behavior is covered by deterministic fixtures, and the real video checkpoint gate either passes or reports a precise external blocker.

## Broader Intent

`mlx-cv` should grow from image-mode SAM 3.1 into a truthful video-tracking admission path without overstating parity. This phase should carry forward the real-checkpoint discipline established in the previous phases: implemented surfaces must be runnable, unsupported branches must fail loudly, and missing checkpoint/config access must be represented as a blocker rather than a silent success.

## Work Scale And Shape

This is a capability-sized phase, not a narrow patch. It may touch:

- `src/mlx_cv/models/sam3/`
- `src/mlx_cv/core/types.py`
- tracker or memory helper modules if needed
- conversion and checkpoint-admission utilities
- tests, fixtures, docs, and roadmap/status artifacts

The work should stay within the SAM 3.1 video/tracker boundary. It should not absorb unrelated roadmap phases or change DA3, RF-DETR, LocateAnything, or unrelated model families.

## Product Lens

The user-facing outcome is a local SAM 3.1 video tracking capability that a caller can reason about:

- load or admit a video-capable checkpoint when available,
- start video/session state from a short frame sequence,
- add a text or visual prompt according to the selected upstream mode,
- propagate tracked objects across frames,
- receive typed per-frame detections, masks, and stable track IDs.

When a real checkpoint is unavailable, the caller should get a precise blocker that names what is missing and what action would unblock it.

## Engineering Lens

The implementation should align with the current codebase rather than importing the upstream PyTorch stack into runtime code. Reference code under `references/sam3/` is evidence for naming, architecture, and behavior, but `src/mlx_cv` should remain MLX-native and keep optional decoding or reference-comparison tooling out of the core runtime path.

The existing SAM 3.1 image-mode path is a constraint. Phase 3 may share prompt, processor, tokenizer, and output types where appropriate, but it must not weaken current image-mode errors, conversion gates, or tests.

## Current Evidence

- Roadmap Phase 3 is pending and titled `SAM 3.1 Video / Object Multiplex`.
- Local SAM 3.1 support is currently image-mode only.
- Existing SAM3 prompt and processor code rejects video, tracker, memory, and mask-prompt keys.
- Existing SAM3 conversion code rejects video/tracker checkpoint keys.
- `Result` already has optional `detections`, `masks`, and `tracks`; `Detections` has `track_ids`, while `Tracks` is still minimal.
- Upstream reference evidence includes `build_sam3_video_predictor`, `build_sam3_multiplex_video_predictor`, `Sam3TrackerPredictor`, `SimpleMaskEncoder`, `MultiplexController`, `VideoTrackingDynamicMultiplex`, and the `start_session` / `add_prompt` / `propagate_in_video` request flow.
- `references/sam3/assets/videos/0001/` provides a concrete fixed frame sequence candidate for deterministic tests.

## Scope

### Included

- Audit the upstream SAM 3.1 video, tracker, and multiplex reference path enough to define a local contract for Phase 3.
- Define a checkpoint/config admission gate for video-capable SAM 3.1 checkpoints, including exact failure behavior when checkpoint access is unavailable.
- Add or extend typed video/tracking result surfaces so per-frame masks, detections, track IDs, frame indices, and masklet-like outputs can be represented without breaking existing `Result` callers.
- Add a tracker/session state API aligned with the upstream flow: initialize video state, add prompt, and propagate/update tracked objects.
- Support text/concept video tracking through the SAM3 Video boundary.
- Support visual-prompt tracking through the Sam3Tracker boundary where applicable.
- Add Object Multiplex-aware state and batching shape for multiple objects, including fixed-capacity grouping metadata.
- Add deterministic short-clip fixtures and tests for stable IDs, masks, memory behavior, and multi-object grouping.
- Preserve image-mode SAM 3.1 behavior and conversion errors.
- Update docs and roadmap/status artifacts so the implemented behavior, checkpoint requirements, and remaining blockers are clear.

### Conditional Or Optional

- Video-file decoding can live behind optional dependencies or tooling. A frame-sequence path is sufficient for the core acceptance path if it is deterministic and documented.
- Reference PyTorch comparisons can be provided as an opt-in tool or gate when the upstream environment and checkpoint are configured. They must not be required for default unit tests unless the required assets are present.

### Anti-Goals

- Do not claim full upstream SAM 3.1 video parity unless a real checkpoint gate supports that claim.
- Do not bundle model weights in the repository.
- Do not port the upstream PyTorch runtime into `src/mlx_cv`.
- Do not implement training, large-scale evaluation, SA-Co, VEval, or H100 performance benchmarking.
- Do not claim upstream Object Multiplex speedups from local shape tests alone.
- Do not build a video service, GUI, streaming server, or notebook-first workflow.
- Do not rework SAM 3.1 image-mode behavior beyond shared interfaces required by this phase.
- Do not modify unrelated model families or roadmap phases.

## Required Outcomes

### R1. Reference And Gate Contract

The phase must produce a clear local contract for SAM3 Video, Sam3Tracker, Object Multiplex, checkpoint key families, config requirements, supported prompts, and unsupported branches. The gate must distinguish:

- local implementation failure,
- missing checkpoint/config,
- unsupported upstream branch,
- optional reference-environment absence.

### R2. Typed Video Result Surface

The shared result types must represent video tracking outputs without forcing callers into ad hoc dictionaries. Required information includes frame indices, object or track IDs, masks, detection scores or labels where available, and enough state metadata to describe memory/multiplex behavior.

### R3. Session And Prompt API

There must be a local API for starting video state, adding prompts, and propagating tracked results. It should preserve upstream vocabulary where useful while fitting the existing `mlx_cv` style.

### R4. Local Tracker/Memory Path

The implementation must include a deterministic local path that exercises tracker state, memory updates, and propagation over a short fixed clip. If a full MLX port of an upstream component is blocked by checkpoint/config access, the blocker must be isolated and recorded while keeping the typed state and fixture behavior testable.

### R5. Object Multiplex Shape

Multi-object tracking must use an Object Multiplex-aware representation rather than only a single-object loop. Tests should cover at least two tracked objects and validate stable grouping metadata or bucket assignment behavior.

### R6. Verification And Documentation

The phase must end with focused tests plus the existing regression suite needed to protect image-mode behavior. Documentation must describe supported inputs, checkpoint setup, output shape, known blockers, and the exact claim level reached.

## Constraints And Risks

- The exact public availability and naming of SAM 3.1 video/multiplex checkpoint files may be external to the repo. Treat this as a gate input, not as an implementation excuse.
- Object Multiplex architecture details should be taken from `references/sam3/`, not inferred from marketing language.
- Runtime code should avoid hard dependencies on OpenCV, Torch, CUDA, or xformers.
- Tests should remain deterministic and practical for local execution.
- Reference assets under `references/` can inform fixtures, but generated or copied test fixtures should stay small and intentional.
- The existing SAM3 image-mode conversion gate currently rejects video/tracker keys. Phase 3 must update or separate that behavior without making image-mode checkpoint handling ambiguous.

## Acceptance Criteria

- A reference audit or equivalent code/docs artifact names the selected SAM3 Video, Sam3Tracker, and Object Multiplex surfaces and maps them to local modules and checkpoint gates.
- The public output path can return per-frame masks with stable object or track IDs through typed result structures.
- A short fixed frame sequence can be processed with a text/concept prompt and produces deterministic tracked IDs and masks.
- Visual-prompt tracking is implemented where applicable, or explicitly blocked with the upstream reason and a test covering the failure mode.
- Multi-object Object Multiplex grouping is represented in state and covered by deterministic tests.
- Missing video checkpoint/config access produces a precise required-mode failure, not a skip that looks like success.
- Existing SAM 3.1 image-mode tests and conversion-gate behavior still pass.
- Docs and roadmap/status artifacts state the supported inputs, checkpoint requirements, limitations, and verification commands.

