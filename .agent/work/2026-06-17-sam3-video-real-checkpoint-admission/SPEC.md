# SPEC: SAM 3.1 Video Real Checkpoint Admission

Change: `2026-06-17-sam3-video-real-checkpoint-admission` - Stage: frame - Source: `.agent/steering/ROADMAP.md` Phase 2, prior SAM 3.1 video local-contract change, and upstream SAM 3.1 reference evidence.

## Bounded Goal

Finish roadmap Phase 2 by moving SAM 3.1 video/Object Multiplex from local deterministic contract coverage to a real SAM 3.1 checkpoint admission attempt. The result must be either a real-checkpoint gate pass or a precise `BLOCKED:<reason>` record that names the missing checkpoint, config, model access, reference runtime, local conversion, comparison tap, or numeric mismatch.

## Broader Intent

Keep `mlx-cv` checkpoint-first. The previous SAM 3.1 video/Object Multiplex change proved local session, prompt, tracking, and Object Multiplex contracts, but it did not prove that real SAM 3.1 video weights can be acquired, admitted, loaded, compared, or truthfully blocked. This phase closes that evidence gap without overstating upstream parity.

## Target User

Maintainers and users who need to know how to configure SAM 3.1 video/Object Multiplex checkpoints, where weights must live, what license or access boundary applies, and whether the local implementation is real-checkpoint ready or blocked by a named component.

## Work Scale And Shape

- Scale: one roadmap phase focused on the existing SAM 3.1 video surface.
- Shape: official source/provenance audit, out-of-git cache/download support, required checkpoint gate, upstream reference capture attempt, local comparison boundary, status/docs cleanup, and regression.
- Selected lenses: product truthfulness, engineering safety, runtime boundary, and checkpoint governance.

## Required Outcome

- Official checkpoint source is identified from upstream reference evidence: SAM 3.1 video/Object Multiplex uses Hugging Face model repo `facebook/sam3.1` with checkpoint `sam3.1_multiplex.pt` and `config.json`.
- Checkpoint access remains out of git. Any download or cache population must use an explicit out-of-git path, honor Hugging Face terms/auth requirements, and record provenance or checksum evidence when available.
- `MLX_CV_SAM3_VIDEO_CHECKPOINT`, `MLX_CV_SAM3_VIDEO_CONFIG`, `MLX_CV_SAM3_VIDEO_MODEL_ID`, and `MLX_CV_REQUIRE_SAM3_VIDEO_GATE` form the executable admission contract.
- The latest Phase 2 status artifact lives under this change directory and supersedes the earlier local-contract status for real checkpoint admission. The earlier `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json` remains historical evidence only.
- The release parity matrix remains bounded to its existing image/detection/depth rows. This phase must not add `sam3_video` to `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`.
- A required gate distinguishes unset env, missing checkpoint path, tiny/LFS/unusable checkpoint, missing config, missing auth/download access, missing reference runtime dependencies, upstream builder/runtime failure, unsupported local conversion path, missing comparison taps, numeric mismatch, and pass.
- If a usable checkpoint and reference runtime are available, the gate attempts SAM 3.1 video/Object Multiplex reference execution through upstream `build_sam3_predictor(version="sam3.1")` or `build_sam3_multiplex_video_predictor` and compares the smallest stable local outputs that are available.
- If real upstream-vs-local numeric comparison cannot be completed, the gate still completes only by recording a component-specific blocker, not by falling back to a bare "comparison is not implemented" failure or by treating local deterministic fixtures as upstream parity.
- Runtime package imports remain clean: no Torch, Hugging Face, CUDA-only packages, network clients, or imports from `references/` enter `src/mlx_cv/`.
- Docs explain the exact status, setup knobs, cache boundary, source/auth requirements, and whether the current result is a pass or a precise blocker.

## Constraints And Risks

- Hugging Face access for `facebook/sam3.1` may require accepted terms and authentication. Lack of access is a valid external blocker when recorded precisely.
- Network or checkpoint downloads require execution-time approval and must write only to out-of-git cache paths.
- The local SAM 3.1 video implementation is a deterministic MLX contract surface today, not a proven full SAM 3.1 video port. This phase must not hide that boundary.
- Upstream SAM 3.1 reference code lives under `references/sam3/` and may require Torch/runtime dependencies. Those dependencies may be used only in tools/tests, never in package runtime.
- The previous SAM 3.1 image gate intentionally rejects video/tracker checkpoints. This phase must preserve that image-mode guard while adding video-specific admission.
- Object Multiplex support must be treated as a checkpoint/source/runtime capability claim only after the real SAM 3.1 multiplex path is admitted or precisely blocked.
- Full numeric parity may be blocked by missing local converter coverage, missing stable taps, unavailable upstream runtime, or unavailable weights. A precise blocker is acceptable; an unqualified success claim is not.

## Source Evidence

- Roadmap Phase 2 objective and exit signal: `.agent/steering/ROADMAP.md`.
- Prior local SAM 3.1 video/Object Multiplex implementation: `.agent/work/2026-06-17-sam3-video-object-multiplex/`, `src/mlx_cv/models/sam3/video.py`, `src/mlx_cv/core/tracking.py`, and `tests/test_sam3_video_*`.
- Existing SAM 3.1 video gate/status: `tools/sam3_video_upstream.py` and `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`.
- Upstream SAM 3.1 release/source evidence: `references/sam3/README.md`, `references/sam3/RELEASE_SAM3p1.md`, `references/sam3/sam3/model_builder.py`, `references/sam3/scripts/qualitative_test.py`, and `references/sam3/scripts/measure_speed.py`.
- SAM 3.1 image checkpoint guard that must remain separate: `tools/sam3_image_upstream.py`, `tests/test_sam3_upstream_parity.py`, and `src/mlx_cv/models/sam3/convert.py`.
- Runtime guard: `tests/test_runtime_dependency_guards.py`.

## Acceptance Criteria

1. Phase boundary: roadmap Phase 2 is bound to `2026-06-17-sam3-video-real-checkpoint-admission`, Phase 1 stays done, Phase 3 stays pending, and no new model-family work enters this change.
2. Official source/provenance: status/docs record `facebook/sam3.1`, `sam3.1_multiplex.pt`, `config.json`, terms/auth expectations, out-of-git cache behavior, and checksum/provenance status when known.
3. Required gate contract: `MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1` cannot silently skip; it must pass only through a real pass or a precise `BLOCKED:<reason>` admission/comparison record.
4. Checkpoint/config admission: the gate distinguishes unset env, nonexistent path, directory/file shape errors, tiny or LFS-stub files, missing config, unsupported model ID, download/auth unavailable, and usable checkpoint/config pairs.
5. Download/cache hygiene: any helper or command for obtaining checkpoint/config files writes outside git, supports explicit cache paths, records source metadata, and does not make network access mandatory for default tests.
6. Upstream reference attempt: with a usable checkpoint/config and required reference dependencies, the gate attempts the SAM 3.1 video/Object Multiplex upstream path through the reference builder, fixed tiny video inputs, and stable prompt/session operations.
7. Local comparison boundary: with upstream outputs available, the gate compares the smallest stable local outputs available today; missing converter, missing local tap capture, or unsupported checkpoint-to-MLX load path becomes a named blocker.
8. No fail-stub regression: checkpoint-present paths never return a bare "comparison is not implemented" failure. A default-mode test forces this branch to a component-specific blocker when a fake admitted checkpoint is present.
9. Existing local coverage preserved: SAM 3.1 video session/tracking/Object Multiplex tests still pass, SAM 3.1 image-mode rejection of video/tracker checkpoints still passes, and `sam3_video` remains absent from the release parity matrix.
10. Runtime hygiene and regression: targeted SAM 3.1 video/image tests, status JSON validation, runtime dependency guards, `git diff --check`, and the full test suite pass or are explicitly blocked only by approved external checkpoint/runtime access.

## Scope Coverage Decisions

- Included: official SAM 3.1 video/Object Multiplex checkpoint source discovery, out-of-git cache/download contract, provenance/checksum recording, required gate behavior, reference runtime capture attempt, local comparison attempt, precise blocker handling, docs/status cleanup, and regression.
- Included: preserving the prior local SAM 3.1 video/Object Multiplex contract and the existing SAM 3.1 image/video checkpoint separation.
- Deferred: full SAM 3.1 video MLX checkpoint conversion if the phase proves a missing converter/tap/component blocker, new model-family selection, Phase 3 expansion, LocateAnything image/text closeout work, and any performance or quality claim beyond the gate result.
- Assumption: If execution cannot obtain approved network access, accepted Hugging Face terms, or a usable checkpoint, the phase still completes by recording the exact external blocker and keeping all claims conservative.

## Anti-Goals

- Do not add a new model family.
- Do not redo the already-completed local SAM 3.1 video/Object Multiplex implementation.
- Do not add `sam3_video` to the release parity matrix.
- Do not claim upstream parity from skipped tests, synthetic-only fixtures, or local deterministic contract tests.
- Do not commit or redistribute upstream weights.
- Do not import upstream reference code or heavyweight runtime dependencies from `src/mlx_cv/`.
- Do not weaken SAM 3.1 image-mode rejection of video/tracker checkpoints.
- Do not hide checkpoint, auth, config, reference-runtime, converter, tap, or comparison blockers behind generic wording.
