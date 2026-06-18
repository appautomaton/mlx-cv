# SPEC: Existing Checkpoint Closeout

Change: `2026-06-17-existing-checkpoint-closeout` - Stage: frame - Source: `.agent/steering/ROADMAP.md` Phase 1, `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, current LocateAnything and SAM 3.1 image gates.

## Bounded Goal

Finish roadmap Phase 1 by resolving the remaining existing-family checkpoint blockers for LocateAnything-3B and SAM 3.1 image-mode: each path must either pass a real-checkpoint upstream/reference gate or retain a precise `BLOCKED:<reason>` record that names the missing checkpoint, reference runtime, tap path, conversion/load component, or comparison component.

## Broader Intent

Keep `mlx-cv` checkpoint-first before adding another model family. RF-DETR Nano and DA3-SMALL already have real upstream-vs-MLX passes; LocateAnything and SAM 3.1 image-mode should no longer sit behind fail-stub gates that become unimplemented once a checkpoint appears.

## Target User

Maintainers and users who need to know whether the existing grounding and image segmentation surfaces can run real upstream weights locally, what exact artifact or runtime unblocks them, and which claims are still only local-fixture coverage.

## Work Scale And Shape

- Scale: phase-sized hardening change over two existing model paths.
- Shape: checkpoint source/provenance audit, out-of-git cache admission, required blocker gates, optional upstream/reference comparison, status/docs cleanup, and regression.
- Selected lenses: product, engineering, runtime.

## Required Outcome

- **LocateAnything-3B:** identify the official checkpoint source and license/provenance, reject the current 135-byte LFS stubs as unusable, support an out-of-git checkpoint directory through `MLX_CV_LOCATEANYTHING_CHECKPOINT`, and replace the current fail-stub parity test with pass-or-precise-blocker behavior.
- **SAM 3.1 image-mode:** identify the official image checkpoint source and license/provenance, support an out-of-git checkpoint through `MLX_CV_SAM3_IMAGE_CHECKPOINT`, audit the upstream image predictor/tap path, and replace the current fail-stub parity test with pass-or-precise-blocker behavior.
- Checkpoint downloads or cache population happen outside git and only through explicit execution-time approval or existing local paths.
- `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` remains the bounded release-parity status source. This change may update only the `locateanything` and `sam3_image` entries and must not add `sam3_video` or new model-family rows.
- Docs and steering text describe full-checkpoint parity only for gates that actually pass. Otherwise they keep local fixture coverage and blocker wording.
- Runtime package imports and base dependencies remain clean: no `torch`, `transformers`, `triton`, CUDA-only packages, `huggingface_hub`, network clients, or imports from `references/` enter `src/mlx_cv/`.

## Constraints And Risks

- `references/` is evidence and optional test/runtime material; `src/mlx_cv/` must not import from it or inject it into `sys.path`.
- Upstream weights are user-fetched or approval-fetched and are never committed.
- LocateAnything weights use NVIDIA non-commercial terms. This change surfaces that license boundary but does not gate code support on use case.
- The local `references/LocateAnything-3B/model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors` files are 135-byte LFS stubs, not usable checkpoint shards.
- SAM3 reference source exists under `references/sam3/`, and upstream scripts can build predictors with an optional `checkpoint_path`; stable image-mode taps still need to be proven in this workspace.
- If a real checkpoint is available but the local converter/model cannot load or compare it yet, the correct outcome is a component-specific blocker, not a pytest fail that says the comparison is unimplemented.
- Default parity tolerance remains the release-parity policy from `parity-status.json`; loosening beyond `max_without_replan` requires returning to planning.

## Source Evidence

- Phase 1 objective and exit signal: `.agent/steering/ROADMAP.md`.
- Current status matrix: `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`.
- Current fail-stub gates: `tests/test_la_upstream_parity.py`, `tests/test_sam3_upstream_parity.py`.
- LocateAnything local code and loader: `src/mlx_cv/models/locateanything/`, especially `convert.py`.
- LocateAnything reference source and stubs: `references/LocateAnything-3B/README.md`, `references/LocateAnything-3B/model-00001-of-00002.safetensors`, `references/LocateAnything-3B/model-00002-of-00002.safetensors`.
- SAM 3.1 local code and loader: `src/mlx_cv/models/sam3/`, especially `convert.py`.
- SAM 3.1 reference source: `references/sam3/sam3/__init__.py`, `references/sam3/sam3/model_builder.py`, `references/sam3/scripts/qualitative_test.py`.
- Runtime guard: `tests/test_runtime_dependency_guards.py`.

## Acceptance Criteria

1. Phase boundary: roadmap Phase 1 is bound to `2026-06-17-existing-checkpoint-closeout`, and no Phase 2/3 work is pulled into this change.
2. Checkpoint provenance: LocateAnything and SAM 3.1 image entries record source, license, out-of-git cache expectations, checkpoint env vars, and checksum/provenance status when known.
3. LocateAnything admission: the gate distinguishes unset env, nonexistent path, LFS stub directory, incomplete shard set, dependency absence, converter/load failure, missing upstream comparator, numeric mismatch, and upstream pass.
4. LocateAnything real path: when a usable checkpoint and reference runtime are available, the required gate attempts a real upstream/reference-vs-MLX comparison for fixed grounding inputs, including decoded boxes/points and stable taps where available.
5. LocateAnything blocker path: when the real path cannot run, the gate passes only after asserting a precise `BLOCKED:<reason>` status and never presents local integration fixture success as full-checkpoint parity.
6. SAM 3.1 image admission: the gate distinguishes unset env, nonexistent path, unusable checkpoint, dependency absence, image-vs-video checkpoint mismatch, missing stable image tap capture, converter/load failure, numeric mismatch, and upstream pass.
7. SAM 3.1 image real path: when a usable checkpoint and reference runtime are available, the required gate attempts a real upstream/reference-vs-MLX comparison for text and PCS-style image prompts, including masks, paired detections, token/text evidence, and stable taps where available.
8. SAM 3.1 image blocker path: when the real path cannot run, the gate passes only after asserting a precise `BLOCKED:<reason>` status and never presents local tiny image fixtures as upstream parity.
9. Status truthfulness: README, steering docs, and architecture docs derive LocateAnything and SAM 3.1 image status from `parity-status.json`; passing models are called passed, blocked models are called blocked, and local fixtures are labeled local.
10. Runtime hygiene and regression: targeted gates, local fixture tests, converter tests, runtime dependency guards, and the full test suite pass without adding reference dependencies to package runtime.

## Scope Coverage Decisions

- Included: LocateAnything checkpoint/source audit, usable-checkpoint detection, full-checkpoint gate behavior, reference/tap comparison attempt where feasible, status update, docs update, and local regression.
- Included: SAM 3.1 image checkpoint/source audit, image checkpoint admission, upstream image predictor/tap audit, reference/tap comparison attempt where feasible, status update, docs update, and local regression.
- Deferred: SAM 3.1 video/Object Multiplex real checkpoint admission, DA3 deferred branches, new model-family selection, DEIMv2, EoMT-DINOv3, Sapiens2, YOLO26, RT-DETRv4.
- Assumption: If execution cannot obtain approved network access or required model access, the phase still completes by recording the exact external blocker and keeping claims conservative.

## Anti-Goals

- Do not add a new model family.
- Do not implement SAM3 video, Object Multiplex, or tracker checkpoint parity.
- Do not commit or redistribute upstream weights.
- Do not vendor upstream source into `src/mlx_cv`.
- Do not add Torch, Transformers, Hugging Face, or network libraries to package runtime.
- Do not remove or weaken existing image-vs-video checkpoint rejection.
- Do not claim upstream parity from skipped tests, local deterministic fixtures, or synthetic-only inputs.
