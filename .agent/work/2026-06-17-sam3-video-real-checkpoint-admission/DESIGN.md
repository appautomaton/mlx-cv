# DESIGN: SAM 3.1 Video Real Checkpoint Admission

Change: `2026-06-17-sam3-video-real-checkpoint-admission` - Stage: plan - Spec: `SPEC.md`

## Design Goal

Make SAM 3.1 video/Object Multiplex checkpoint readiness explicit and executable without changing the package runtime boundary. The design must answer four questions with evidence: where the checkpoint comes from, whether it is usable locally, whether upstream reference execution can run, and whether local MLX comparison can pass or is blocked by a named component.

## Artifact Ownership

- Canonical checkpoint-admission status: `.agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json`.
- Historical local-contract status: `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`.
- Existing release parity matrix: `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`.

The new status artifact owns the latest SAM 3.1 video checkpoint-admission truth. The historical local-contract status remains evidence that deterministic video/session/Object Multiplex plumbing exists. The release parity matrix stays scoped to the existing release rows and must not gain a `sam3_video` entry in this phase.

## Status Schema

The status artifact should stay small and machine-checkable:

- `schema_version`
- `phase`
- `model`
- `display_name`
- `status`
- `claim_level`
- `blocked_reason`
- `blocker_kind`
- `checkpoint_env`
- `config_env`
- `model_id_env`
- `cache_dir_env`
- `required_gate_env`
- `official_model_id`
- `checkpoint_name`
- `config_name`
- `source_url`
- `license_or_terms`
- `provenance_status`
- `checkpoint_sha256`
- `config_sha256`
- `reference_path`
- `reference_surfaces`
- `comparison_scope`

`status` is `UPSTREAM_PASSED` only after checkpoint/config admission, upstream reference execution, local execution, and comparison all succeed. Otherwise it is `BLOCKED:<specific reason>`.

## Checkpoint Source And Cache Boundary

Upstream evidence points SAM 3.1 video/Object Multiplex at Hugging Face repo `facebook/sam3.1`, checkpoint `sam3.1_multiplex.pt`, and `config.json`. The admission code should treat those as defaults while still allowing explicit user paths.

Expected knobs:

- `MLX_CV_SAM3_VIDEO_MODEL_ID`: defaults to `facebook/sam3.1` for source metadata.
- `MLX_CV_SAM3_VIDEO_CHECKPOINT`: explicit local checkpoint path.
- `MLX_CV_SAM3_VIDEO_CONFIG`: explicit local config path.
- `MLX_CV_SAM3_VIDEO_CACHE_DIR`: optional out-of-git cache root for fetch/admission helpers.
- `MLX_CV_REQUIRE_SAM3_VIDEO_GATE`: required-mode switch.

Default tests must not require network. If a helper supports Hugging Face download, it must be opt-in, write outside git, and surface missing auth/terms/dependency/network as a precise blocker.

## Gate State Machine

`tools/sam3_video_upstream.py` should evaluate the gate in a stable order:

1. Source metadata: identify model ID, checkpoint name, config name, and terms/auth note.
2. Path/cache resolution: resolve explicit checkpoint/config paths or optional cache paths.
3. Admission: reject unset env, missing paths, directories where files are required, tiny or LFS-stub files, missing config, unsupported model ID, and unavailable download/auth.
4. Reference readiness: ensure `references/sam3/` and required reference dependencies are usable without importing them from package runtime.
5. Reference execution: call upstream SAM 3.1 video/Object Multiplex builder paths on fixed tiny video inputs and prompt/session operations when checkpoint/config are usable.
6. Local execution/comparison: compare the smallest stable outputs the MLX local surface can produce; otherwise report the missing converter, tap, or comparator.
7. Status write: emit `UPSTREAM_PASSED` or one precise `BLOCKED:<reason>`.

Required mode may pass with a blocker only when the blocker has been written and asserted. It must not pass through a skip.

## Runtime Boundary

All Torch, Hugging Face, and reference-code work stays in tests or tools. `src/mlx_cv/` must not import from `references/`, `torch`, `huggingface_hub`, CUDA-only packages, or network clients. The runtime dependency guard remains a required verification command.

## Reference And Comparison Ladder

The execution should climb this ladder and stop at the first precise blocker:

1. Official source identified.
2. Checkpoint/config path available.
3. Checkpoint/config shape admitted.
4. Reference runtime and upstream builder importable.
5. Upstream SAM 3.1 video/Object Multiplex session runs on fixed inputs.
6. Local MLX path can load or represent the checkpoint-relevant state.
7. Stable local outputs or taps can be compared.
8. Numeric comparison passes within the selected tolerance.

This phase is allowed to end at a component blocker because the current local video implementation may not yet include full real-checkpoint conversion. It is not allowed to end with ambiguous wording or to call local deterministic coverage upstream parity.

## Test Strategy

- Unit tests cover admission states without network: unset env, missing file, tiny file, missing config, fake admitted checkpoint/config, unsupported source, and cache metadata.
- A default-mode test forces the fake admitted checkpoint branch to the component-specific comparison blocker, preventing regression to a bare fail-stub.
- Reference-path tests are optional or env-gated when real dependencies/checkpoints are absent, but required mode must assert the exact blocker.
- Existing local SAM 3.1 video/Object Multiplex tests continue to run.
- Existing SAM 3.1 image tests continue to prove video/tracker checkpoint rejection.
- Runtime dependency guards and full regression close the phase.
