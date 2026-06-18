# SPEC: Release Parity Hardening

Change: `2026-06-16-release-parity-hardening` - Stage: frame - Source: `.agent/steering/ROADMAP.md` Phase 1, current model implementations, local `references/`.

## Bounded Goal

Harden the existing LocateAnything, RF-DETR, and SAM 3.1 image-mode paths by replacing or qualifying local fixture confidence with upstream-reference/full-checkpoint parity evidence.

## Broader Intent

Complete Phase 1 of the roadmap without scope creep or scope removal: strengthen release claims for the already implemented model paths before starting DA3 multi-view, SAM video/tracker, or any new model family.

## Target User

Library maintainers and users who need to know which MLX-native model paths match their upstream reference implementations, which paths are still local-fixture-only, and which claims are blocked by missing external checkpoints or reference environments.

## Work Scale And Shape

- Scale: phase-sized hardening change covering three existing model families.
- Shape: parity tooling, reference-run capture, converter/config corrections, env-gated verification, and truthful status docs.
- Selected lenses: product, engineering, runtime.

## Required Outcome

- **LocateAnything-3B:** an env-gated full-checkpoint/reference parity workflow for the local tokenizer-backed VLM path, or a precise blocker if the NVIDIA/Hugging Face checkpoint/runtime cannot be obtained in execution.
- **RF-DETR:** an upstream-reference parity workflow for **RF-DETR Nano COCO** (`rfdetr-nano`, Apache 2.0), chosen as the smallest non-deprecated Apache detection variant; local tiny detector fixtures remain fast CI coverage.
- **SAM 3.1 image-mode:** an upstream image-mode reference parity workflow for text and PCS box/exemplar prompts where stable reference tap points are available, or a precise blocker if the SAM3 reference/runtime does not expose stable image-mode taps in this workspace.
- Each workflow separates out-of-band reference/Torch execution from package runtime.
- No upstream weights are committed. Full-checkpoint parity commands are env/path-gated and expect user-fetched weights or approved downloads.
- A per-model hardening matrix is maintained at `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` with status values `LOCAL_FIXTURE_ONLY`, `UPSTREAM_PASSED`, or `BLOCKED:<reason>`. Skipped env-gated tests must never be interpreted as upstream parity success.
- Parity tolerance policy is explicit: use the harness default `atol=1e-4, rtol=1e-4` unless a model-specific gate records a justified tolerance in `parity-status.json`; loosening beyond `1e-3` requires returning to planning.
- Docs/status text distinguishes upstream-reference/full-checkpoint parity from local tiny-oracle or local integration fixtures.
- Runtime package imports and base dependencies remain clean: no `torch`, `transformers`, `triton`, CUDA-only packages, or imports from `references/` enter `src/mlx_cv/`.

## Constraints And Risks

- `references/` is read-only reference material: `src/mlx_cv/` must never import from it.
- Weight licenses are surfaced, not gated; weights remain user-fetched and are not redistributed.
- RF-DETR Plus XL/2XL and PML-licensed variants are out of scope for the RF-DETR parity target.
- RF-DETR segmentation checkpoints remain out of scope and must continue to be rejected by the detection loader.
- SAM 3.1 video/tracker/Object Multiplex is out of scope; this phase only covers existing image-mode SAM 3.1.
- Full parity may require Torch/reference environments and external checkpoints. If a prerequisite cannot be obtained, the correct result is a recorded blocker and truthful docs, not a stronger release claim.
- High-risk drift areas are LocateAnything token/coordinate decoding, RF-DETR postprocess/top-k and decoder/reference-point handling, and SAM text/tokenizer plus mask/box postprocess.

## Source Evidence

- Roadmap Phase 1 is Release Parity Hardening and now covers LocateAnything, RF-DETR, and SAM 3.1 image-mode.
- Current status says LocateAnything upstream full-checkpoint parity is deferred, RF-DETR has a committed tiny detector fixture with full upstream checkpoint parity deferred, and SAM 3.1 image-mode has committed tiny image fixtures with video/tracker deferred: `README.md`, `docs/ARCHITECTURE.md`, `.agent/steering/PROJECT.md`, `.agent/steering/REQUIREMENTS.md`.
- Current model paths live under `src/mlx_cv/models/locateanything/`, `src/mlx_cv/models/rfdetr/`, and `src/mlx_cv/models/sam3/`.
- Current mint tools are `tools/mint_locateanything_fixture.py`, `tools/mint_rfdetr_fixture.py`, and `tools/mint_sam3_fixture.py`.
- Current local fixture tests include `tests/test_la_integration_fixture.py`, `tests/test_la_parity.py`, `tests/test_rfdetr_parity.py`, and `tests/test_sam3_parity.py`.

## Acceptance Criteria

1. LocateAnything scope: the change documents the exact upstream checkpoint/runtime target, expected license caveat, required local inputs, tolerance policy, and whether full-checkpoint reference parity is available or blocked.
2. LocateAnything parity: an env-gated command compares local MLX LocateAnything outputs to upstream/reference outputs for fixed grounding prompts, including decoded boxes/points and stable intermediate taps when available, or records `BLOCKED:<reason>` in `parity-status.json` with the missing prerequisite. An unset env var may skip the pytest gate, but the skip is never counted as parity passed.
3. RF-DETR scope: the change documents RF-DETR Nano COCO as the first upstream detection checkpoint target, including upstream filename, expected MD5, Apache 2.0 status, and explicit exclusion of Plus/PML and segmentation variants.
4. RF-DETR parity: an env-gated command compares local MLX RF-DETR outputs to upstream RF-DETR outputs for final boxes, scores, class IDs, and stable intermediate taps; `bisect` localizes injected drift. An unavailable checkpoint/reference environment records `BLOCKED:<reason>`, not success.
5. SAM 3.1 image scope: the change documents the exact SAM 3.1 image-mode reference target and excludes video/tracker/Object Multiplex runtime work.
6. SAM 3.1 image parity: an env-gated command compares local MLX SAM 3.1 image outputs to upstream reference outputs for text and PCS box/exemplar prompts, including masks, paired boxes/scores, token/text path evidence, and stable taps when available, or records `BLOCKED:<reason>` with the missing prerequisite.
7. Fast CI remains useful: existing local tiny/integration fixture tests for all three model families continue to pass and remain clearly labeled as local coverage where they are not upstream full-checkpoint parity.
8. Runtime hygiene: package runtime and `core/` remain free of `torch`, `transformers`, `triton`, CUDA-only dependencies, imports from `references/`, and `sys.path` injection of `references` under `src/mlx_cv/`.
9. Status truthfulness: README, steering docs, and architecture status text derive each model's hardening status from `parity-status.json`; they describe a model as upstream-reference/full-checkpoint hardened only if its env-gated parity command passes, otherwise they retain explicit hardening-gap/blocker wording.
10. Regression: targeted LocateAnything, RF-DETR, SAM 3.1, parity harness, conversion, processor, and runtime guard tests pass, followed by the full test suite.

## Scope Coverage Decisions

- Included: LocateAnything full-checkpoint/reference parity, RF-DETR Nano detection checkpoint parity, SAM 3.1 image-mode upstream reference parity, reference capture tooling, converter/config fixes required by those parity targets, env-gated parity tests/commands, docs/status truthfulness, runtime dependency guards.
- Deferred: SAM 3.1 video/tracker/Object Multiplex, DA3 multi-view, RF-DETR segmentation variants, RF-DETR Plus XL/2XL, DEIMv2, EoMT-DINOv3, Sapiens2, YOLO26, RT-DETRv4.
- Assumption: Full parity can be completed only when the required checkpoint/reference prerequisites are locally available or approved for download; otherwise the phase records blockers and keeps status truthful.

## Anti-Goals

- Do not port a new model family.
- Do not implement SAM video/tracker/Object Multiplex.
- Do not broaden RF-DETR to all variants or segmentation checkpoints.
- Do not commit upstream weights or vendor reference code.
- Do not add Torch/reference dependencies to package runtime.
- Do not weaken existing unsupported-variant rejections.
- Do not claim release parity from skipped, local-only, or synthetic fixture runs.
