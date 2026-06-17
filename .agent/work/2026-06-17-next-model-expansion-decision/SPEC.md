# SPEC: Next Model Expansion Decision

Change: `2026-06-17-next-model-expansion-decision` - Stage: frame - Source: `.agent/steering/ROADMAP.md` Phase 3, `docs/BUILDING-BLOCKS.md`, `docs/ARCHITECTURE.md`, local `references/` corpus, release parity status, and current official candidate sources.

## Bounded Goal

Finish roadmap Phase 3 by selecting exactly one next model family from DEIMv2, EoMT-DINOv3, or Sapiens2, with enough evidence to frame that selected family as the next bounded real-checkpoint admission change.

## Broader Intent

Continue expanding `mlx-cv` by evidence, not momentum. Existing core families now either have real checkpoint passes or precise blockers, so the next family should be chosen for checkpoint availability, spine-contract leverage, result-surface value, implementation risk, and truthful parity path.

## Target User

Maintainers deciding which current-generation vision family should receive the next implementation investment and future users who need to understand why that family was selected before any port claims are made.

## Work Scale And Shape

- Scale: one roadmap decision phase, not a model implementation phase.
- Shape: source/license/checkpoint inventory, candidate scoring, Result-contract impact analysis, smallest real-checkpoint admission target, selected-family decision, and follow-on framing brief.
- Selected lenses: product, engineering, runtime.

## Required Outcome

- Candidate matrix covers exactly:
  - **DEIMv2**: DINOv3-backed real-time detection, Apache-2.0 code, Hugging Face checkpoints such as `Intellindust/DEIMv2_DINOv3_S_COCO`, reference path `references/DEIMv2`.
  - **EoMT-DINOv3**: encoder-only segmentation/panoptic/instance/semantic candidate, MIT code, DINOv3 delta-weight coupling, reference path `references/eomt`.
  - **Sapiens2**: human-centric pose/body-part segmentation/surface normal/pointmap/matting candidate, Sapiens2 license, safetensors checkpoints under `facebook/sapiens2*`, reference path `references/sapiens2`.
- Decision criteria are explicit and weighted enough to make the result auditable: checkpoint accessibility, smallest credible gate, Result-contract impact, reuse of existing spine blocks, missing ops/backbone work, license/access risk, blast radius, and user-facing value.
- The phase produces a selected family and records why the other two are deferred, without deleting them from future consideration.
- The phase produces a follow-on framing brief for the selected family that names the first implementation objective, source/model ID, checkpoint/config path, out-of-git cache expectation, smallest parity/admission target, expected blocker taxonomy, and explicit anti-goals.
- The phase updates roadmap state only for this decision outcome. The selected-family implementation remains a future change with its own SPEC/PLAN.

## Constraints And Risks

- Do not implement DEIMv2, EoMT, or Sapiens2 in this change.
- Do not download model weights during this phase unless execution later requests explicit approval; selection must be possible from source metadata, local references, and existing cache evidence.
- Do not commit upstream weights, converted weights, or derived full-checkpoint artifacts.
- `src/mlx_cv/` runtime remains untouched unless execution discovers a tiny documentation-only correction is necessary; the expected output is decision artifacts and steering docs.
- Existing release parity matrix stays bounded to existing rows and must not gain the selected model during the decision phase.
- Local `references/` are evidence only. Runtime code must not import from them.
- Sapiens2 has a custom license and biometric/human-use restrictions; EoMT-DINOv3 has DINOv3 delta-weight access coupling; DEIMv2 has deformable-attention and multi-scale adapter risk despite a clearer small checkpoint path.
- If current official sources contradict local docs, execution must record the contradiction and use current primary source truth in the decision.

## Source Evidence

- Roadmap Phase 3: `.agent/steering/ROADMAP.md`.
- Current repo constraints: `.agent/steering/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`, `docs/BUILDING-BLOCKS.md`, `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`.
- Local candidate references: `references/DEIMv2/README.md`, `references/DEIMv2/configs/deimv2/`, `references/eomt/README.md`, `references/eomt/model_zoo/dinov3.md`, `references/eomt/models/eomt.py`, `references/sapiens2/README.md`, `references/sapiens2/docs/MODEL_ZOO.md`, `references/sapiens2/LICENSE.md`.
- Current primary/near-primary public sources: `https://github.com/Intellindust-AI-Lab/DEIMv2`, `https://github.com/tue-mps/eomt`, `https://github.com/facebookresearch/sapiens2`, `https://huggingface.co/facebook/sapiens2`, and `https://huggingface.co/collections/facebook/sapiens2`.
- Existing MLX surfaces to consider: `src/mlx_cv/backbones/vision/dinov3/`, `src/mlx_cv/backbones/vision/necks/`, `src/mlx_cv/heads/detection/`, `src/mlx_cv/heads/segmentation/`, `src/mlx_cv/heads/dense/`, `src/mlx_cv/core/types.py`, `src/mlx_cv/transforms/`, and `src/mlx_cv/ops/`.

## Acceptance Criteria

1. Phase boundary: roadmap Phase 3 is bound to `2026-06-17-next-model-expansion-decision`, Phase 1 and Phase 2 remain done, and no model implementation work enters this change.
2. Candidate inventory: DEIMv2, EoMT-DINOv3, and Sapiens2 each have source URL, local reference path, release/source evidence, license/access notes, model/checkpoint candidates, and expected out-of-git cache shape.
3. Checkpoint-first gate target: each candidate has a smallest credible real-checkpoint admission/parity target or a precise reason no such target is credible yet.
4. Spine impact: each candidate has a concrete `Result` contract impact and missing-block/ops assessment grounded in existing `src/mlx_cv/` surfaces and reference entry points.
5. Decision method: the selected family is chosen through explicit criteria, not preference prose; the decision artifact records scored tradeoffs and defers the other two with reasons.
6. Follow-on framing: the selected family has a next-change brief with objective, likely slug, model ID/checkpoint source, cache/env variables, first gate command shape, docs/status artifact expectation, and anti-goals.
7. Runtime hygiene: no `src/mlx_cv/` reference imports, no new runtime dependency, no committed weights, and no `sam3_video` or new family row added to the existing release parity matrix.
8. Verification: artifact checks, source/link checks, release-matrix bound check, runtime dependency guard, and `git diff --check` pass.

## Scope Coverage Decisions

- Included: selecting exactly one family from DEIMv2, EoMT-DINOv3, and Sapiens2; recording evidence and tradeoffs; preparing a follow-on implementation brief.
- Included: current official source verification for candidate availability and release/checkpoint claims.
- Deferred: implementing the selected model, downloading weights, converting weights, adding runtime APIs, adding new `Result` fields, adding parity rows, or editing `src/mlx_cv/`.
- Deferred: YOLO26 remains watchlist-only and RT-DETRv4 remains dropped unless a future roadmap update changes that.

## Anti-Goals

- Do not port a model in this phase.
- Do not pick more than one family.
- Do not broaden candidate scope beyond DEIMv2, EoMT-DINOv3, and Sapiens2.
- Do not claim upstream parity or local fixture support for the selected family.
- Do not commit weights or converted checkpoints.
- Do not add heavyweight runtime dependencies or imports from `references/`.
- Do not mutate prior Phase 1 or Phase 2 status artifacts except by linking them as evidence.
