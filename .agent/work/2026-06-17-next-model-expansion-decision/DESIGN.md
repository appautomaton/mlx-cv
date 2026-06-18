# DESIGN: Next Model Expansion Decision

Change: `2026-06-17-next-model-expansion-decision` - Stage: plan - Spec: `SPEC.md`

## Design Goal

Make the next model-family choice auditable and checkpoint-first. This change should end with one selected family and a follow-on implementation brief, not with a model port or subjective recommendation.

## Decision Artifacts

Execution should produce these artifacts under `.agent/work/2026-06-17-next-model-expansion-decision/`:

- `CANDIDATE-MATRIX.md`: one section per candidate with source URL, local reference path, release/checkpoint evidence, license/access notes, likely checkpoint/cache layout, runtime/reference dependency notes, and first gate hypothesis.
- `SPINE-IMPACT.md`: candidate-by-candidate impact on `Result`, processor/transform path, backbone/neck/head reuse, missing ops, and likely local fixture shape.
- `DECISION.md`: scored comparison, selected family, deferred-family reasons, and confidence/risk notes.
- `NEXT-CHANGE-BRIEF.md`: follow-on implementation objective for the selected family with source/model ID, checkpoint path/env names, smallest real-checkpoint admission target, expected blocker taxonomy, and anti-goals.

These are decision artifacts. They should link evidence and name paths, but they should not duplicate whole reference docs.

## Candidate Evaluation Rubric

Use a 100-point rubric so the selected family can be reviewed:

| Criterion | Weight | Meaning |
|---|---:|---|
| Checkpoint-first feasibility | 30 | A small real checkpoint, config, cache plan, and reference runtime can plausibly be admitted without ambiguous blockers. |
| Spine leverage | 20 | The model reuses existing MLX blocks or exercises a missing block that clearly benefits future models. |
| Result-surface value | 20 | The family adds a valuable typed output surface without forcing broad unrelated API churn. |
| Implementation risk | 15 | Missing ops, converter complexity, reference dependency risk, and expected parity/tap difficulty. Higher score means lower risk. |
| License/access cleanliness | 10 | Code/weights can be used as external artifacts with clear license notes and no redistribution ambiguity. |
| Roadmap fit | 5 | The choice follows the checkpoint-first order and does not reopen already closed blockers. |

Execution may adjust sub-scores only with evidence. If two candidates are close, select the one with the smaller real-checkpoint gate and cleaner follow-on spec.

## Candidate-Specific Gate Hypotheses

- **DEIMv2:** likely gate starts with `Intellindust/DEIMv2_DINOv3_S_COCO` or the smallest DINOv3-backed COCO variant. It should evaluate DINOv3STAs, spatial prior adapter, hybrid encoder, DEIM transformer, postprocessor, and deformable-attention risk against existing RF-DETR/detection surfaces.
- **EoMT-DINOv3:** likely gate starts with an EoMT-DINOv3 COCO panoptic or ADE20K semantic checkpoint. It must account for delta weights relative to DINOv3 and the encoder-only query-token/mask-head design.
- **Sapiens2:** likely gate starts with the smallest useful human-centric checkpoint. It must decide whether a pretrain-only backbone gate is acceptable or whether a task head such as body-part segmentation is required for a user-visible `Result` surface.

## Runtime Boundary

Reference code and heavyweight dependencies remain evidence only. This phase should update steering/work artifacts, not package runtime. Any future selected-family implementation must keep downloads and weights out of git and isolate reference/Torch/Hugging Face work to tools/tests.
