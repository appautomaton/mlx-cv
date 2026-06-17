# Decision: Next Model Expansion

Change: `2026-06-17-next-model-expansion-decision`

## Verdict

Select **EoMT-DINOv3** as the next model-family implementation target.

This is a selection for the next bounded real-checkpoint admission change. It is not an implementation claim, not a local support claim, and not an upstream parity claim.

## Scored Rubric

Weights come from `DESIGN.md`: checkpoint-first feasibility 30, spine leverage 20, result-surface value 20, implementation risk 15, license/access cleanliness 10, roadmap fit 5.

| Family | Checkpoint (30) | Spine (20) | Result (20) | Implementation (15) | License (10) | Roadmap (5) | Score | Status | Evidence note |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| DEIMv2 | 24 | 14 | 12 | 8 | 8 | 3 | 69 | Status: deferred | Clear HF-linked checkpoint path, but it deepens the already-covered detection lane and adds STA/deformable/multi-scale decoder work. |
| EoMT-DINOv3 | 20 | 18 | 18 | 11 | 8 | 5 | 80 | Status: selected | Best balance: reuses DINOv3, opens segmentation/panoptic mask outputs, keeps `Result` churn low, and has a small COCO panoptic first gate. |
| Sapiens2 | 16 | 12 | 17 | 5 | 3 | 3 | 56 | Status: deferred | High user value, but custom license restrictions, larger checkpoints, and normals/pointmaps/matting result-surface pressure make it too broad for the next gate. |

## Why EoMT-DINOv3 Wins

EoMT-DINOv3 is the most useful next stress test for the library spine. It exercises the existing DINOv3 backbone in a different output pillar, adds mask/class outputs that map naturally into `Result.masks`, and avoids both the detection overlap of DEIMv2 and the broad human-centric surface expansion of Sapiens2.

The main EoMT risk is not hidden: the DINOv3 EoMT weights are deltas against original DINOv3 weights. That makes the next phase a checkpoint-admission phase, not a promise of immediate parity. If DINOv3 base access or delta composition is blocked, the next change should record a precise blocker instead of claiming a pass.

## Deferred Candidates

### DEIMv2

DEIMv2 has the clearest detection-shaped `Result` fit and a public model-zoo path for `Intellindust/DEIMv2_DINOv3_S_COCO`. It is deferred because the project already has RF-DETR Nano as the real-checkpoint detection anchor, and DEIMv2 would spend the next phase on another detection family before opening a stronger segmentation or human-centric surface. It also carries nontrivial STA, hybrid encoder, postprocessor, and possible deformable/multi-scale sampling work.

Revisit DEIMv2 after EoMT proves the DINOv3 reuse path for segmentation or if a later roadmap explicitly prioritizes real-time detector breadth.

### Sapiens2

Sapiens2 has the broadest user-facing value: body-part segmentation, pose, normals, pointmaps, and matting. It is deferred because its license has explicit human/biometric-use restrictions, its task family is wide, and several outputs would force future `Result` widening beyond the first gate. A body-part segmentation gate could use `Result.masks`, but the overall implementation would still need the Sapiens2 backbone variants, task heads, large-resolution transforms, and careful license notes.

Revisit Sapiens2 when the project is ready for a dedicated human-centric phase with explicit license handling and result-surface decisions.

## Risk Register

| Risk | Handling in next change |
|---|---|
| DINOv3 base access or license acceptance blocks EoMT delta weights | Record `BLOCKED:dinov3_base_access_missing` or equivalent in the selected-family status artifact. |
| Delta-weight composition is ambiguous | Implement a status gate that distinguishes missing base weights from missing local composition/converter logic. |
| Panoptic metadata is broader than current `Masks` | Start with mask/class tensor admission; defer full panoptic JSON/COCO serialization until the first checkpoint gate is honest. |
| Reference code imports heavy PyTorch stack | Keep reference execution in tools/tests only; package runtime must stay import-light. |
| Decision phase accidentally becomes implementation phase | No `src/mlx_cv/` edits in this change and no release-parity matrix row for EoMT. |

## Decision Boundary

This decision authorizes a follow-on spec for EoMT-DINOv3 real-checkpoint admission. It does not authorize downloading weights in this phase, committing weights, adding package runtime support, or claiming upstream parity.
