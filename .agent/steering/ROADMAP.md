# Roadmap

## Direction

Keep checkpoint claims evidence-based: architecture and local fixtures may establish readiness, but `UPSTREAM_PASSED` requires a real upstream/local numeric comparison.

## Phase 1: SAM3 Real-Parity Closeout

- status: active
- change: `2026-06-18-sam3-real-architecture-port`
- objective: Complete the gated SAM3 image and video numeric comparisons now that the faithful 1468-tensor detector, 1797-tensor video model, streaming memory, and Object-Multiplex association are implemented.
- why now: The architecture gap is closed; external parity is the remaining acceptance boundary for the active change.
- likely outputs: image and video comparison evidence, measured tolerances, corrected preprocessing/reference details if exposed, and truthful parity-matrix promotion or blocker updates.
- evidence: `.agent/work/2026-06-18-sam3-real-architecture-port/PLAN.md`, `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, `tools/sam3_upstream.py`, `tools/sam3_video_upstream.py`
- exit signal: `sam3_image` and `sam3_video` pass their required real-checkpoint gates and the active change passes verification.

## Phase 2: EoMT-DINOv3 Real Checkpoint Admission

- status: pending
- change:
- objective: Admit the selected EoMT-DINOv3 family through the smallest credible real-checkpoint gate or record a precise external, delta-composition, converter, or comparison blocker.
- why now: It is the previously verified next-family decision and adds the missing panoptic/semantic segmentation pillar after existing-family closeout.
- likely outputs: bounded SPEC/PLAN, checkpoint/base-weight admission contract, reference capture, initial local result contract, and an honest status artifact.
- evidence: `.agent/work/2026-06-17-next-model-expansion-decision/DECISION.md`, `.agent/work/2026-06-17-next-model-expansion-decision/NEXT-CHANGE-BRIEF.md`
- exit signal: EoMT-DINOv3 has a real-checkpoint reference/local gate result or a precise blocker, without expanding unrelated model families.

## Deferred or Not Now

- YOLO26: watchlist only because of AGPL/Enterprise licensing.
- RT-DETRv4: dropped while it remains redundant with the RF-DETR/DEIMv2 lane.
- Sapiens2 and DEIMv2: not part of the current two-phase roadmap.
