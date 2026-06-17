# Roadmap

This file tracks forward work only. Completed phase evidence lives under `.agent/work/`,
committed tests/fixtures, and status docs.

## Direction

`mlx-cv` is an MLX-native, inference-only computer-vision library for Apple Silicon. The
next roadmap is checkpoint-first: local/tiny fixtures are useful for architecture plumbing,
but no new model surface should be treated as credible until at least one real pretrained
checkpoint has loaded, run, and matched its upstream reference.

## Checkpoint Gate Policy

- Raw upstream checkpoints and converted weights stay outside git.
- Real checkpoint work uses an out-of-git cache, explicit license notes, and checksum or
  provenance verification.
- Small derived parity cases may be committed when they contain inputs, expected outputs,
  and taps rather than redistributable model weights.
- A skipped env-gated test or local tiny fixture is not upstream parity.
- RF-DETR Nano established the first upstream real checkpoint pass. DA3-SMALL multi-view
  now passes the required upstream-vs-MLX gate across the fixed synthetic input plus the
  real SOH image pair and robot video-derived still-frame inputs that exposed the earlier
  drift. Future gates must distinguish real passes from precise external blockers and must
  not advertise skipped tests or synthetic-only evidence as parity.

## Phase 1: Existing Checkpoint Closeout - LocateAnything And SAM 3.1 Image

- status: pending
- change: (empty when unframed)
- objective: Apply the proven real-checkpoint pattern to LocateAnything-3B and SAM 3.1 image-mode, replacing fail-stub upstream tests with real comparisons where prerequisites are available and preserving precise external blockers where they are not.
- why now: LocateAnything and SAM image-mode have higher external friction than RF-DETR: LocateAnything local safetensors are git-LFS pointer stubs with NVIDIA non-commercial weight licensing, and SAM 3.1 image-mode lacks a configured checkpoint and stable public tap capture path in this workspace.
- likely outputs: LocateAnything and SAM image upstream capture/comparison bodies; precise remediation text for LFS/license/checkpoint/tap blockers; status docs derived from `parity-status.json`; no stronger claim for any model that does not run a real checkpoint gate.
- evidence: `references/LocateAnything-3B/`, `references/sam3/`, `tests/test_la_upstream_parity.py`, `tests/test_sam3_upstream_parity.py`, `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`
- exit signal: LocateAnything and SAM 3.1 image-mode each either pass a real-checkpoint upstream parity gate or carry a precise externally actionable `BLOCKED:<reason>` while docs advertise only passing models as hardened.

## Phase 2: Depth Anything 3 Multi-View Geometry

- status: complete
- change: `2026-06-16-depth-anything-v3-multiview-checkpoint`
- objective: Extend the existing DA3 monocular path into official DA3-SMALL multi-view/camera geometry surfaces, including camera pose/intrinsics and multi-view depth/confidence outputs.
- why now: DA3 is the cleanest geometry expansion, but it needs a truthful real-checkpoint validation path rather than another local-fixture-only model surface.
- likely outputs: multi-view processor contract; camera pose/intrinsics data model; multi-view depth output path; optional pose-conditioned hooks; deterministic geometry fixtures; DA3-SMALL required upstream-vs-local checkpoint gate; demo evidence under `/tmp/mlx-cv-da3-demo/`, `/tmp/mlx-cv-da3-real-demo/`, and `/tmp/mlx-cv-da3-real-video-demo/`.
- evidence: `src/mlx_cv/models/depth_anything_v3/`, `src/mlx_cv/parity/da3_real.py`, `tests/test_da3_upstream_parity.py`, `tools/da3_demo.py`, `docs/depth-anything-v3.md`, `references/Depth-Anything-3/`, `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`
- exit signal: A fixed multi-view input and the real SOH/robot sample inputs return typed depth/camera outputs through `Result`-compatible fields, no unrelated spine churn, and the required DA3 upstream-vs-local parity command passes outside sandbox with Metal access for depth, confidence, extrinsics, intrinsics, and selected taps.

## Phase 3: SAM 3.1 Video / Object Multiplex

- status: pending
- change: (empty when unframed)
- objective: Add the deferred SAM video/tracker memory path using precise upstream naming: SAM3 Video for concept/text video detection and tracking, and Sam3Tracker for visual-prompt segmentation where applicable.
- why now: Image-mode SAM 3.1 is already present locally, but video tracking should wait until the real-checkpoint discipline exists and the SAM image-mode checkpoint outcome is understood.
- likely outputs: tracker state API; memory-bank representation; video frame processor; Object Multiplex-aware batching shape; typed tracked masks/detections with stable object IDs; deterministic short-clip fixtures; real video-checkpoint gate or precise external blocker.
- evidence: `references/sam3/`, `src/mlx_cv/models/sam3/`, `src/mlx_cv/core/types.py`
- exit signal: A short fixed video clip produces stable tracked object IDs and masks through the shared result surface, memory behavior is covered by fixtures, image-mode behavior does not regress, and the video checkpoint gate has a real pass or precise external blocker.

## Phase 4: Next Model Expansion Decision

- status: pending
- change: (empty when unframed)
- objective: Pick exactly one new model family after the checkpoint-gated existing paths are understood, then frame it as its own bounded change with a real-checkpoint admission gate.
- why now: Expansion should follow evidence that current model families can run real pretrained weights; the next family should be selected by the output pillar needed next, not by repository momentum.
- likely outputs: one selected family from DEIMv2, EoMT-DINOv3, or Sapiens2; explicit `Result` contract impact; source and license notes; smallest real checkpoint parity target; fetch/cache/checksum plan.
- evidence: `docs/BUILDING-BLOCKS.md`, `.agent/steering/REQUIREMENTS.md`, `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`
- exit signal: One model family is selected with a framed objective, explicit result-contract impact, source/license notes, and a smallest credible real-checkpoint parity gate; YOLO26 remains watchlist-only and RT-DETRv4 remains dropped unless new evidence changes the ranking.

## Deferred or Not Now

- YOLO26: watchlist only because AGPL/Enterprise licensing makes it unsuitable as a clean near-term target unless treated as external/comparative only.
- RT-DETRv4: dropped for now because it overlaps heavily with RF-DETR and DEIMv2 in the same real-time DETR lane.
