# Roadmap

This file tracks forward work only. Completed phase evidence lives under `.agent/work/`,
committed tests/fixtures, docs, and status artifacts.

## Direction

`mlx-cv` remains checkpoint-first: local/tiny fixtures are useful for architecture
plumbing, but no model surface should be described as real upstream parity until at
least one real pretrained checkpoint has loaded, run, and matched its upstream
reference or recorded a precise external blocker.

## Checkpoint Gate Policy

- Raw upstream checkpoints and converted weights stay outside git.
- Real checkpoint work uses an out-of-git cache, explicit license notes, and checksum or
  provenance verification.
- Small derived parity cases may be committed when they contain inputs, expected outputs,
  and taps rather than redistributable model weights.
- A skipped env-gated test or local deterministic fixture is not upstream parity.
- RF-DETR Nano and DA3-SMALL multi-view have passed real upstream-vs-MLX gates.
- LocateAnything full-checkpoint parity, SAM 3.1 image parity, and SAM 3.1 video
  parity still require external checkpoint/tap work before stronger claims are allowed.

## Phase 1: Existing Checkpoint Closeout

- status: done
- change: `2026-06-17-existing-checkpoint-closeout`
- objective: Resolve the remaining existing-family checkpoint blockers before adding another model family: LocateAnything-3B full-checkpoint parity, SAM 3.1 image checkpoint/tap parity, and the status wording that derives from those blockers.
- why now: The roadmap should not expand to a new family while existing public surfaces still carry externally actionable checkpoint blockers.
- likely outputs: usable out-of-git LocateAnything checkpoint or precise LFS/license blocker; SAM 3.1 image checkpoint/tap comparison body or precise blocker; updated per-model status artifacts; docs that advertise only passing gates as hardened.
- evidence: `references/LocateAnything-3B/`, `references/sam3/`, `tests/test_la_upstream_parity.py`, `tests/test_sam3_upstream_parity.py`, `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`
- exit signal: LocateAnything and SAM 3.1 image-mode either pass real-checkpoint upstream parity gates or carry precise `BLOCKED:<reason>` records with docs that make no stronger claim.

## Phase 2: SAM 3.1 Video Real Checkpoint Admission

- status: done
- change: `2026-06-17-sam3-video-real-checkpoint-admission`
- objective: Move SAM 3.1 video/Object Multiplex from local deterministic contract coverage to a real checkpoint admission attempt.
- why now: Local video/tracker/Object Multiplex plumbing is verified, but `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json` still records `BLOCKED:MLX_CV_SAM3_VIDEO_CHECKPOINT is unset`.
- likely outputs: identified official SAM3 video/Object Multiplex checkpoint source; out-of-git cache path; license/provenance/checksum record; configured `MLX_CV_SAM3_VIDEO_CHECKPOINT` / config/model envs; required gate result; exact blocker if upstream-vs-local numeric comparison cannot be completed.
- evidence: `src/mlx_cv/models/sam3/video.py`, `src/mlx_cv/core/types.py`, `src/mlx_cv/core/tracking.py`, `tools/sam3_video_upstream.py`, `tests/test_sam3_video_*`, `tests/test_sam3_object_multiplex.py`, `docs/sam3-video.md`, `.agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json`, `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`
- exit signal: The SAM3 video gate either passes against a usable real checkpoint or records a precise blocker that names the missing checkpoint, config, reference runtime, or local comparison component.

## Phase 3: Next Model Expansion Decision

- status: active
- change: `2026-06-17-next-model-expansion-decision`
- objective: Pick exactly one new model family after the checkpoint-gated existing paths are understood, then frame it as its own bounded change with a real-checkpoint admission gate.
- why now: Expansion should follow evidence that current model families can run or precisely block real pretrained weights; the next family should be selected by output-pillar need, not by repository momentum.
- likely outputs: one selected family from DEIMv2, EoMT-DINOv3, or Sapiens2; explicit `Result` contract impact; source and license notes; smallest real checkpoint parity target; fetch/cache/checksum plan.
- evidence: `docs/BUILDING-BLOCKS.md`, `.agent/steering/REQUIREMENTS.md`, `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`
- exit signal: One model family is selected with a framed objective, explicit result-contract impact, source/license notes, and a smallest credible real-checkpoint parity gate; YOLO26 remains watchlist-only and RT-DETRv4 remains dropped unless new evidence changes the ranking.

## Deferred or Not Now

- YOLO26: watchlist only because AGPL/Enterprise licensing makes it unsuitable as a clean near-term target unless treated as external/comparative only.
- RT-DETRv4: dropped for now because it overlaps heavily with RF-DETR and DEIMv2 in the same real-time DETR lane.
