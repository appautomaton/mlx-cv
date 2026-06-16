# Roadmap

This file tracks only forward work. Completed phase history is not tracked here; durable evidence for
completed work lives under `.agent/work/`, committed tests/fixtures, and status docs.

## Direction

`mlx-cv` is an MLX-native, inference-only computer-vision library for Apple Silicon. The next roadmap
keeps the existing spine honest by hardening current model claims first, then expands only where the
model adds a distinct output pillar or exercises a reusable contract.

## Source-Grounded Model Anchors

- **DINOv3** - arXiv submitted 2025-08-13; Hugging Face backbones available 2025-08-14 per the official repo. This remains the dense-feature foundation anchor.
- **LocateAnything-3B** - `nvidia/LocateAnything-3B`; paper "LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding"; arXiv submitted 2026-05-26. Strong grounding/parity-hardening anchor with a non-commercial model-license caveat.
- **RF-DETR** - "RF-DETR: Neural Architecture Search for Real-Time Detection Transformers"; Roboflow model page release 2025-03-20; arXiv submitted 2025-11-12. Already implemented locally, so the roadmap treats it as a hardening target rather than a new model-admission precedent.
- **Depth Anything 3 / DA3** - "Depth Anything 3: Recovering the Visual Space from Any Views"; arXiv submitted 2025-11-13; official repo released paper, project page, code, and models on 2025-11-14.
- **SAM 3** - "SAM 3: Segment Anything with Concepts"; Meta publication 2025-11-19; arXiv submitted 2025-11-20.
- **SAM 3.1 / Object Multiplex** - SAM 3.1 release notes dated 2026-03-27. This is a video/tracker efficiency update to SAM 3, not a separate model family.
- **DEIMv2** - "Real-Time Object Detection Meets DINOv3"; arXiv submitted 2025-09-25; official repo released the DEIMv2 series on 2025-09-26.
- **EoMT-DINOv3** - Encoder-only Mask Transformer with DINOv3 support; base EoMT paper "Your ViT is Secretly an Image Segmentation Model" submitted 2025-03-24, with DINOv3 support now advertised by the official repo/model cards.
- **Sapiens2** - "Sapiens2"; OpenReview published 2026-01-26; arXiv submitted 2026-04-23; official repo initial model release 2026-04-24.
- **YOLO26** - Ultralytics YOLO26; official launch 2026-01-14; arXiv submitted 2026-06-02. Watchlist only because AGPL/Enterprise licensing is a poor fit for the clean permissive roadmap.
- **RT-DETRv4** - "RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models"; arXiv submitted 2025-10-29; full code/config/checkpoint release 2025-11-17. Dropped unless new evidence changes the ranking.

## Phase 1: Release Parity Hardening

- **Objective:** Harden the existing local LocateAnything, RF-DETR, and SAM 3.1 image-mode paths from local/tiny fixture confidence to stronger upstream-reference or full-checkpoint parity where the upstream runtime and weights are available.
- **Why now:** These are already runnable local model paths, but current status text still distinguishes local integration/tiny-oracle gates from full upstream parity. Closing that gap is required before stronger release claims.
- **Likely outputs:** reference-run capture notes; checkpoint/config conversion audits; env-gated full-checkpoint parity commands; upstream/reference fixtures where stable; tolerance policy; docs/status wording that distinguishes true upstream parity from local fixture gates.
- **Out of scope:** SAM 3.1 video/tracker/Object Multiplex, DA3 multi-view, and new model families.
- **Exit signal:** LocateAnything, RF-DETR detector, and SAM 3.1 image-mode each have truthful upstream-reference/full-checkpoint parity evidence or an explicit recorded blocker; local fixture gates still pass and docs do not overstate skipped or blocked parity.

## Phase 2: Depth Anything 3 Multi-View Geometry

- **Objective:** Extend the existing DA3 monocular path into the official multi-view/camera geometry surface where it is contractually useful: camera pose/intrinsics, multi-view depth consistency, and related dense geometry outputs.
- **Why now:** DA3 is the cleanest next geometry expansion. It exercises existing dense-output contracts while adding camera metadata and multi-view shape pressure before heavier video tracking.
- **Likely outputs:** multi-view processor contract; camera pose/intrinsics data model; multi-view depth output path; optional pose-conditioned depth hooks; fixture coverage for deterministic geometry shapes and original-image mapping.
- **Out of scope:** 3D Gaussian rendering as a product feature unless the spec explicitly chooses it; monocular depth regressions; non-MLX backends.
- **Exit signal:** A fixed multi-view input returns typed depth/camera outputs through `Result`-compatible fields with deterministic fixture coverage and no unrelated spine churn.

## Phase 3: SAM 3.1 Video / Object Multiplex

- **Objective:** Add the deferred SAM video/tracker memory path using precise upstream naming: SAM3 Video for concept/text video detection+tracking, and Sam3Tracker for visual prompt / SAM1/SAM2-style segmentation where applicable.
- **Why now:** Image-mode SAM 3.1 is already present locally; video tracking is the remaining high-value capability and the main consumer for the tracker/memory-bank contract.
- **Likely outputs:** tracker state API; memory-bank representation; video frame processor; Object Multiplex-aware batching shape; typed tracked masks/detections with stable object IDs; fixture coverage for deterministic short clips.
- **Out of scope:** Replacing static segmentation with EoMT; training/fine-tuning; product claims beyond the committed video fixture gate.
- **Exit signal:** A short fixed video clip produces stable tracked object IDs and masks through the shared result surface, with memory behavior covered by fixtures and no image-mode regression.

## Phase 4: Next Model Expansion Decision

- **Objective:** Pick exactly one new model family after Phases 1-3 based on the output pillar needed next, then frame it as its own bounded change.
- **Decision options:**
  - **DEIMv2** when DINOv3-native real-time detection is the priority and RF-DETR hardening is already credible.
  - **EoMT-DINOv3** when static closed-set semantic/instance/panoptic segmentation becomes a first-class need.
  - **Sapiens2** when human-centric dense outputs become a first-class pillar; expect `Result` additions for pose, normals, pointmap, matting, and possibly albedo.
- **Out of scope:** Porting multiple expansion families in one phase; adopting YOLO26 as a near-term target; reviving RT-DETRv4 without new evidence.
- **Exit signal:** One model family is selected with a framed objective, explicit result-contract impact, source/license notes, and a smallest credible parity fixture.

## Watchlist / Dropped

- **YOLO26** - Watchlist only. It is real and broad, but AGPL/Enterprise licensing makes it unsuitable as a clean near-term target unless treated as external/comparative only.
- **RT-DETRv4** - Dropped for now. It is real and permissively licensed, but overlaps heavily with RF-DETR and DEIMv2 in the same real-time DETR lane.
