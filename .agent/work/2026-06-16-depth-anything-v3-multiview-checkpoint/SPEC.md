# SPEC: Depth Anything 3 Multi-View Real Checkpoint

Change: `2026-06-16-depth-anything-v3-multiview-checkpoint` - Stage: frame

## Bounded Goal

Extend the existing Depth Anything V3 monocular MLX path into DA3 any-view multi-view inference with a real Apache-licensed DA3 checkpoint resolver, conversion/load path, fixed multi-view upstream comparison, typed depth/confidence/camera outputs, and a required upstream-vs-MLX parity gate.

## Broader Intent

Move DA3 from fixture-only monocular plumbing to credible pretrained multi-view geometry support while preserving the checkpoint-first rule established by RF-DETR: no hardened model claim without a real checkpoint that loads, runs, and matches the upstream reference.

## Work Scale And Shape

- Scale: capability-sized model port extension.
- Shape: parity/checkpoint-gated architecture admission plus runtime API extension.
- Selected lenses: product, engineering, runtime.
- Stakeholder: Apple Silicon `mlx-cv` users who need local image-set geometry results, and maintainers who need truthful checkpoint status.

## Source Evidence

- Roadmap Phase 2 requires multi-view processor contracts, camera pose/intrinsics, deterministic geometry fixtures, and a real DA3 checkpoint gate.
- Local DA3 code currently implements `DepthAnythingV3Monocular`, `DA3Processor`, and tiny fixture parity only.
- Upstream DA3 main-series any-view models output depth, confidence, extrinsics, and intrinsics for image lists; `DA3-SMALL` and `DA3-BASE` are listed as Apache-2.0 any-view checkpoints.
- Hugging Face currently exposes `depth-anything/DA3-SMALL` as an Apache-2.0 0.08B any-view model for multi-view depth, pose estimation, and pose conditioning; `depth-anything/DA3-BASE` is the Apache-2.0 fallback.

## Constraints And Risks

- Primary checkpoint target is `depth-anything/DA3-SMALL`; fallback to `depth-anything/DA3-BASE` only if Small is unavailable, unusable, or mismatched against the local upstream reference.
- Raw upstream checkpoints, configs, and converted MLX weights stay outside git. The resolver must record model id, revision/provenance, license, file hashes or Xet/SHA evidence, and cache paths.
- Network/download behavior is explicit: normal CI never downloads; required phase-closing mode may download only when the user or environment opts in.
- Runtime package imports must remain MLX/NumPy/Pillow-oriented. Torch, torchvision, OpenCV, Hugging Face Hub, and upstream DA3 imports stay in tools or env-gated tests.
- Multi-view means still-image sets, not video/streaming. The output contract must remain compatible with existing single-image `Result` usage.
- Upstream DA3 may use CUDA-oriented autocast paths. The real gate should prefer a tiny fixed CPU/float32 capture when feasible and must fail loudly in required mode rather than silently counting a skipped run as parity.

## Required Outcome

R1. A DA3 checkpoint resolver/download gate resolves `config.json` and `model.safetensors` for the selected checkpoint from env/config or an out-of-git cache and prints reproducible provenance.

R2. An upstream reference capture runs a fixed three-view input through the selected DA3 checkpoint and records comparable outputs: processed image shape, depth, confidence, extrinsics, intrinsics, and selected internal taps. Three views are required so the upstream reference-view selection path is exercised.

R3. A real checkpoint architecture contract names the DA3 Small/Base inference contract: DINOv2 `vits` or `vitb`, layers `[5, 7, 9, 11]`, `alt_start=4`, `qknorm_start=4`, `rope_start=4`, `cat_token=True`, `DualDPT` output dimension 2, and camera encoder/decoder tensor groups.

`cat_token=True` doubles the effective backbone output dimension passed to the DA3 heads: DA3-Small uses `DualDPT.dim_in=768` rather than 384, and DA3-Base uses `DualDPT.dim_in=1536` rather than 768.

R4. The public MLX runtime accepts a list of still images plus optional per-view extrinsics/intrinsics, and returns view-ordered multi-view depth/confidence plus camera geometry without breaking existing monocular DA3 behavior.

R5. Converted local weights strict-load every inference tensor needed for DA3 Small/Base multi-view depth and camera output; unsupported branches such as Gaussian splatting are rejected or explicitly excluded.

R6. A required upstream-vs-MLX gate compares the selected fixed input under the real checkpoint and fails on missing checkpoint, bad provenance, skipped upstream capture, missing local load, or output drift beyond explicit measured tolerances.

R7. README, architecture docs, DA3 docs, and status records advertise only the real gate result. Tiny fixture coverage remains described as architecture plumbing, not upstream parity.

## Acceptance Criteria

- AC1: `tools/da3_checkpoint.py --model-id depth-anything/DA3-SMALL --required` resolves a local cached checkpoint or fails with a precise remediation; `--download` records provenance and never writes model weights into git.
- AC2: Normal no-checkpoint test runs skip real DA3 gates cleanly; `MLX_CV_REQUIRE_DA3_GATE=1` fails if checkpoint/config/provenance/upstream dependencies are missing.
- AC3: The upstream capture test produces fixed-input depth/confidence with shape `(V,H,W)` and camera arrays with shapes `(V,3,4)` or `(V,4,4)` and `(V,3,3)`.
- AC4: The local MLX multi-view processor/model path uses an explicit multi-view result representation, preserving view order and original-image mapping without weakening existing single-view `DepthMap` validation.
- AC5: The conversion/load test proves the selected real checkpoint strict-loads the local DA3 multi-view model with no silent tensor drops outside documented unsupported branches.
- AC6: The required parity gate compares upstream and MLX depth, confidence, extrinsics, intrinsics, and selected taps with explicit tolerances, and cannot pass from a skipped test.
- AC7: Existing DA3 monocular, DINOv2, runtime dependency guard, and full regression tests still pass.
- AC8: Documentation and status files distinguish `UPSTREAM_PASSED`, precise `BLOCKED:<reason>`, and fixture-only states for DA3.
- AC9: A visible local demo artifact is produced outside git: at minimum per-view depth visualizations plus a JSON summary of camera intrinsics/extrinsics and parity metadata.

## Scope Coverage Decisions

- Included: all roadmap Phase 2 DA3 multi-view work needed for a real checkpoint-gated result, including download/cache/provenance, upstream capture, local MLX architecture admission, conversion/load, parity, docs, and demo evidence.
- Deferred: roadmap Phase 1 LocateAnything/SAM closeout remains pending and is not modified by this change.
- Deferred: SAM 3.1 video/object multiplex, DA3 streaming, DA3 nested giant-large, DA3 metric scaling, GLB/PLY/3DGS export, and benchmark-suite evaluation.
- Anti-goal: no PyTorch/upstream DA3 dependency in normal `mlx-cv` runtime imports.
- Anti-goal: no bundled model weights or converted checkpoints in git.
- Anti-goal: no claim that DA3 is hardened from a skipped gate, tiny fixture, or local-only comparison.

## Assumptions

- `depth-anything/DA3-SMALL` remains publicly downloadable and Apache-2.0; if this changes, execution must fall back to `depth-anything/DA3-BASE` or return to planning with a precise reason.
- A fixed low-resolution three-view capture is small enough to run locally for upstream reference and MLX parity on this machine.
