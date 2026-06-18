# Antigravity Gemini Plan Review

[Antigravity Capsule] Goal: Evaluate DA3 multi-view SPEC/DESIGN/PLAN | SID: ebf45ccd-e4c5-4767-812d-7b00b1b6b266 | Model: Gemini 3.1 Pro (High) | Sandbox: on | Files: SPEC.md, DESIGN.md, PLAN.md, ROADMAP.md, DA3 source/tests/reference | Last: APPROVED | Next: execute when authorized

## Verdict

APPROVED

## Key Findings

- Roadmap and intent alignment: the spec maps to roadmap Phase 2 and moves DA3 from monocular fixture coverage to real-checkpoint gated multi-view parity.
- Architecture and codebase plausibility: the plan's DA3 Small contract matches upstream `da3-small.yaml`: DINOv2 `vits`, out layers `[5, 7, 9, 11]`, `alt_start=4`, `qknorm_start=4`, `rope_start=4`, `cat_token=True`, `DualDPT`, `CameraEnc`, and `CameraDec`.
- Risk mitigation: the plan accounts for multi-view `cat_token`, camera pose convention, unsupported Gaussian/sky/metric branches, and checkpoint gate semantics.
- Gate quality: `MLX_CV_REQUIRE_DA3_GATE=1` is positioned correctly so normal CI can skip while required parity cannot pass from a skipped gate.

## Required Corrections

None.

## Execution Advice

- Capture taps immediately before and after upstream `ref_view_strategy` / `cat_token` handling; this is the highest-risk layout drift point.
- Document and enforce final DA3 extrinsics as `w2c`, because upstream decodes `c2w` then returns `affine_inverse(c2w)`.
- Use fixed same-size multi-view inputs for parity to avoid mixed-size output ambiguity.
- Omit sky-mask handling for Small/Base unless a selected checkpoint includes a sky head; upstream safely bypasses it when absent.
- Run upstream capture in an isolated subprocess, or evaluate dependency guards in a pristine interpreter, to avoid PyTorch/upstream import leakage.
