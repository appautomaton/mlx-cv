# Antigravity Claude Opus Plan Review

[Antigravity Capsule] Goal: Sharp independent DA3 SPEC/DESIGN/PLAN review | SID: cfc6ceb8-436f-40ce-9638-4b2adf382328 | Model: Claude Opus 4.6 (Thinking) | Sandbox: on | Files: SPEC.md, DESIGN.md, PLAN.md, ROADMAP.md, DA3 source/tests/reference | Last: APPROVED_WITH_RISKS | Next: apply plan corrections before execute

## Verdict

APPROVED_WITH_RISKS

Claude disagreed with the prior Gemini approval as too optimistic. It found no strictly blocking issue, but identified material under-specifications that would likely cause execution stalls or rework if left implicit.

## Key Findings

- DA3 DINOv2 is a new any-view ViT variant, not a config swap over the existing local `ViTBackbone`. Missing or under-specified primitives include per-block `qk_norm`, DA3-specific 2D RoPE, alternating local/global attention, camera-token injection, reference-view selection, `cat_token` concatenation, and split normalization.
- DualDPT is a new module, not a `DPTHead` extension. It has separate main/aux fusion chains, UV positional embeddings, a 7-channel auxiliary ray output, and multi-view feature handling.
- The current `DepthMap`/`Result` shape is single-view; the plan needs a concrete multi-view result strategy rather than a generic `Result-compatible` phrase.
- `cat_token=True` doubles DA3 Small effective head input dimension to 768 and DA3 Base to 1536; this needs to be explicit in the spec/design/plan.
- Exactly three views should be used for parity if the plan intends to exercise upstream reference-view selection, because two views skip that path.
- Upstream capture should avoid mixed-precision drift by forcing CPU/float32 or otherwise recording precision behavior explicitly.
- Camera encoder/decoder requires quaternion/FOV pose utilities and matrix inversion conventions that must be named in the camera-head slice.

## Required Plan Corrections

- Add explicit DA3 any-view ViT primitives to `DESIGN.md` and Slice 5.
- Specify the multi-view `Result` approach in Slice 4.
- Add `cat_token` doubled-dimension note to `SPEC.md`.
- Use exactly three views for the parity fixture.
- Add camera utility dependencies to the architecture contract and camera-head slice.
- Ensure DA3 status is created or updated in `parity-status.json`.

## Execution Advice

- Prototype upstream `process_attention` layout first: local `(B*S,N,C)` and global `(B,S*N,C)`.
- Add optional `qk_norm` to shared attention/block primitives without breaking DINOv2/DINOv3 users.
- Treat DA3 RoPE as distinct from existing DINOv3 RoPE.
- Capture taps before/after reference-view selection, camera-token injection, output-layer `cat_token`, DualDPT main/aux logits, CameraDec 9D pose encoding, and final w2c extrinsics.
- Port quaternion math exactly; upstream uses scalar-last quaternion order.
