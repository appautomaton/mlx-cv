# Antigravity Gemini Critical Recheck

[Antigravity Capsule] Goal: Fresh critical DA3 SPEC/DESIGN/PLAN review | SID: d9f83495-ca19-4f8b-b67f-95547d33832a | Model: Gemini 3.1 Pro (High) | Sandbox: on | Files: corrected SPEC.md, DESIGN.md, PLAN.md, ROADMAP.md, DA3 source/tests/reference | Last: APPROVED | Next: auto-execute when authorized

## Verdict

APPROVED

## Blocking Findings

None.

## Non-Blocking Risks

- Reference-view selection and reordering must actually be exercised. If the fixed three-view input selects the first view as reference, reorder/restore logic may not be proven.
- `cat_token=True` split normalization is easy to implement incorrectly. Upstream normalizes only the second half of the concatenated features.
- CPU/float32 upstream capture may be slow if input resolution is too large.

## Execution Advice

- Use a low-resolution fixed three-view capture, for example 256x256 if it preserves the required upstream behavior.
- Choose fixed views that force a non-first reference view, proving local reorder/restore logic.
- Prototype alternating `(B*V,N,C)` local and `(B,V*N,C)` global attention early in Slice 5.
- Enable `qk_norm` conditionally per block at `qknorm_start=4`.
- Keep DualDPT main and auxiliary fusion paths independent; do not share weights or reuse the old `DPTHead`.
- Document DA3 camera geometry as final `w2c` extrinsics.
- Export taps around `alt_start`, especially before and after camera-token injection.
