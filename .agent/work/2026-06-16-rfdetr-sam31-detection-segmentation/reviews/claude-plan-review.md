# Claude Code Plan Review

Reviewer: Claude Code Opus 4.8  
Session: `dc4882f7-d81d-4471-8d92-1125f11edd33`  
Mode: read-only plan mode with `Read`, `Glob`, and `Grep` tools  
Effort: max

## Verdict

`APPROVE_WITH_RISK`

Claude's first pass approved the plan with required risk corrections. The plan and design were updated, then sent back to the same session. Claude confirmed all required corrections were addressed. After an independent pass, two clarification edits were made for conditional `ExemplarPrompt` handling and SAM score ownership; Claude's final delta review preserved the `APPROVE_WITH_RISK` verdict and found no new inconsistency.

## Accepted Corrections

- SAM 3.1 tokenizer scope now requires committed real or reduced CLIP-style BPE assets plus a canonical string-to-token-id fixture.
- SAM 3.1 prompt scope is narrowed to text and PCS box-exemplar grounding; SAM1-style point/click prompting is deferred unless a dedicated fixture is added.
- Runtime dependency guards now include `triton`, CUDA-only packages, and tokenizer helper imports in addition to `torch` and `transformers`.
- Default execution topology is serial; RF-DETR and SAM 3.1 are not treated as parallel-safe without explicit shared-file partitioning.
- Slice 1 now only fills missing result/mask serialization and validation gaps instead of claiming prompt dataclasses need to be created.
- SAM 3.1 grounding boxes/scores are carried through `Result.detections` when emitted.
- SAM 3.1 fixture minting may use submethod-level taps when top-level forward is not stable enough.
- `ExemplarPrompt` is accepted only when mapped to the fixture-backed image-mode box-exemplar path; otherwise it fails clearly and remains deferred.
- SAM mask/object scores stay on paired `Result.detections.scores` unless a concrete parity blocker proves `Masks` needs a minimal typed extension.

## Remaining Execution Guidance

- Use a pure ASCII canonical category string for the tokenizer fixture unless Unicode parity is explicitly solved without adding runtime dependencies.
- Keep `BoxPrompt` and conditional cross-image `ExemplarPrompt` semantics distinct.
- Mint RF-DETR deformable attention from the pure-PyTorch reference path with `align_corners=False` and zero padding semantics.
- For SAM 3.1 parity, prefer `forward_image`, `forward_text`, and `forward_grounding` submethod taps over the top-level dataclass-heavy forward path when needed.
