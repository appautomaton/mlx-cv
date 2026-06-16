# Claude Final Delta Review Prompt

Continue session `dc4882f7-d81d-4471-8d92-1125f11edd33`.

After your `APPROVE_WITH_RISK` re-review, I independently accepted two of your residual risks as worthwhile clarification edits:

1. `DESIGN.md` and `PLAN.md` now say `ExemplarPrompt` is accepted only when it maps to the fixture-backed image-mode box-exemplar grounding path; otherwise it fails clearly and remains deferred.
2. `PLAN.md` now says SAM mask/object scores should stay on paired `Result.detections.scores` when grounding boxes are emitted unless a concrete parity blocker proves `Masks` needs a minimal typed extension.

Please re-read the revised:

- `.agent/work/2026-06-16-rfdetr-sam31-detection-segmentation/DESIGN.md`
- `.agent/work/2026-06-16-rfdetr-sam31-detection-segmentation/PLAN.md`

Question: do these clarification edits preserve the verdict, or did they introduce a plan inconsistency?

Output:

- Verdict: `APPROVE`, `APPROVE_WITH_RISK`, or `NEEDS_CORRECTION`.
- If `NEEDS_CORRECTION`, provide reason and concrete guidance.
- If not, keep the response short and list any remaining execution-only risks.
