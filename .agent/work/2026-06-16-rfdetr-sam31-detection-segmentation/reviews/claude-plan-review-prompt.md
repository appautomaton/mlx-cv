# Claude Plan Review Prompt

You are an independent engineering reviewer for the local repo at `/Users/ac/dev/ai/mlx-cv`.

Review the current Automaton plan for the active change:

- SPEC: `.agent/work/2026-06-16-rfdetr-sam31-detection-segmentation/SPEC.md`
- DESIGN: `.agent/work/2026-06-16-rfdetr-sam31-detection-segmentation/DESIGN.md`
- PLAN: `.agent/work/2026-06-16-rfdetr-sam31-detection-segmentation/PLAN.md`
- Roadmap: `.agent/steering/ROADMAP.md`

Task:

1. Independently evaluate whether the PLAN is safe, coherent, and complete for executing the SPEC.
2. Read the repo and reference files as needed. Do not edit files and do not execute the plan.
3. Check that the slices are ordered correctly, verifiable, not overbroad, and do not hide unresolved scope decisions.
4. Check that RF-DETR and SAM 3.1 are scoped correctly against the roadmap and SPEC, especially:
   - RF-DETR detection only, not RF-DETR segmentation variants.
   - SAM 3.1 image-mode segmentation, not video/tracking.
   - SAM 3.1 text prompt path is not silently reduced to geometry-only prompts.
   - Reference parity is required before shipped-model claims.
   - Runtime package must not gain hard torch/transformers/triton/CUDA deps.
5. If the PLAN needs correction, provide:
   - the reason,
   - the exact artifact/file/section evidence,
   - guidance on how to address it,
   - whether the issue blocks execution or can be carried as an explicit risk.

Output format:

```
VERDICT: APPROVE | APPROVE_WITH_RISK | NEEDS_CORRECTION

FINDINGS:
- [severity] [file:line or section] Finding, reason, and guidance.

PLAN_CORRECTIONS:
- Required correction text or `none`.

EXECUTION_GUIDANCE:
- Concrete guidance for the executor.
```

Do not summarize the whole plan. Lead with actionable findings.
