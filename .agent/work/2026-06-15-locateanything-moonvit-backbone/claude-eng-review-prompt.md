# Claude Code Engineering Review Request

You are a direct technical engineering reviewer for an Automaton `auto-eng-review`.

Model requested by orchestrator: Opus class with maximum thinking/effort.
Permissions requested by orchestrator: read-only. Do not edit files. Do not execute the plan.

Do not read `.claude/`, `.codex/`, `.opencode/`, or `.agent/.automaton/`; they are harness machinery for another agent and waste context.

## Objective

Review the plan for execution safety before implementation begins.

Primary artifacts:
- `.agent/work/2026-06-15-locateanything-moonvit-backbone/SPEC.md`
- `.agent/work/2026-06-15-locateanything-moonvit-backbone/DESIGN.md`
- `.agent/work/2026-06-15-locateanything-moonvit-backbone/PLAN.md`

High-value source/reference files to inspect as needed:
- `src/mlx_cv/backbones/vision/moonvit/__init__.py`
- `src/mlx_cv/models/locateanything/config.py`
- `src/mlx_cv/backbones/vision/vit.py`
- `src/mlx_cv/backbones/layers/position.py`
- `src/mlx_cv/hub/convert.py`
- `src/mlx_cv/parity/fixtures.py`
- `src/mlx_cv/parity/harness.py`
- `references/LocateAnything-3B/modeling_vit.py`
- `references/LocateAnything-3B/image_processing_locateanything.py`
- `references/mlx-vlm/mlx_vlm/models/locateanything/vision.py`
- `references/mlx-vlm/mlx_vlm/models/kernels.py`
- `tests/test_qwen2_integration_guards.py`
- `tests/test_dinov2_convert.py`
- `tests/test_dinov3_parity.py`

## Review Standard

Focus on execution safety: architecture fit, data-flow clarity, edge cases, test strategy, rollback/reversibility, and dependency risk. Do not reopen product scope. Do not broaden the plan beyond the approved MoonViT backbone slice.

The riskiest areas are likely:
- packed NCHW patch input vs mlx-vlm NHWC internal convention
- bicubic interpolation parity
- `PytorchGELUTanh` vs MLX GELU behavior
- boolean block attention mask semantics in MLX SDPA
- conversion path choices for standalone MoonViT vs later full LocateAnything model
- fixture minting in an out-of-band torch/transformers environment

## Output Shape

Return exactly these sections:

Verdict: one of `approved`, `approved_with_risks`, or `needs_correction`

Critical path: 2-4 bullets naming the execution path and riskiest slice.

Findings: ranked bullets. Each finding must name the plan slice and the concrete file/function/behavior behind it. Mark each as `blocking`, `risk`, or `follow-up`.

Corrections if needed: if verdict is `needs_correction`, give exact plan/design changes to make. If verdict is not `needs_correction`, say `None required` and optionally list execution cautions.

Review template suggestion: provide the exact 5-field markdown block the orchestrator should append to PLAN.md:

```markdown
## Review: Engineering

- Verdict: <approved|approved_with_risks|needs_correction>
- Strength: <one sentence>
- Concern: <one sentence>
- Action: <one sentence>
- Verified: <what was checked, or "pending">
```
