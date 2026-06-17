# PLAN: Next Model Expansion Decision

Change: `2026-06-17-next-model-expansion-decision` - Stage: plan - Spec: `SPEC.md` - Design: `DESIGN.md`

## Goal

Execute roadmap Phase 3 by selecting exactly one next model family from DEIMv2, EoMT-DINOv3, or Sapiens2 and preparing the follow-on real-checkpoint admission brief.

## Architecture Approach

See `DESIGN.md`. This is a decision phase: execution produces candidate evidence, a scored decision, and a next-change brief. It must not implement the selected model or mutate package runtime.

## Requirement Traceability

| SPEC criteria | Satisfying slices |
|---|---|
| AC1, AC2, AC7 | Slice 1 |
| AC3, AC7 | Slice 2 |
| AC4 | Slice 3 |
| AC5, AC6 | Slice 4 |
| AC8 | Slice 5 |

## Execution Routing And Topology

- Default execution: direct, serial, continue through all slices when verification passes.
- Subagent recommended: Slices 1 through 4, because they require evidence cross-checking across candidate reference trees and current public sources.
- Checkpoints: none. The approved phase authorizes the agent to choose one family by evidence; no human decision checkpoint is needed unless source evidence contradicts the candidate set.
- Parallel-safe groups: none. The decision artifacts depend on each other and should be integrated serially.
- External access: read-only browsing or source verification is allowed when needed; checkpoint downloads require execution-time approval and are not expected for this phase.

## Ordered Slice Sequence

### Slice 1: Candidate Source, License, And Checkpoint Inventory

**Objective:** Build the factual candidate inventory for DEIMv2, EoMT-DINOv3, and Sapiens2.

**Acceptance criteria:**
- `CANDIDATE-MATRIX.md` exists and has one section each for DEIMv2, EoMT-DINOv3, and Sapiens2.
- Each candidate section records current official/source URL, local reference path, release/status evidence, license/access notes, checkpoint/model IDs, likely cache layout, and reference runtime dependencies.
- The matrix distinguishes repo-local evidence from current public source evidence.
- No candidate outside the approved three is added.

**Verification:** `test -f .agent/work/2026-06-17-next-model-expansion-decision/CANDIDATE-MATRIX.md && rg -n "DEIMv2|EoMT-DINOv3|Sapiens2|checkpoint|license|cache|references/" .agent/work/2026-06-17-next-model-expansion-decision/CANDIDATE-MATRIX.md`

**Execution:** subagent recommended

**Touches:** `.agent/work/2026-06-17-next-model-expansion-decision/CANDIDATE-MATRIX.md`, local reference/docs reads only.

**Produces:** Candidate inventory grounded in source and local reference evidence.

**Status:** complete
**Evidence:** added `CANDIDATE-MATRIX.md`; `test -f .../CANDIDATE-MATRIX.md && rg -n "DEIMv2|EoMT-DINOv3|Sapiens2|checkpoint|license|cache|references/" .../CANDIDATE-MATRIX.md` passed; checkpoint IDs are tagged as `confirmed-against-current-source` or `local-reference-only`.
**Risks / next:** EoMT and Sapiens2 have access/license coupling that must stay explicit in the first-gate analysis.

### Slice 2: Checkpoint-First Gate Feasibility

**Objective:** Identify the smallest credible real-checkpoint admission or parity gate for each candidate.

**Acceptance criteria:**
- Each candidate has a named first-gate target or a precise explanation why no credible first gate exists yet.
- Gate targets include model ID/checkpoint file, config/source metadata, env/cache shape, reference entry point, expected output/taps, and likely blocker taxonomy.
- Download/auth/license blockers are separated from local converter/reference-comparison blockers.
- The existing release parity matrix is not expanded.

**Verification:** `rg -n "first gate|model ID|checkpoint|env|cache|blocker|DEIMv2|EoMT-DINOv3|Sapiens2" .agent/work/2026-06-17-next-model-expansion-decision/CANDIDATE-MATRIX.md && python -c "import json; s=json.load(open('.agent/work/2026-06-16-release-parity-hardening/parity-status.json')); assert set(s['models']) == {'da3_multiview','locateanything','rfdetr','sam3_image'}"`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `.agent/work/2026-06-17-next-model-expansion-decision/CANDIDATE-MATRIX.md`.

**Produces:** Candidate-specific first-gate hypotheses.

**Status:** complete
**Evidence:** extended `CANDIDATE-MATRIX.md` with first-gate targets for DEIMv2, EoMT-DINOv3, and Sapiens2; `rg -n "first gate|model ID|checkpoint|env|cache|blocker|DEIMv2|EoMT-DINOv3|Sapiens2" .../CANDIDATE-MATRIX.md` passed; release-matrix bound check printed `release matrix bounded`.
**Risks / next:** no release row was added; selected-family implementation must create its own status artifact first.

### Slice 3: Spine And Result Contract Impact

**Objective:** Compare how each candidate affects the existing `mlx-cv` spine and typed result surface.

**Acceptance criteria:**
- `SPINE-IMPACT.md` exists and covers each candidate.
- Each candidate records `Result` impact, processor/transform impact, backbone/neck/head reuse, missing ops, converter/load complexity, local fixture shape, and risk to import-light runtime boundaries.
- The analysis is grounded in existing `src/mlx_cv/` directories and candidate reference entry points.
- The artifact names any needed future `Result` widening as future-scope, not as part of this decision phase.

**Verification:** `test -f .agent/work/2026-06-17-next-model-expansion-decision/SPINE-IMPACT.md && rg -n "Result|processor|transform|backbone|neck|head|ops|runtime|DEIMv2|EoMT-DINOv3|Sapiens2" .agent/work/2026-06-17-next-model-expansion-decision/SPINE-IMPACT.md`

**Execution:** subagent recommended

**Depends on:** Slice 2

**Touches:** `.agent/work/2026-06-17-next-model-expansion-decision/SPINE-IMPACT.md`.

**Produces:** Spine impact and implementation-risk comparison.

**Status:** complete
**Evidence:** added `SPINE-IMPACT.md`; `test -f .../SPINE-IMPACT.md && rg -n "Result|processor|transform|backbone|neck|head|ops|runtime|DEIMv2|EoMT-DINOv3|Sapiens2" .../SPINE-IMPACT.md` passed; future `Result` widening is explicitly scoped outside this decision change.
**Risks / next:** scoring must account for EoMT's DINOv3 delta-weight blocker and Sapiens2's broader license/result-surface risk.

### Slice 4: Scored Decision And Follow-On Brief

**Objective:** Select exactly one model family and prepare the follow-on real-checkpoint admission brief.

**Acceptance criteria:**
- `DECISION.md` exists and applies the `DESIGN.md` scoring rubric with numeric scores and evidence-backed notes.
- Exactly one family is marked `selected`; the other two are marked `deferred` with reasons.
- `NEXT-CHANGE-BRIEF.md` exists and names the selected family's next objective, likely slug, model/checkpoint source, env/cache variables, first gate command shape, expected status artifact, blocker taxonomy, and anti-goals.
- The decision does not claim the selected model is implemented, locally supported, or upstream-passed.

**Verification:** `test -f .agent/work/2026-06-17-next-model-expansion-decision/DECISION.md && test -f .agent/work/2026-06-17-next-model-expansion-decision/NEXT-CHANGE-BRIEF.md && rg -n "selected|deferred|score|DEIMv2|EoMT-DINOv3|Sapiens2" .agent/work/2026-06-17-next-model-expansion-decision/DECISION.md && rg -n "objective|slug|checkpoint|env|cache|gate|blocker|anti-goals" .agent/work/2026-06-17-next-model-expansion-decision/NEXT-CHANGE-BRIEF.md && python -c "from pathlib import Path; import re; t=Path('.agent/work/2026-06-17-next-model-expansion-decision/DECISION.md').read_text(); assert t.count('Status: selected') == 1; assert t.count('Status: deferred') == 2; rows={line.split('|')[1].strip(): line for line in t.splitlines() if line.startswith('| ') and 'Status:' in line}; assert set(rows) == {'DEIMv2','EoMT-DINOv3','Sapiens2'}; assert all(len(re.findall(r'\b\d+\b', row)) >= 7 for row in rows.values()); print('decision invariant ok')"`

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `.agent/work/2026-06-17-next-model-expansion-decision/DECISION.md`, `.agent/work/2026-06-17-next-model-expansion-decision/NEXT-CHANGE-BRIEF.md`.

**Produces:** One selected family and a future implementation brief.

**Status:** complete
**Evidence:** added `DECISION.md` and `NEXT-CHANGE-BRIEF.md`; artifact/coverage greps passed; exact invariant command printed `decision invariant ok` after asserting one `Status: selected`, two `Status: deferred`, and numeric scores for DEIMv2, EoMT-DINOv3, and Sapiens2.
**Risks / next:** EoMT-DINOv3 is selected only for a future admission gate; the decision artifact explicitly makes no local-support or upstream-parity claim.

### Slice 5: Roadmap, Hygiene, And Verification

**Objective:** Publish the Phase 3 decision state and verify the repo stayed decision-only.

**Acceptance criteria:**
- Roadmap Phase 3 remains active until verify and points to this change.
- Steering docs or roadmap are updated only if needed to link the selected follow-on brief; no implementation phase is silently marked active.
- No `src/mlx_cv/` file changes are present unless a plan correction explicitly justifies them.
- Runtime dependency guards pass, the release parity matrix remains bounded, and `git diff --check` passes.

**Verification:** Run `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_runtime_dependency_guards.py`, then `python -c "import json; s=json.load(open('.agent/work/2026-06-16-release-parity-hardening/parity-status.json')); assert set(s['models']) == {'da3_multiview','locateanything','rfdetr','sam3_image'}"`, then `git diff --check`.

**Depends on:** Slices 1 through 4

**Touches:** `.agent/steering/ROADMAP.md`, `.agent/work/2026-06-17-next-model-expansion-decision/PLAN.md`.

**Produces:** Verified decision-phase artifact set.

**Status:** pending

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Decision artifacts | `test -f .agent/work/2026-06-17-next-model-expansion-decision/CANDIDATE-MATRIX.md && test -f .agent/work/2026-06-17-next-model-expansion-decision/SPINE-IMPACT.md && test -f .agent/work/2026-06-17-next-model-expansion-decision/DECISION.md && test -f .agent/work/2026-06-17-next-model-expansion-decision/NEXT-CHANGE-BRIEF.md` |
| Candidate coverage | `rg -n "DEIMv2|EoMT-DINOv3|Sapiens2" .agent/work/2026-06-17-next-model-expansion-decision/CANDIDATE-MATRIX.md .agent/work/2026-06-17-next-model-expansion-decision/SPINE-IMPACT.md .agent/work/2026-06-17-next-model-expansion-decision/DECISION.md` |
| Decision result | `rg -n "selected|deferred|score|objective|checkpoint|gate|blocker" .agent/work/2026-06-17-next-model-expansion-decision/DECISION.md .agent/work/2026-06-17-next-model-expansion-decision/NEXT-CHANGE-BRIEF.md && python -c "from pathlib import Path; import re; t=Path('.agent/work/2026-06-17-next-model-expansion-decision/DECISION.md').read_text(); assert t.count('Status: selected') == 1; assert t.count('Status: deferred') == 2; rows={line.split('|')[1].strip(): line for line in t.splitlines() if line.startswith('| ') and 'Status:' in line}; assert set(rows) == {'DEIMv2','EoMT-DINOv3','Sapiens2'}; assert all(len(re.findall(r'\b\d+\b', row)) >= 7 for row in rows.values()); print('decision invariant ok')"` |
| Runtime guard | `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_runtime_dependency_guards.py` |
| Release matrix bound | `python -c "import json; s=json.load(open('.agent/work/2026-06-16-release-parity-hardening/parity-status.json')); assert set(s['models']) == {'da3_multiview','locateanything','rfdetr','sam3_image'}"` |
| Diff hygiene | `git diff --check` |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: A decision-only phase with near-zero blast radius — markdown artifacts plus a roadmap link, no `src/mlx_cv/` touch — and every evidence anchor is verified present: the three `references/{DEIMv2,eomt,sapiens2}` trees, the cited reference files, and all eight `src/mlx_cv/` spine surfaces including the existing DINOv3 backbone that two of the three candidates build on.
- Concern: Slices 1–4 verify only with `test -f` plus `rg` keyword greps, which confirm the artifacts exist and name the candidates but cannot detect a shallow analysis, stale checkpoint-availability evidence (e.g. `Intellindust/DEIMv2_DINOv3_S_COCO`, `facebook/sapiens2*`), or a decision that violates the "exactly one selected" invariant.
- Action: Strengthen Slice 4's verification to assert exactly one family is marked `selected` with a numeric score per candidate, and have `CANDIDATE-MATRIX.md` flag each checkpoint ID as confirmed-against-current-source versus local-reference-only, so the decision stays evidence-grounded rather than keyword-shaped.
- Verified: PLAN/DESIGN/SPEC read; confirmed `references/DEIMv2`, `references/eomt`, `references/sapiens2` and the cited files (`eomt/models/eomt.py`, `sapiens2/LICENSE.md`, `sapiens2/docs/MODEL_ZOO.md`, READMEs) exist; confirmed the eight cited spine surfaces exist (`backbones/vision/dinov3`, `necks`, `heads/{detection,segmentation,dense}`, `core/types.py`, `transforms`, `ops`); confirmed `tests/test_runtime_dependency_guards.py` and the bounded release-parity matrix; decision-only blast radius and runtime-boundary anti-goals reviewed.
