# Repository instructions

## Dependency boundary

The root `pyproject.toml` describes the installable `mlx-cv` package. Keep only
build requirements, core runtime dependencies, and user-facing runtime extras
there. Do not add test, publishing, or upstream-reference dependencies to the
root project metadata.

A developer environment may contain any tools needed for its task. The boundary
applies to project metadata and runtime imports, not to packages installed in a
venv.

## Progressive disclosure

Read only the material relevant to the current task:

- Public API and usage: `README.md`.
- Module boundaries and runtime invariants: `docs/ARCHITECTURE.md`.
- Model package formats: `docs/model-packages.md`, then the model-specific doc.
- Model-specific parity: read the relevant model documentation only when
  working on that model's parity check.
- Shared parity fixtures and bisection: `tools/parity/`.
- Upstream capture and release staging: the relevant top-level file under
  `tools/`; these are repository workflows, not `mlx_cv` runtime modules.
- Historical decisions and evidence: `.agent/work/` (read-only; never rewrite or
  reorganize it as current documentation).

## Environments

- The root `pyproject.toml` uses `.venv`, which is the MLX environment and does
  not include PyTorch.
- Any task that requires PyTorch must use `.venv-torch`.
- Manage `.venv-torch` dependencies as needed for the current parity or
  reference check; do not add them to the root `pyproject.toml`.

## Repository hygiene

Keep permanent tests in `tests/` and do not move existing fixtures or local
reference checkouts. Put newly generated caches, downloads, converted weights,
and debugging artifacts in a fresh system temporary directory. Clean that
directory after the run unless the user explicitly asks to inspect an artifact.

Keep parity helpers, upstream-framework integrations, and release orchestration
under `tools/`. Code under `src/mlx_cv/` is wheel content and must not import
repository-only `tools` modules.
