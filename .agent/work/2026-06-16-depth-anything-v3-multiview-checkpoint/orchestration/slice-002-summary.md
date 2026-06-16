# Slice 2 Orchestration Summary

Final status: complete

Decision: upstream DA3 multi-view capture is implemented and verified for `depth-anything/DA3-SMALL`.

Changed files:
- `tools/da3_upstream.py`: added tool-local upstream DA3 capture, checkpoint loading, CPU/float32 execution, selector-call recording, output schema validation, and NPZ export.
- `tests/test_da3_upstream_capture.py`: added optional skip, required fail, mock schema, selector-evidence, runtime-boundary, and real-checkpoint gate coverage.
- `src/mlx_cv/parity/fixtures.py`: added deterministic three-view `(3,112,112,3)` uint8 fixture.
- `pyproject.toml`, `uv.lock`: added reproducible `test`/`da3-reference` optional dependency coverage for clean uv environments.
- `tests/test_runtime_dependency_guards.py`, `tests/test_qwen2_integration_guards.py`, `tests/test_da3_parity.py`: narrowed runtime dependency guards to base `[project.dependencies]` with Python 3.9 TOML compatibility.
- `PLAN.md`: corrected verification commands to use `--extra test` and `--extra da3-reference`.

Verification:
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test --extra mlx pytest tests/test_da3_parity.py tests/test_runtime_dependency_guards.py tests/test_qwen2_integration_guards.py tests/test_da3_upstream_capture.py` -> 19 passed, 1 skipped.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_da3_upstream_capture.py tests/test_da3_checkpoint_gate.py` -> 16 passed, 1 skipped.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL PYTHONPATH=references/Depth-Anything-3/src uv run --extra test --extra da3-reference pytest tests/test_da3_upstream_capture.py tests/test_da3_checkpoint_gate.py` -> 17 passed, 1 warning.
- `UV_PROJECT_ENVIRONMENT=/tmp/mlx-cv-py39-guard-venv UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --python 3.9 --extra test python ...` -> guard modules imported.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_CACHE=/tmp/mlx-cv-empty-da3-cache-slice2-check uv run --extra test pytest tests/test_da3_upstream_capture.py::test_da3_upstream_capture_runs_real_checkpoint -q -rs` -> 1 skipped, checkpoint not configured.
- `python3 -m py_compile tools/da3_upstream.py tests/test_da3_upstream_capture.py tests/test_runtime_dependency_guards.py tests/test_qwen2_integration_guards.py tests/test_da3_parity.py src/mlx_cv/parity/fixtures.py` -> passed.
- `git diff --check` -> passed.

Review verdicts:
- Implementer: `DONE_WITH_CONCERNS`; concern resolved by downloading/caching DA3-SMALL and making the reference extra reproducible.
- Spec review: `APPROVED`, then re-review `APPROVED`.
- Quality review: `CHANGES_REQUESTED` for incomplete reference extra and selector fallback; fixed. Follow-up `CHANGES_REQUESTED` for Python 3.9 TOML compatibility and duplicate raw pyproject guard; fixed. Final re-review `APPROVED`.

Unresolved risks or next action:
- Real-checkpoint capture depends on the external `references/Depth-Anything-3/src` checkout and the cached/resolved DA3 checkpoint.
- Next: Slice 3 real DA3 architecture contract.
