# Slice 3 Orchestration Summary

Final status: complete

Decision: DA3 real-checkpoint architecture contract is implemented and verified for `depth-anything/DA3-SMALL`.

Changed files:
- `tools/da3_real_architecture_contract.py`: added tools-only config/checkpoint contract auditor with provenance, architecture, tensor-group, tensor-shape, unsupported-branch, and local-monocular-gap checks.
- `tests/test_da3_real_architecture_contract.py`: added optional skip, required missing checkpoint/config/provenance failures, and real DA3-SMALL contract coverage.
- `PLAN.md`: corrected Slice 3 verification to include `--extra da3-reference`, then recorded completion evidence.

Verification:
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL uv run --extra test --extra da3-reference pytest tests/test_da3_real_architecture_contract.py tests/test_runtime_dependency_guards.py` -> 10 passed.
- `UV_PROJECT_ENVIRONMENT=/tmp/mlx-cv-slice3-clean-reference-venv4 UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_DA3_GATE=1 MLX_CV_DA3_MODEL_ID=depth-anything/DA3-SMALL uv run --extra test --extra da3-reference pytest tests/test_da3_real_architecture_contract.py tests/test_runtime_dependency_guards.py` -> 10 passed.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_da3_real_architecture_contract.py tests/test_runtime_dependency_guards.py` -> 10 passed in the current env.
- `python3 -m py_compile tools/da3_real_architecture_contract.py tests/test_da3_real_architecture_contract.py` -> passed.
- `git diff --check` -> passed.

Review verdicts:
- Implementer: `DONE`.
- Spec review: `APPROVED`.
- Quality review: `CHANGES_REQUESTED` for omitted real DualDPT aux LayerNorm tensors; fixed by requiring `head.scratch.output_conv2_aux.0.2.{weight,bias}` and asserting complete `437/437` required tensor coverage. Final re-review `APPROVED`.

Unresolved risks or next action:
- DA3-BASE is represented in the contract table but was not live-checkpoint exercised in this slice.
- Next: Slice 4 public multi-view result/processor contract.
