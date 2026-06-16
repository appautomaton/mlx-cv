# Slice 8 Summary

Final status: complete.

Changed files:
- `tests/test_rfdetr_upstream_parity.py`: replaced placeholder/status-only gate with real upstream-vs-MLX RF-DETR Nano checkpoint parity.
- `src/mlx_cv/parity/rfdetr_real.py`: runs local real-checkpoint capture on `mx.cpu` to match the Torch CPU oracle.
- `src/mlx_cv/heads/detection/rfdetr.py`: stabilizes two-stage proposal ordering for near-tied scores before decoder self-attention.
- `src/mlx_cv/models/rfdetr/processor.py`: matches upstream RF-DETR postprocess by not clipping final boxes.
- `tests/test_rfdetr_nano_decoder.py`, `tests/test_rfdetr_processor.py`: cover the proposal-order and no-clipping regressions.

Verification:
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=/tmp/mlx-cv-checkpoints/rf-detr-nano.pth PYTHONPATH=references/rf-detr/src uv run pytest -q tests/test_rfdetr_upstream_parity.py tests/test_rfdetr_real_forward.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py tests/test_rfdetr_nano_decoder.py tests/test_rfdetr_processor.py`: passed with 26 tests and printed checkpoint path/MD5.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_CACHE=/tmp/mlx-cv-empty-rfdetr-cache uv run pytest -q tests/test_rfdetr_upstream_parity.py`: passed with 1 passed, 1 skipped.
- `git diff --check`: passed.

Reviewer verdicts:
- Implementer initially blocked on real parity drift.
- Coordinator diagnosed the blocker as default-stream execution, near-tie proposal ordering, and RF-DETR postprocess clipping.
- Spec reviewer approved after requiring visible checkpoint path/MD5 output.
- Quality reviewer approved with no issues.

Unresolved risks:
- Real checkpoint gates need the verified out-of-git checkpoint and unsandboxed MLX device access.
