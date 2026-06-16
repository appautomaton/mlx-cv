# Slice 7 Summary

Final status: complete.

Changed files:
- `src/mlx_cv/parity/fixtures.py`: added opt-in RF-DETR self-attention tap ordering while preserving the tiny fixture default order.
- `src/mlx_cv/parity/rfdetr_real.py`: added local MLX real-checkpoint RF-DETR Nano capture with upstream-style tensor preprocessing, strict-loaded weights, raw outputs, typed detections, ordered taps, and named tap gap.
- `tests/test_rfdetr_real_forward.py`: added dedicated local real-forward gate with optional skip, required failure paths, preprocessing equality check, raw output checks, class-id semantics checks, and ordered tap checks.

Verification:
- Sandbox command failed before collection with `No Metal device available`.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=/tmp/mlx-cv-checkpoints/rf-detr-nano.pth uv run pytest tests/test_rfdetr_real_forward.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` passed outside sandbox with 10 tests.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run pytest tests/test_rfdetr_real_forward.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` passed outside sandbox with 9 passed, 1 skipped.

Reviewer verdicts:
- Initial spec reviewer: changes requested because local preprocessing used generic PIL bicubic instead of upstream tensor resize.
- Spec re-review: approved after local capture matched upstream-style preprocessing and test-side torchvision comparison was added.
- Quality reviewer: approved.

Unresolved risks:
- Real checkpoint gates require a verified checkpoint and unsandboxed Metal access.
