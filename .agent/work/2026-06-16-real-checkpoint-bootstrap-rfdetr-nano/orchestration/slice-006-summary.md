# Slice 6 Summary

Final status: complete.

Changed files:
- `src/mlx_cv/models/rfdetr/config.py`: added `RFDETRConfig.rfdetr_nano()` for the real Nano checkpoint shape.
- `src/mlx_cv/models/rfdetr/convert.py`: added RF-DETR HF-style DINOv2 remaps, qkv packing, explicit `mask_token` inference-only exclusion, duplicate detection, and strict missing-key validation.
- `tools/rfdetr_convert_checkpoint.py`: added tool-side `.pth` to out-of-git `.npz` conversion and converted-cache resolution.
- `tests/test_rfdetr_real_checkpoint_load.py`: added dedicated real checkpoint conversion/load gate.
- `tests/test_rfdetr_convert.py`: added converter coverage for DINOv2 remaps, qkv packing, explicit exclusions, and strict load failures.

Verification:
- Sandbox command failed during MLX import before collection with `No Metal device available`.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=/tmp/mlx-cv-checkpoints/rf-detr-nano.pth uv run pytest tests/test_rfdetr_real_checkpoint_load.py tests/test_rfdetr_convert.py tests/test_runtime_dependency_guards.py` passed outside sandbox with 23 tests.

Reviewer verdicts:
- Spec reviewer: approved.
- Quality reviewer: approved.

Unresolved risks:
- Real checkpoint gates require a verified checkpoint and unsandboxed Metal access.
