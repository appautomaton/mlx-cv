# Next Change Brief: EoMT-DINOv3 Real Checkpoint Admission

Likely change slug: `2026-06-17-eomt-dinov3-real-checkpoint-admission`

## Objective

Admit the selected **EoMT-DINOv3** family through a real-checkpoint status gate. The gate should either load and compare the smallest credible EoMT-DINOv3 checkpoint path or record a precise blocker that names the missing external checkpoint, DINOv3 base access, delta-weight composition, local converter, or comparison tap.

## Selected Source And Checkpoint

| Field | Target |
|---|---|
| Source repo | https://github.com/tue-mps/eomt |
| Local reference | `references/eomt/` |
| Model ID | `tue-mps/coco_panoptic_eomt_small_640_dinov3` |
| Checkpoint file | `pytorch_model.bin` |
| Config | `references/eomt/configs/dinov3/coco/panoptic/eomt_small_640_2x.yaml` |
| Reference module | `references/eomt/models/eomt.py:EoMT` |
| Important access note | EoMT DINOv3 weights are deltas against original DINOv3 weights; the base DINOv3 checkpoint is a required external artifact. |

## Env And Cache Shape

| Variable/path | Purpose |
|---|---|
| `MLX_CV_EOMT_DINOV3_CHECKPOINT` | Absolute path to `pytorch_model.bin` for `tue-mps/coco_panoptic_eomt_small_640_dinov3`. |
| `MLX_CV_EOMT_DINOV3_BASE_CHECKPOINT` | Absolute path to the required original DINOv3 base checkpoint. |
| `MLX_CV_EOMT_DINOV3_CONFIG` | Optional override for the EoMT config path; default should be the local reference config above. |
| `MLX_CV_REQUIRE_EOMT_DINOV3_GATE=1` | Required-gate mode; no silent skip when the user asks for the gate. |
| `/tmp/mlx-cv-checkpoints/eomt-dinov3/` | Out-of-git checkpoint/cache root for EoMT artifacts. |
| `/tmp/mlx-cv-checkpoints/dinov3/` | Out-of-git cache root for base DINOv3 artifacts if not already present. |

## First Gate Shape

Likely status artifact:

`.agent/work/2026-06-17-eomt-dinov3-real-checkpoint-admission/eomt-dinov3-status.json`

Likely command shapes:

```bash
UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_eomt_dinov3_upstream.py
```

```bash
UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run python tools/eomt_dinov3_upstream.py --status .agent/work/2026-06-17-eomt-dinov3-real-checkpoint-admission/eomt-dinov3-status.json
```

The future spec/plan can rename the tool or test, but it should preserve the gate semantics: no checkpoint means precise `BLOCKED`, not a fake pass.

## Expected Outputs And Taps

- Reference input: one RGB image resized/preprocessed according to the EoMT DINOv3 COCO panoptic config.
- Reference outputs: final mask logits and class logits from `EoMT.forward`.
- Local output target: `Result.masks` for the first public surface; full panoptic metadata can be deferred until the tensor gate is honest.
- Comparison taps: DINOv3 patch/token features after base+delta resolution, query-token insertion boundary, final mask logits, final class logits.

## Blocker Taxonomy

- `external_checkpoint_missing`
- `dinov3_base_access_missing`
- `delta_weight_composition_missing`
- `unsupported_checkpoint_format`
- `local_query_token_path_missing`
- `local_scale_block_missing`
- `converter_missing`
- `comparison_tap_missing`
- `reference_runtime_unavailable`

## Anti-Goals

- Do not download or commit weights into git.
- Do not add EoMT to the release parity matrix during admission.
- Do not claim upstream parity unless the real checkpoint, reference output, and local output have actually matched within a defined tolerance.
- Do not implement full panoptic dataset evaluation or COCO panoptic serialization in the first admission change.
- Do not add PyTorch, Lightning, or reference-code imports to package import paths.
- Do not implement DEIMv2 or Sapiens2 as part of the EoMT admission change.
