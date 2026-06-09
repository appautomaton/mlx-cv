# references/ — upstream reference code (local, git-ignored)

Local shallow clones of upstream **reference implementations**, kept here for parity and
verification work only. **Everything in this folder except this README is git-ignored** and is
never committed.

## Discipline (ARCHITECTURE §16.6)

- **Read-from, never depend-on.** `src/mlx_cv/` must never import from `references/`. These clones
  are oracles we compare against, not runtime dependencies.
- **Clean-room / license hygiene.** mlx-cv code is MIT. Reference code carries its own licenses
  (Apache, research, etc.); keeping it git-ignored guarantees none of it leaks into the committed
  MIT tree.
- Use it to: mint golden fixtures, read a reference `sanitize()` / decode loop, bisect a parity gap.

## Populate (shallow, code-only)

```bash
# MLX oracle — the merged mlx-vlm LocateAnything port (fast pre-gate, same framework)
git clone --depth 1 https://github.com/Blaizzy/mlx-vlm.git references/mlx-vlm

# PyTorch reference truth — config + modeling code only, skip the ~7.66 GB LFS weights
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://huggingface.co/nvidia/LocateAnything-3B references/LocateAnything-3B
```

## Cloned corpus (foundation-validation)

Reference repos currently cloned here (code-only, LFS-skipped), and the spine contract each
stress-tests. Purpose: verify the spine generalizes (ARCHITECTURE §6) *before* hardening it —
the foundation, not any single model, is the product.

| Repo | Model | Spine contract it validates |
|------|-------|------------------------------|
| `mlx-vlm`, `LocateAnything-3B` | LocateAnything-3B | backbone `llm` kind + VLM + token decode (anchor) |
| `dinov3` | DINOv3 | backbone registry (`vision` kind); port-once-reuse |
| `rf-detr` | RF-DETR | DETR head + deformable attn + `detections` |
| `Depth-Anything-3` | Depth Anything V3 | dense-prediction head + `depth` |
| `sam3` | SAM 3.1 | `Tracker` mixin + video memory + mask decoder + prompt (hardest) |
| `eomt` | EoMT | encoder-only ViT (no separate head) |
| `sapiens2` | Sapiens2 | composable `Result` (pose + normals + depth + seg) |
| `DEIMv2` | DEIMv2 | DINOv3-backed detector (backbone reuse) |
