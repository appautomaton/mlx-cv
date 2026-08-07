# Roadmap

## Working principles

- Finish and document the current runtime surface before expanding it.
- Keep generated tests, caches, builds, downloads, and debug artifacts in an
  isolated system temporary directory.
- Treat committed fixtures as regression coverage and require a real
  upstream/local numeric comparison for every `UPSTREAM_PASSED` claim.
- Keep historical plans and evidence under `.agent/work/`; this file describes
  current and next work only.

## Completed: public-surface consolidation

Status: complete

The existing model implementations are ahead of the common user-facing API. The
current objective is to make the repository internally consistent before adding
another model family.

Completed outcomes:

- current, accurate, and visually coherent user documentation;
- one lazy `mlx_cv.load(...)` story across supported model families;
- a uniform public loader catalog separate from low-level compute registries;
- shared `Result`/`VideoResult` semantics across every supported runtime;
- Pillow visualization through `Result.draw()`;
- temporary-directory isolation for all generated verification artifacts;
- static and runtime checks that prevent documentation and public API drift.

Recorded verification at completion: the normal suite passed with 456 tests
and 13 expected skips, with pytest state and caches directed to a system
temporary root. This is dated evidence, not the current collected-test count.

## Completed: EoMT-DINOv3 checkpoint admission

Status: complete

The historical plan selected a delta checkpoint that required separate gated
DINOv3 base weights. The current upstream package now provides a complete
Safetensors checkpoint, so the admitted runtime uses that simpler and more
reproducible contract.

Completed outcomes:

- EoMT-DINOv3 small-640 config, DINOv3 LayerScale path, query insertion, mask
  and class heads, learned upscaling, preprocessing, and panoptic postprocessing;
- strict loading of the complete official 231-tensor Transformers package;
- preservation of checkpointed `attn_mask_probs` inference behavior;
- lazy `mlx_cv.load("eomt-dinov3")` dispatch and shared panoptic `Result` output;
- deterministic local coverage plus required real-checkpoint/reference gates;
- automated CPU comparison at all 12 block boundaries and final mask/class
  logits;
- automated real-image Metal checks for query classes, mask signs, segment
  labels, and panoptic-map agreement.

Measured evidence: CPU maximum absolute error was `0.0007782` for mask logits
and `0.00003171` for class logits. On the real COCO sample, the Metal and
Transformers panoptic maps agreed on `99.9899%` of pixels with identical four
segment labels.

The original selection evidence remains in
`.agent/work/2026-06-17-next-model-expansion-decision/`.

## Current work order

No release, tag, or additional model expansion is scheduled. Keep the five
supported families internally consistent, preserve historical `.agent/work/`
records, and keep all newly generated verification artifacts outside Git.
