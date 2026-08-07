# EoMT–DINOv3 panoptic segmentation

`mlx-cv` implements the COCO panoptic EoMT small-640 model as an MLX-native
inference runtime. The supported source is the current complete
[`tue-mps/eomt-dinov3-coco-panoptic-small-640`](https://huggingface.co/tue-mps/eomt-dinov3-coco-panoptic-small-640)
package, whose architecture is also implemented in
[Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/eomt_dinov3/modeling_eomt_dinov3.py).

## Public surface

```python
import mlx_cv

model = mlx_cv.load("eomt-dinov3")
result = model.predict(image)

panoptic_ids = result.masks.data
segments = result.metadata["segments_info"]
rendered = result.draw(image)
```

The processor preserves aspect ratio, pads the resized image on its bottom and
right edges to 640×640, applies ImageNet normalization, and maps masks back to
the original image size. The result contains:

- `Result.masks.kind == "panoptic"`;
- a two-dimensional segment-ID map with `-1` for unassigned pixels;
- ordered `segments_info` entries with segment ID, class ID, score, and label
  when the package supplies `id2label`.

## Architecture and checkpoint contract

The admitted model uses a 12-layer DINOv3-small patch-16 backbone with four
storage tokens, 200 EoMT queries inserted before the final three transformer
blocks, a 134-way class predictor (133 COCO classes plus no-object), a three-layer
mask MLP, and two learned 2× scale blocks.

The official package is a complete 231-tensor Safetensors checkpoint. Loading is
strict: separate query/key/value projections are packed into the local attention
tree, PyTorch convolution layouts are converted to MLX layouts, unsupported keys
are rejected, and every non-deterministic local parameter must be present.

`attn_mask_probs` is part of the inference contract. The released checkpoint
stores `[-1, -1, -1]`, which disables intermediate hard-mask attention while
retaining query insertion. Dropping this tensor changes final predictions and is
therefore rejected by strict coverage.

## Verification

`tools/eomt_reference.py` runs the official Transformers model on CPU and writes
one compressed reference contract. The NPZ contains the source image, official
`pixel_values`, all 12 transformer block outputs, final mask and class logits,
the panoptic map, ordered segment IDs and labels, and JSON provenance. The gate
also records and validates the `0.8` object threshold, `0.5` mask threshold,
`0.8` overlap threshold, and COCO stuff-class IDs. It does not infer a pass from
previously recorded numbers.

Create the capture and run the opt-in gates from an isolated system temporary
root:

```bash
MLX_CV_TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mlx-cv-tests.XXXXXX")"
trap 'rm -rf "$MLX_CV_TEST_ROOT"' EXIT

EOMT_PACKAGE=/absolute/path/to/eomt-dinov3-coco-panoptic-small-640
EOMT_IMAGE=/absolute/path/to/reference-image.jpg

HF_HOME="$MLX_CV_TEST_ROOT/huggingface" \
.venv-torch/bin/python tools/eomt_reference.py \
  --model "$EOMT_PACKAGE" \
  --image "$EOMT_IMAGE" \
  --output "$MLX_CV_TEST_ROOT/eomt-reference.npz"

PYTHONDONTWRITEBYTECODE=1 \
MLX_CV_EOMT_DINOV3_PACKAGE="$EOMT_PACKAGE" \
MLX_CV_EOMT_DINOV3_REFERENCE_OUTPUT="$MLX_CV_TEST_ROOT/eomt-reference.npz" \
MLX_CV_REQUIRE_EOMT_DINOV3_GATE=1 \
MLX_CV_REQUIRE_EOMT_DINOV3_METAL_GATE=1 \
.venv/bin/python -m pytest -q tests/test_eomt_dinov3_real_checkpoint.py \
  --basetemp="$MLX_CV_TEST_ROOT/pytest" \
  -o "cache_dir=$MLX_CV_TEST_ROOT/pytest-cache"
```

The CPU gate compares every `block_00` through `block_11` output plus final mask
and class logits with `atol=rtol=0.002`. The Metal gate requires identical class
argmaxes and ordered panoptic segment IDs/labels, at least `99.9%` mask-sign
agreement, and at least `99.98%` panoptic-map agreement. The switches are
separate so a CPU reference host can require the detailed numerical gate without
claiming that Metal ran. `MLX_CV_EOMT_DINOV3_INPUT` remains available only for a
shape/load smoke check; a complete parity capture carries its own exact input.

The previously recorded run against Transformers 5.14.1 and the official full
checkpoint measured maximum CPU errors of `0.0007782` for masks and
`0.00003171` for classes. On a 640×480 COCO sample, all 200 Metal class argmaxes
matched, mask signs agreed on `99.9559%` of pixels, the four ordered segment
labels matched, and panoptic maps agreed on `99.9899%` of pixels. These values
are dated evidence; the automated gates above enforce the current contract.

Checkpoint downloads, reference runtimes, images, captures, and pytest state
belong in the system temporary test root, not Git.

## Deliberate limits

- COCO panoptic small-640 only;
- one image per processor call;
- no training, dataset evaluator, semantic-only convenience API, COCO panoptic
  serializer, or larger-variant promise;
- legacy delta-only `pytorch_model.bin` artifacts are not complete packages.
