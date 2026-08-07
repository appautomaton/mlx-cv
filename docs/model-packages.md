# Local model packages

Every supported runtime can load from a local directory through either
`mlx_cv.load(...)` or its model-specific `from_pretrained(...)` constructor.

| Runtime key | Loader | Required package files |
|---|---|---|
| `locateanything-3b` | `LocateAnythingPipeline` | `model.safetensors`, `config.json`, tokenizer assets |
| `rfdetr-nano` | `RFDETRModel` | `model.safetensors` or `model.npz`, `config.json` |
| `depth-anything-v3-monocular` | `DepthAnythingV3Monocular` | `model.safetensors` or `model.npz`, `config.json` |
| `depth-anything-v3-multiview` | `DepthAnythingV3MultiView` | `model.safetensors` or `model.npz`, `config.json` |
| `eomt-dinov3-coco-panoptic-small-640` | `EoMTDINOv3` | `model.safetensors` or `model.npz`, `config.json` |
| `sam3.1-image` | `SAM3Processor` | `model.safetensors`, `bpe_simple_vocab_16e6.txt.gz` |
| `sam3.1-video` | `SAM3VideoSession` | `model.safetensors` |

Local directories bypass Hub imports and network lookup. Direct checkpoint files
are also accepted by RF-DETR, DA3, and EoMT when a `config` object or dictionary
is passed explicitly. Direct SAM image checkpoints additionally require
`bpe_path`.

## Configuration contract

RF-DETR accepts either its normalized configuration or a compact Nano marker:

```json
{
  "model_type": "rfdetr",
  "variant": "nano"
}
```

DA3 packages declare the runtime mode. A full normalized configuration may be
placed under `config`; without it, `monocular` selects the default monocular
contract and `multiview` selects DA3-SMALL.

```json
{
  "model_type": "depth-anything-v3",
  "mode": "multiview"
}
```

Package loaders reject incompatible family or mode markers. RF-DETR and DA3 use
strict parameter coverage by default; callers must opt out explicitly when
loading a deliberately partial development fixture.

EoMT accepts the complete official Transformers configuration
(`model_type: "eomt_dinov3"`) or this compact local marker:

```json
{
  "model_type": "eomt-dinov3",
  "variant": "coco-panoptic-small-640"
}
```

The current full package
`tue-mps/eomt-dinov3-coco-panoptic-small-640` is the supported source. The older
`pytorch_model.bin` artifact selected in historical planning was a delta against
a separately gated DINOv3 checkpoint; it is not accepted as a complete runtime
package.

## Unified loading

```python
import mlx_cv

locate = mlx_cv.load("locateanything-3b", "/path/to/locateanything-package")
rfdetr = mlx_cv.load("rfdetr-nano", "/path/to/rfdetr-package")
da3 = mlx_cv.load(
    "depth-anything-v3-multiview",
    "/path/to/da3-small-package",
)
eomt = mlx_cv.load("eomt-dinov3")
sam_image = mlx_cv.load("sam3.1-image", "/path/to/sam31-package")
sam_video = mlx_cv.load("sam3.1-video", "/path/to/sam31-package")
```

`mlx_cv.available_models()` returns the canonical runtime keys. Aliases such as
`rf-detr-nano`, `da3-small`, `eomt-dinov3`, and `sam3-video` resolve to the same
catalog entries.
Concrete model modules are imported only after dispatch, so `import mlx_cv`
remains MLX-free.

## Generated package staging

Generated package roots must use a system temporary directory or another
explicit artifact store:

```bash
MLX_CV_TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mlx-cv-tests.XXXXXX")"
trap 'rm -rf "$MLX_CV_TEST_ROOT"' EXIT
```

The existing staging tool builds and verifies the self-contained
LocateAnything/SAM packages:

```bash
python tools/huggingface_release.py stage --all \
  --staging-root "$MLX_CV_TEST_ROOT/model-packages"

python tools/huggingface_release.py verify --all \
  --staging-root "$MLX_CV_TEST_ROOT/model-packages"
```

Its verifier checks the manifest allowlist, sizes, SHA256 hashes, Safetensors
metadata, license presence, and model-card front-matter marker. The marker check
is structural rather than a complete YAML-schema parse.

RF-DETR and DA3 currently consume already-converted local package directories;
their offline conversion tools remain separate from runtime loading. EoMT can
consume either converted local weights or the current complete official
Safetensors package with strict local parameter validation.

## Remote resolution

Exact remote repository IDs, aliases, revisions, cache paths, tokens, and
`HF_HUB_OFFLINE=1` are handled by the same resolver when the optional Hub
dependency is installed. Resolution downloads files only and never executes
Python code from a model repository.

Remote creation, upload, deletion, visibility changes, tags, and publication are
outside the current workflow.
