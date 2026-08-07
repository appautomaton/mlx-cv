# Upstream reference implementations

This directory is the repository's existing local workspace for upstream
reference checkouts used by parity tools and tests. The checkouts are ignored by
Git; this policy file is tracked.

Do not relocate, delete, or rewrite an existing checkout as routine cleanup.
Some current tools and opt-in tests intentionally resolve paths beneath
`references/`, and a checkout may contain local compatibility changes required
by its comparison harness.

## Boundary

- Reference code is a test oracle, not a runtime dependency.
- `src/mlx_cv/` must not add a reference checkout to `sys.path`, import it, or
  require its framework at runtime.
- Reference repositories, their licenses, and their Git histories remain
  separate from the `mlx-cv` source distribution.
- Existing directories here are local test infrastructure even though they are
  not tracked by the parent repository.

## Current workspace roles

| Checkout | Canonical source | Current role |
|---|---|---|
| `Depth-Anything-3` | [ByteDance Seed](https://github.com/ByteDance-Seed/Depth-Anything-3) | DA3 upstream parity oracle |
| `LocateAnything-3B` | [NVIDIA](https://huggingface.co/nvidia/LocateAnything-3B) | LocateAnything upstream parity oracle |
| `dinov3` | [Meta](https://github.com/facebookresearch/dinov3) | DINOv3 fixture and implementation oracle |
| `rf-detr` | [Roboflow](https://github.com/roboflow/rf-detr) | RF-DETR upstream parity oracle |
| `sam3` | [Meta](https://github.com/facebookresearch/sam3) | SAM 3.1 upstream parity oracle; may carry local Apple-platform compatibility patches |
| `eomt` | [TUE MPS](https://github.com/tue-mps/eomt) | Available architecture reference; the current EoMT gate uses a Transformers output capture instead |

Other existing checkouts may be retained as historical research infrastructure
even when no current test imports them. Their presence does not make them a
supported `mlx-cv` model family.

Because the checkouts are ignored, the parent repository does not pin their
working-tree state. Any durable parity claim must record the upstream revision
and relevant local compatibility diff with its evidence.

## Temporary test artifacts

New generated test files do not belong in this directory or elsewhere in the
repository. Use a dedicated system temporary root for new pytest state, caches,
downloads, builds, captures, logs, and debug output:

```bash
MLX_CV_TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mlx-cv-tests.XXXXXX")"
trap 'rm -rf "$MLX_CV_TEST_ROOT"' EXIT
```

Pass paths beneath `MLX_CV_TEST_ROOT` to the particular tool or test being run.
The shell trap removes that temporary root on exit. This rule applies to newly
generated artifacts; it does not change or move the repository's existing
tests, committed fixtures, or local reference checkouts.

## Evidence

Generated comparison media and full checkpoints remain untracked. Bounded
status, provenance, and measured results may be recorded under `.agent/work/`
when a work item explicitly requires durable evidence.
