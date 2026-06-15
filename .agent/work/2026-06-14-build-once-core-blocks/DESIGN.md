# DESIGN: Build-once core blocks (lean)

Change: `2026-06-14-build-once-core-blocks` · Spec: `SPEC.md` (this dir)

Non-trivial bits the SPEC leaves to design: **where the mlx families live**, **how one ViT assembly serves both RoPE (DINOv3) and learned-abs (DINOv2) posenc**, and **how parity is held green through the extraction**.

## Directory layout (the mlx-allowed home)

`core/` stays mlx-free, so the `nn.Module` families land under `backbones/` (the existing mlx zone). New:

```
src/mlx_cv/backbones/
  layers/                 ← NEW: shared mlx nn building blocks (the families)
    attention.py          ← Attention: packed-qkv + manual-softmax SDPA; optional rope hook
    block.py              ← TransformerBlock: pre-norm; selectable norm / FFN / LayerScale
    mlp.py                ← MlpFFN (GELU now); SwiGLU = wired slot, NotImplementedError
    patch_embed.py        ← PatchEmbed (conv, NCHW→NHWC)
    position.py           ← posenc suite: RoPE2D (DINOv3) + LearnedAbsPosEmb-interp (DINOv2)
  vision/
    vit.py                ← NEW: ViTBackbone family (assembly → BackboneFeatures) + PositionStrategy
    dinov3/               ← re-expressed: config + thin assembly + convert rules (behavior unchanged)
    dinov2/               ← NEW: config + thin assembly (convert/parity deferred to Phase 3)
src/mlx_cv/hub/           ← NEW seed: convert/sanitize rule engine
  convert.py              ← declarative rules (rename / transpose / drop) → [(path, mx.array)]
```

`core/layers` (BUILDING-BLOCKS #2) is **rejected** — it would force mlx into mlx-free `core/`. `hub/convert.py` follows the architecture's documented home; eng-review may collapse it into `backbones/` if a new top-level is deemed premature.

## Parameterization axes (build only what has a consumer now)

| Axis | DINOv3 | DINOv2 | This phase |
|---|---|---|---|
| posenc | 2D-RoPE (per-block) | learned-abs + interp (once) | **both built** |
| LayerScale | off (Identity) | on (init 1.0) | **on/off built** |
| norm | LayerNorm | LayerNorm | LayerNorm; RMSNorm = slot only |
| FFN | GELU-MLP | GELU-MLP | GELU-MLP; SwiGLU = slot only |
| attention | packed-qkv + SDPA | packed-qkv + SDPA | built |
| patch | 16 | 14 | config-driven |

Slots (RMSNorm/SwiGLU) are selectable enum values that raise `NotImplementedError` until Phase 4 (Qwen2) supplies the consumer — wired, not implemented.

## PositionStrategy — the one assembly, two posenc paths

The single tricky abstraction. DINOv3 applies RoPE **inside each attention block** (sin/cos from the patch grid, prefix tokens skipped); DINOv2 adds a learned pos-emb **once** after patch-embed and uses **no** rope in attention. `ViTBackbone` holds a `PositionStrategy`:

- `RoPEStrategy`: computes `(sin, cos)` per forward from the grid; assembly passes them into each block; `Attention` applies rope to q/k (prefix-skipped). DINOv3.
- `AbsPosStrategy`: adds (bicubic-interpolated) learned pos-emb to **`[cls, patch]` only**; blocks receive `rope=None`; `Attention` skips rope. DINOv2.

`Attention.__call__(x, rope=None, n_prefix=...)` — `rope` is the `(sin,cos)` pair or `None`. One attention, one assembly; the strategy is the main thing DINOv3 vs DINOv2 swap (plus LayerScale on/off, patch size, dims, registers).

**⟢ Token-assembly order (corrected per eng-review B2).** Register/storage tokens are inserted **after** the abs-pos step, so they receive **no** positional embedding — matching the DINOv2-with-registers reference (`references/rf-detr/.../dinov2_with_windowed_attn.py:425–459`: `cat([cls, patch]` → `+ pos` → `cat([cls, registers, patch])`). The unified assembly that serves both strategies:

```
1. patches = patch_embed(x)
2. x = [cls, patches]
3. if abs:  x = x + abs_pos(x)            # pos on cls+patch ONLY
4. insert storage/register after cls:      x = [cls, storage…, patch]   # specials get NO pos
5. if rope: sin,cos = rope(grid)           # else None
6. blocks(x, rope=sin/cos|None, n_prefix=1+n_storage)   # rope hits patch only
7. final norm → split [cls, storage…, patch]
```

For DINOv3 (rope) step 3 is skipped and RoPE already skips the `n_prefix` (cls+storage), so inserting specials at step 4 is **numerically identical** to the current `[cls, storage, patch]`-then-RoPE path — the reorder is parity-safe for DINOv3 and correct for DINOv2.

## Parity preservation (the gate that rides every slice)

The refactor must be **behavior-preserving for DINOv3**. Method:
- Extract → immediately rewire DINOv3 to import the extracted family → run `tests/test_dinov3_parity.py` (loads the committed fixture + minted weights, CPU stream, all ordered taps ≤ 2e-6, `assert_parity` atol 1e-4). Green after **every** slice.
- The ordered taps (`patch_embed`, `rope_sincos`, `block_NN`, `norm`, `cls/storage/patch`) localize any drift via `bisect` — keep `capture_taps` and the tap names/order intact through the assembly refactor (Slice 3 risk).
- DINOv3's converted weight paths (`blocks.{i}.attn.qkv`, `mlp.fc1`, `patch_embed.proj`, `periods`) must stay stable or the convert rules (Slice 4) update in lockstep; the parity loader is the check.

## Convert engine shape

`hub/convert.py`: a small rule set — `Rename(src,dst)`, `Transpose(key, axes)`, `Drop(key)`, default pass-through — applied over a `state_dict` to yield `[(mlx_path, mx.array)]` for `tree_unflatten`. DINOv3's three fixes (drop `mask_token`, rename `rope_embed.periods→periods`, transpose `patch_embed.proj.weight`) become declarative rules. One consumer this phase (DINOv3); full multi-model generalization proven in Phase 3.

## "No new block code" — how Slice 5 proves it

`backbones/vision/dinov2/` contains only `config.py` + a thin `modeling.py` that imports `ViTBackbone` + the shared families and wires a config — **no** `class *Attention/*Block/*Mlp` and **no** rope/posenc def. Verified by a grep over `dinov2/**/*.py` (Python files only — no `__pycache__` noise) returning no such definitions, plus the forward returning correct `BackboneFeatures` shapes/token-order.

**⟢ Grep is necessary, not sufficient (eng-review R1).** A name grep misses renamed classes or copied-in logic. The slice's real proof is the pairing: (a) grep finds no block/attention/mlp/posenc *definition*, AND (b) `dinov2/modeling.py` imports the families from `backbones/layers` + `backbones/vision/vit` (asserted), AND (c) the diff is small enough to eyeball at review. Treat the import-assertion + small-diff as the binding check; the grep is the cheap tripwire.
