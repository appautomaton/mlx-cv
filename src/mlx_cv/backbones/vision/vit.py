"""Shared ViT assembly (build-once family) — one backbone, two posenc paths.

`ViTBackbone` is the parameterized ViT body every ViT-family model composes:
patch-embed -> prepend ``[cls, storage…]`` -> N pre-norm `TransformerBlock`s ->
final norm -> split into ``cls / storage / patch`` (the spine ``BackboneFeatures``
contract). It reuses the Slice-1/2 leaves (`PatchEmbed`, `TransformerBlock`, the
RoPE helpers) and never re-defines a block / attention / rope of its own.

The one tricky abstraction is the **position-encoding seam**. DINOv3 applies 2D
RoPE *inside each attention block* (sin/cos from the patch grid, prefix tokens
skipped); DINOv2 adds a learned absolute pos-emb *once* after patch-embed and
runs no rope. `ViTBackbone` dispatches on a small position-strategy object so the
single assembly serves both:

* `RoPEStrategy` — fully implemented here (DINOv3). Holds ``rope_base``; the
  backbone materializes the top-level ``periods`` buffer from it, and the strategy
  produces ``(sin, cos)`` per forward for the blocks to apply.
* abs strategy (``kind == "abs"``) — a clean **unfilled seam** the DINOv2 slice
  fills *without editing this assembly*: it implements `add_pos` (adds the learned
  pos-emb to ``[cls, patch]`` only, before storage insertion) and leaves rope off.

**Unified token-assembly order** (DESIGN §PositionStrategy, eng-review B2):

```
1. patches = patch_embed(x)
2. x = [cls, patches]
3. if abs:  x = x + abs_pos(x)            # pos on cls+patch ONLY
4. insert storage after cls:  x = [cls, storage…, patch]   # specials get NO pos
5. if rope: sin,cos = rope(grid)          # else None
6. blocks(x, rope=sin/cos|None, n_prefix=1+n_storage)      # rope hits patch only
7. final norm -> split [cls, storage…, patch]
```

For the RoPE path step 3 is skipped and RoPE already skips the ``n_prefix``
(cls+storage), so inserting storage at step 4 is numerically identical to the
``[cls, storage, patch]``-then-RoPE order — the reorder is parity-safe for DINOv3
and correct for DINOv2.

mlx lives here, behind the ``[mlx]`` extra; ``core/`` stays mlx-free.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.features import BackboneFeatures, FeatureMap, Layout, TokenLayout
from ..layers import PatchEmbed, TransformerBlock
from ..layers.position import LearnedAbsPosEmb, rope_axial_periods, rope_axial_sincos

__all__ = ["ViTBackbone", "RoPEStrategy", "AbsPosStrategy", "PositionStrategy"]


class PositionStrategy:
    """Base position-encoding seam dispatched by `ViTBackbone`'s assembly.

    Two questions the assembly asks every strategy:

    * ``uses_rope`` — does attention receive ``(sin, cos)``? (RoPE: yes; abs: no)
    * ``needs_periods`` — should the backbone materialize a top-level ``periods``
      buffer for this strategy? (RoPE: yes; abs: no)

    Hooks the assembly calls (default no-ops keep the abs branch a clean seam):

    * ``build(backbone)`` — one-time setup at ``__init__`` (e.g. learned pos-emb
      params for abs); the RoPE buffer is created by the backbone itself.
    * ``add_pos(backbone, x, grid)`` — step 3: add absolute pos-emb to ``[cls,
      patch]`` *before* storage insertion (abs only). Default: identity.
    * ``rope(backbone, hp, wp)`` — step 5: return ``(sin, cos)`` or ``None``.
    """

    kind: str = "none"
    uses_rope: bool = False
    needs_periods: bool = False

    def build(self, backbone: "ViTBackbone") -> None:
        return None

    def add_pos(self, backbone: "ViTBackbone", x: mx.array, grid: tuple[int, int]) -> mx.array:
        return x

    def rope(self, backbone: "ViTBackbone", hp: int, wp: int) -> tuple[mx.array, mx.array] | None:
        return None


class RoPEStrategy(PositionStrategy):
    """Axial 2D-RoPE applied inside each block (DINOv3).

    Holds only ``rope_base``; the backbone owns the top-level ``periods`` buffer
    (so its param path stays ``periods``, not nested under a strategy). Per forward
    this builds ``(sin, cos)`` for the ``hp×wp`` grid from that buffer.
    """

    kind = "rope"
    uses_rope = True
    needs_periods = True

    def __init__(self, rope_base: float = 100.0) -> None:
        self.rope_base = rope_base

    def rope(self, backbone: "ViTBackbone", hp: int, wp: int) -> tuple[mx.array, mx.array]:
        return rope_axial_sincos(backbone.periods, hp, wp)


class AbsPosStrategy(PositionStrategy):
    """Learned absolute pos-emb added once to ``[cls, patch]`` (DINOv2); no rope.

    ``build`` attaches a `LearnedAbsPosEmb` (top-level ``pos_embed`` on the
    backbone); ``add_pos`` adds it at step 3 — *before* storage insertion — so
    registers receive no positional embedding (eng-review B2). ``rope`` stays the
    base no-op, so attention runs without rotary encoding.
    """

    kind = "abs"
    uses_rope = False
    needs_periods = False

    def __init__(self, pretrain_grid: int | tuple[int, int]) -> None:
        self.pretrain_grid = pretrain_grid

    def build(self, backbone: "ViTBackbone") -> None:
        backbone.pos_embed = LearnedAbsPosEmb(backbone.embed_dim, self.pretrain_grid)

    def add_pos(self, backbone: "ViTBackbone", x: mx.array, grid: tuple[int, int]) -> mx.array:
        return x + backbone.pos_embed(grid)


class ViTBackbone(nn.Module):
    """Parameterized ViT body shared by DINOv3 / DINOv2 (assembly only).

    Sub-classes (or callers) bind a config to the plain axes below. The position
    strategy is the main DINOv3-vs-DINOv2 swap (RoPE vs learned-abs), alongside
    LayerScale on/off, patch size, dims, and register count.

    Top-level attribute paths created here — ``patch_embed``, ``cls_token``,
    ``storage_tokens`` (when ``n_storage > 0``), ``periods`` (RoPE strategy only),
    ``blocks``, ``norm`` — are the param-tree paths the weight loader expects, so a
    subclass that only binds config keeps a byte-identical param tree.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int = 16,
        in_chans: int = 3,
        n_storage: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm: str = "layernorm",
        norm_eps: float = 1e-6,
        final_norm_eps: float | None = None,
        ffn: str = "gelu",
        layerscale: bool = False,
        layerscale_init: float = 1.0,
        position: PositionStrategy,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_storage = n_storage
        self.position = position

        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.cls_token = mx.zeros((1, 1, embed_dim))
        if n_storage > 0:
            self.storage_tokens = mx.zeros((1, n_storage, embed_dim))
        # Top-level RoPE periods buffer (path stays ``periods``, not nested).
        if position.needs_periods:
            self.periods = rope_axial_periods(embed_dim // num_heads, position.rope_base)
        self.blocks = [
            TransformerBlock(
                embed_dim, num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                norm=norm, norm_eps=norm_eps, ffn=ffn,
                layerscale=layerscale, layerscale_init=layerscale_init,
            )
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps if final_norm_eps is None else final_norm_eps)
        # One-time strategy setup (abs pos-emb params land here; RoPE is a no-op).
        position.build(self)

    def __call__(self, x: mx.array) -> BackboneFeatures:
        return self.forward_features(x)

    def forward_features(
        self,
        x: mx.array,
        *,
        intermediate_layers: int | list[int] | tuple[int, ...] | None = None,
        capture_taps: bool = False,
    ) -> BackboneFeatures:
        if intermediate_layers is None:
            layers_to_take: set[int] = set()
        elif isinstance(intermediate_layers, int):
            if intermediate_layers < 0 or intermediate_layers > self.depth:
                raise ValueError(
                    f"intermediate_layers={intermediate_layers} outside valid range 0..{self.depth}"
                )
            layers_to_take = set(range(self.depth - intermediate_layers, self.depth))
        else:
            layers_to_take = set(int(i) for i in intermediate_layers)
            invalid = sorted(i for i in layers_to_take if i < 0 or i >= self.depth)
            if invalid:
                raise ValueError(
                    f"intermediate layer indices {invalid} outside valid range 0..{self.depth - 1}"
                )

        taps: dict[str, mx.array] = {}
        patches, (hp, wp) = self.patch_embed(x)              # (B, P, D)
        b, _, d = patches.shape
        cls = mx.broadcast_to(self.cls_token, (b, 1, d))

        # Step 2-3: [cls, patch], then abs-pos on cls+patch only (rope: no-op seam).
        x = mx.concatenate([cls, patches], axis=1)           # (B, 1+P, D)
        x = self.position.add_pos(self, x, (hp, wp))

        # Step 4: insert storage tokens AFTER cls -> [cls, storage…, patch].
        if self.n_storage > 0:
            storage = mx.broadcast_to(self.storage_tokens, (b, self.n_storage, d))
            tokens = mx.concatenate([x[:, :1], storage, x[:, 1:]], axis=1)
        else:
            tokens = x
        if capture_taps:
            taps["patch_embed"] = tokens

        # Step 5: rope sin/cos for the grid (or None for abs strategies).
        rope = self.position.rope(self, hp, wp)
        if capture_taps and rope is not None:
            sin, cos = rope
            taps["rope_sincos"] = mx.stack([sin, cos])

        # Step 6: blocks; rope hits patch tokens only (n_prefix cls+storage skipped).
        n_prefix = 1 + self.n_storage
        z = tokens
        selected_tokens: list[tuple[int, mx.array]] = []
        for i, blk in enumerate(self.blocks):
            z = blk(z, rope=rope, n_prefix=n_prefix)
            if capture_taps:
                taps[f"block_{i:02d}"] = z
            if i in layers_to_take:
                selected_tokens.append((i, z))

        # Step 7: final norm -> split [cls, storage…, patch].
        z_norm = self.norm(z)
        intermediates: list[FeatureMap] = []
        for i, selected in selected_tokens:
            selected_norm = self.norm(selected)
            patch_selected = selected_norm[:, n_prefix:]
            intermediates.append(
                FeatureMap(
                    patch_selected,
                    layout=Layout.BNC,
                    grid=(hp, wp),
                    stride=self.patch_size,
                )
            )
            if capture_taps:
                taps[f"intermediate_{i:02d}"] = patch_selected
        cls_out = z_norm[:, 0]
        storage_out = z_norm[:, 1:n_prefix]
        patch_out = z_norm[:, n_prefix:]
        if capture_taps:
            taps["norm"] = z_norm
            taps["cls"] = cls_out
            if self.n_storage > 0:
                taps["storage"] = storage_out
            taps["patch"] = patch_out

        extras: dict = {"x_prenorm": z}
        if capture_taps:
            extras["taps"] = taps
        return BackboneFeatures(
            patch_tokens=FeatureMap(patch_out, layout=Layout.BNC, grid=(hp, wp), stride=self.patch_size),
            cls_token=cls_out,
            storage_tokens=storage_out if self.n_storage > 0 else None,
            token_layout=TokenLayout.vit(n_storage=self.n_storage),
            intermediates=intermediates,
            extras=extras,
        )
