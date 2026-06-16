"""Canonical fixed inputs + tap schema for golden-fixture parity (§11).

The fixed input is defined **once** here (pure numpy, mlx-free) and reused by
both the out-of-band mint step (PyTorch oracle, ``tools/``) and the MLX parity
test, so the two sides compare on byte-identical input. The tap schema is the
*ordered* list of intermediate probes ``bisect`` walks to localize drift.

Phase-1 oracle (resolved at the plan decision checkpoint): **fixed-seed → MLX**
structural parity on DINOv3 **ViT-S/16** — the PyTorch reference is seeded, run
through ``forward_features``, and its weights are exported to the MLX port; the
two outputs must match. No HF-gated pretrained weights are involved.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DINOV3_VARIANT",
    "DINOV3_FIXTURE_CONFIG",
    "DINOV2_DA3_FIXTURE_CONFIG",
    "DA3_MONOCULAR_FIXTURE_CONFIG",
    "QWEN2_FIXTURE_CONFIG",
    "MOONVIT_FIXTURE_CONFIG",
    "LOCATEANYTHING_FIXTURE_CONFIG",
    "RFDETR_FIXTURE_CONFIG",
    "RFDETR_MS_DEFORM_ATTN_FIXTURE_CONFIG",
    "SAM3_FIXTURE_CONFIG",
    "dinov3_fixed_input",
    "dinov3_tap_order",
    "dinov2_da3_fixed_input",
    "dinov2_da3_tap_order",
    "da3_monocular_tap_order",
    "qwen2_fixed_inputs",
    "moonvit_fixed_inputs",
    "moonvit_tap_order",
    "locateanything_fixed_inputs",
    "locateanything_tap_order",
    "rfdetr_fixed_input",
    "rfdetr_fixed_image",
    "rfdetr_tap_order",
    "rfdetr_ms_deform_attn_fixed_inputs",
    "sam3_fixed_image",
    "sam3_pcs_prompt",
    "sam3_tap_order",
    "sam3_text_prompt",
]

# The real Phase-1 target variant (DINOv3 ViT-S/16 defaults). Used for *shape*
# conformance of the MLX port (Slice 4) at realistic dims.
DINOV3_VARIANT = {
    "name": "vit_small",
    "patch_size": 16,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "n_storage_tokens": 4,
    "img_size": 64,          # 64/16 -> 4x4 = 16 patch tokens
}

# The committed *numerical-parity* fixture config. A size-reduced instance of the
# SAME DINOv3 architecture / code path (RoPE, attention, Mlp, LayerNorm, storage
# tokens, cls/storage/patch split, multi-block residuals) — small enough that the
# fixed-seed weights commit (~0.6 MB) instead of ViT-S/16's ~88 MB. Fidelity of the
# proof is unchanged: "our MLX forward == reference PyTorch forward for identical
# weights" does not depend on the width/depth. Full ViT-S/16 parity can be minted
# on demand with `tools/mint_dinov3_fixture.py --variant vit_small`.
DINOV3_FIXTURE_CONFIG = {
    "name": "dinov3_tiny_fixture",
    "patch_size": 16,
    "embed_dim": 64,
    "depth": 2,
    "num_heads": 2,
    "ffn_ratio": 4.0,
    "n_storage_tokens": 2,
    "img_size": 32,          # 32/16 -> 2x2 = 4 patch tokens; tokens = 1 + 2 + 4 = 7
    "pos_embed_rope_base": 100.0,
    "pos_embed_rope_dtype": "fp32",   # fp32 RoPE for clean fp32 parity (vs reference bf16 default)
}


DINOV2_DA3_FIXTURE_CONFIG = {
    "name": "dinov2_da3_tiny_fixture",
    "patch_size": 14,
    "embed_dim": 32,
    "depth": 4,
    "num_heads": 4,
    "ffn_ratio": 4.0,
    "n_register_tokens": 0,
    "pretrain_grid": 2,
    "img_size": 28,          # 28/14 -> 2x2; avoids pos-emb interpolation drift
    "intermediate_layers": [0, 1, 2, 3],
    "layer_norm_eps": 1e-6,
    "final_norm_eps": 1e-5,
}


DA3_MONOCULAR_FIXTURE_CONFIG = {
    "name": "da3_monocular_tiny_fixture",
    "dinov2": DINOV2_DA3_FIXTURE_CONFIG,
    "dpt": {
        "dim_in": 32,
        "patch_size": 14,
        "output_dim": 2,
        "activation": "exp",
        "conf_activation": "expp1",
        "features": 16,
        "out_channels": [8, 8, 8, 8],
        "pos_embed": False,
        "down_ratio": 1,
        "head_name": "depth",
        "use_sky_head": False,
        "norm_type": "idt",
    },
}


QWEN2_FIXTURE_CONFIG = {
    "name": "qwen2_tiny_fixture",
    "seed": 7,
    "vocab_size": 32,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "max_position_embeddings": 16,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_cache": False,
    "tie_word_embeddings": True,
    "attn_implementation": "sdpa",
    "block_size": 2,
    "causal_attn": False,
    "text_mask_token_id": 7,
}


MOONVIT_FIXTURE_CONFIG = {
    "name": "moonvit_tiny_fixture",
    "seed": 11,
    "hidden_size": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 16,
    "patch_size": 2,
    "num_channels": 3,
    "init_pos_emb_height": 2,
    "init_pos_emb_width": 2,
    "merge_kernel_size": (2, 2),
    "attn_implementation": "sdpa",
    "grid_hws": ((2, 2), (4, 2)),
}


LOCATEANYTHING_FIXTURE_CONFIG = {
    "name": "locateanything_tiny_fixture",
    "seed": 19,
    "vision": {
        "hidden_size": 8,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "patch_size": 2,
        "num_channels": 1,
        "init_pos_emb_height": 2,
        "init_pos_emb_width": 2,
        "merge_kernel_size": (2, 2),
    },
    "text": {
        "vocab_size": 1200,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32,
        "block_size": 6,
        "text_mask_token_id": 1112,
        "null_token_id": 1110,
        "switch_token_id": 1111,
        "eos_token_id": 2,
    },
    "image_token_index": 50,
    "box_start_token_id": 70,
    "box_end_token_id": 71,
    "ref_start_token_id": 72,
    "ref_end_token_id": 73,
    "none_token_id": 74,
    "coord_start_token_id": 100,
    "coord_end_token_id": 1100,
    "text_mask_token_id": 1112,
    "image_size": (10, 20),
    "model_size": (20, 40),
}


RFDETR_FIXTURE_CONFIG = {
    "name": "rfdetr_tiny_fixture",
    "seed": 23,
    "image_size": (28, 28),
    "num_select": 4,
    "labels": ("class_0", "class_1", "class_2"),
    "backbone": {
        "embed_dim": 16,
        "depth": 2,
        "num_heads": 2,
        "patch_size": 14,
        "n_register_tokens": 2,
        "pretrain_grid": 2,
    },
    "out_layers": (0, 1),
    "projector_out_channels": 8,
    "projector_scale_factors": (2.0, 1.0),
    "decoder": {
        "hidden_dim": 8,
        "num_queries": 4,
        "num_heads": 2,
        "num_layers": 1,
        "num_points": 2,
        "num_classes": 3,
    },
}


RFDETR_MS_DEFORM_ATTN_FIXTURE_CONFIG = {
    "name": "rfdetr_ms_deform_attn_tiny_fixture",
    "value_spatial_shapes": ((2, 2), (1, 3)),
    "batch_size": 1,
    "num_heads": 2,
    "head_dim": 2,
    "num_queries": 2,
    "num_points": 2,
}


SAM3_FIXTURE_CONFIG = {
    "name": "sam3_tiny_fixture",
    "seed": 31,
    "image_size": (32, 32),
    "num_select": 3,
    "labels": ("background", "object"),
    "text_prompt": "cat",
    "box_prompt": [[4, 4, 20, 24]],
    "exemplar_shape": (16, 16, 3),
    "exemplar_boxes": [[2, 2, 10, 12]],
    "image": {
        "image_size": 32,
        "patch_size": 4,
        "embed_dim": 8,
        "depth": 2,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "text_dim": 6,
        "out_layers": (0, 1),
        "neck_channels": 4,
        "neck_scales": (1.0, 0.5),
    },
    "text": {
        "d_model": 6,
        "context_length": 8,
        "width": 8,
        "heads": 2,
        "layers": 1,
        "mlp_ratio": 2.0,
    },
    "decoder": {
        "hidden_dim": 4,
        "num_queries": 3,
        "num_layers": 1,
        "num_heads": 1,
        "num_classes": 2,
        "text_dim": 6,
    },
}


def dinov3_fixed_input(seed: int = 0, img_size: int | None = None) -> np.ndarray:
    """Deterministic ``(1, 3, H, W)`` float32 input for a DINOv3 parity fixture."""
    h = w = DINOV3_VARIANT["img_size"] if img_size is None else img_size
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 3, h, w)).astype(np.float32)


def dinov2_da3_fixed_input(seed: int = 0, img_size: int | None = None) -> np.ndarray:
    """Deterministic ``(1, 3, H, W)`` float32 input for the DA3 DINOv2 fixture."""
    h = w = DINOV2_DA3_FIXTURE_CONFIG["img_size"] if img_size is None else img_size
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 3, h, w)).astype(np.float32)


def qwen2_fixed_inputs() -> dict[str, np.ndarray]:
    """Deterministic no-cache SDLM input for the tiny Qwen2 parity fixture."""
    cfg = QWEN2_FIXTURE_CONFIG
    return {
        "input_ids": np.array([[3, 5, cfg["text_mask_token_id"], cfg["text_mask_token_id"]]], dtype=np.int64),
        "position_ids": np.array([[0, 1, 0, 1]], dtype=np.int64),
    }


def moonvit_fixed_inputs() -> dict[str, np.ndarray]:
    """Deterministic packed-patch input for the tiny MoonViT fixture."""
    cfg = MOONVIT_FIXTURE_CONFIG
    rng = np.random.default_rng(cfg["seed"])
    grid_hws = np.array(cfg["grid_hws"], dtype=np.int32)
    seq_len = int(np.sum(grid_hws[:, 0] * grid_hws[:, 1]))
    pixel_values = rng.standard_normal(
        (
            seq_len,
            cfg["num_channels"],
            cfg["patch_size"],
            cfg["patch_size"],
        )
    ).astype(np.float32)
    return {"pixel_values": pixel_values, "grid_hws": grid_hws}


def locateanything_fixed_inputs() -> dict[str, np.ndarray]:
    """Deterministic local integration input for LocateAnything."""
    cfg = LOCATEANYTHING_FIXTURE_CONFIG
    vocab = cfg["text"]["vocab_size"]
    logits = np.full((6, vocab), -20.0, dtype=np.float32)
    sampled = [
        cfg["box_start_token_id"],
        cfg["coord_start_token_id"] + 250,
        cfg["coord_start_token_id"] + 250,
        cfg["coord_start_token_id"] + 750,
        cfg["coord_start_token_id"] + 750,
        cfg["box_end_token_id"],
    ]
    for row, token in enumerate(sampled):
        logits[row, token] = 20.0
    generated = np.array(
        [
            cfg["ref_start_token_id"],
            12,
            cfg["ref_end_token_id"],
            *sampled,
            cfg["box_start_token_id"],
            cfg["coord_start_token_id"] + 500,
            cfg["coord_start_token_id"] + 500,
            cfg["box_end_token_id"],
        ],
        dtype=np.int32,
    )
    return {
        "input_ids": np.array([[10, cfg["image_token_index"], 11]], dtype=np.int32),
        "cached_image_features": np.arange(32, dtype=np.float32).reshape(1, 32) / 31.0,
        "pbd_block_logits": logits,
        "generated_ids": generated,
    }


def rfdetr_fixed_input(seed: int | None = None) -> np.ndarray:
    """Deterministic ``(1, 3, H, W)`` float32 input for the RF-DETR fixture."""
    cfg = RFDETR_FIXTURE_CONFIG
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    h, w = cfg["image_size"]
    return rng.standard_normal((1, 3, h, w)).astype(np.float32)


def rfdetr_fixed_image() -> np.ndarray:
    """Deterministic ``(H, W, 3)`` uint8 image for RF-DETR predict tests."""
    h, w = RFDETR_FIXTURE_CONFIG["image_size"]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.uint8), np.arange(w, dtype=np.uint8), indexing="ij")
    return np.stack(
        [
            (xx * 7 + yy) % 255,
            (yy * 11 + 13) % 255,
            (xx * 3 + yy * 5 + 29) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)


def sam3_fixed_image() -> np.ndarray:
    """Deterministic ``(H, W, 3)`` uint8 image for SAM3 predict/parity tests."""
    h, w = SAM3_FIXTURE_CONFIG["image_size"]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.uint8), np.arange(w, dtype=np.uint8), indexing="ij")
    return np.stack(
        [
            (xx * 5 + yy * 3 + 7) % 255,
            (yy * 9 + 19) % 255,
            (xx * 11 + yy + 23) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)


def sam3_text_prompt() -> str:
    """Deterministic SAM3 text prompt."""
    return str(SAM3_FIXTURE_CONFIG["text_prompt"])


def sam3_pcs_prompt() -> list[dict]:
    """Deterministic SAM3 PCS box plus exemplar prompt."""
    cfg = SAM3_FIXTURE_CONFIG
    exemplar = np.zeros(tuple(cfg["exemplar_shape"]), dtype=np.uint8)
    yy, xx = np.meshgrid(
        np.arange(exemplar.shape[0], dtype=np.uint8),
        np.arange(exemplar.shape[1], dtype=np.uint8),
        indexing="ij",
    )
    exemplar[..., 0] = (xx * 13 + 5) % 255
    exemplar[..., 1] = (yy * 17 + 11) % 255
    exemplar[..., 2] = (xx + yy * 3 + 29) % 255
    return [
        {"boxes": np.asarray(cfg["box_prompt"], dtype=np.float64)},
        {"exemplar_image": exemplar, "exemplar_boxes": np.asarray(cfg["exemplar_boxes"], dtype=np.float64)},
    ]


def rfdetr_ms_deform_attn_fixed_inputs() -> dict[str, np.ndarray]:
    """Deterministic tiny input for RF-DETR deformable-attention parity."""
    cfg = RFDETR_MS_DEFORM_ATTN_FIXTURE_CONFIG
    shapes = np.array(cfg["value_spatial_shapes"], dtype=np.int32)
    spatial_size = int(np.sum(shapes[:, 0] * shapes[:, 1]))
    value = (
        np.arange(
            cfg["batch_size"] * cfg["num_heads"] * cfg["head_dim"] * spatial_size,
            dtype=np.float32,
        ).reshape(cfg["batch_size"], cfg["num_heads"], cfg["head_dim"], spatial_size)
        + 1.0
    ) / 10.0
    sampling_locations = np.array(
        [
            [
                [
                    [[[0.25, 0.25], [0.75, 0.75]], [[0.0, 0.5], [1.1, 0.5]]],
                    [[[0.5, 0.5], [0.0, 0.0]], [[0.5, 0.5], [1.0, 0.5]]],
                ],
                [
                    [[[0.1, 0.9], [0.9, 0.1]], [[0.5, 0.0], [-0.2, 0.5]]],
                    [[[0.25, 0.75], [0.75, 0.25]], [[0.1, 0.5], [0.9, 0.5]]],
                ],
            ]
        ],
        dtype=np.float32,
    )
    attention_weights = np.array(
        [
            [
                [[[0.4, 0.1], [0.3, 0.2]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.1], [0.2, 0.2]]],
            ]
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [
                [0.18299997, 0.66599995, 1.26874995, 1.75],
                [0.1243, 0.33220002, 1.6500001, 2.2940001],
            ]
        ],
        dtype=np.float32,
    )
    return {
        "value": value.astype(np.float32),
        "value_spatial_shapes": shapes,
        "sampling_locations": sampling_locations,
        "attention_weights": attention_weights,
        "expected": expected,
    }


def dinov3_tap_order(depth: int | None = None) -> list[str]:
    """Ordered intermediate taps for ``bisect``.

    Forward order: patch-embed -> RoPE sin/cos -> each transformer block ->
    final norm -> the cls / storage / patch split of the normed sequence.
    """
    depth = DINOV3_VARIANT["depth"] if depth is None else depth
    taps = ["patch_embed", "rope_sincos"]
    taps += [f"block_{i:02d}" for i in range(depth)]
    taps += ["norm", "cls", "storage", "patch"]
    return taps


def dinov2_da3_tap_order(
    depth: int | None = None,
    intermediate_layers: list[int] | tuple[int, ...] | None = None,
) -> list[str]:
    """Ordered taps for DA3-style DINOv2 parity."""
    depth = DINOV2_DA3_FIXTURE_CONFIG["depth"] if depth is None else depth
    layers = DINOV2_DA3_FIXTURE_CONFIG["intermediate_layers"] if intermediate_layers is None else intermediate_layers
    taps = ["patch_embed"]
    taps += [f"block_{i:02d}" for i in range(depth)]
    taps += [f"intermediate_{int(i):02d}" for i in layers]
    taps += ["norm", "cls", "patch"]
    return taps


def da3_monocular_tap_order() -> list[str]:
    """Ordered taps for end-to-end DA3 monocular parity."""
    taps = [f"dinov2.{k}" for k in dinov2_da3_tap_order()]
    taps += [f"dpt.projected_{i}" for i in range(4)]
    taps += ["dpt.fusion_4", "dpt.fusion_3", "dpt.fusion_2", "dpt.fusion_1", "dpt.output_logits"]
    return taps


def moonvit_tap_order(depth: int | None = None, num_images: int | None = None) -> list[str]:
    """Ordered taps for MoonViT packed-patch parity."""
    cfg = MOONVIT_FIXTURE_CONFIG
    depth = cfg["num_hidden_layers"] if depth is None else depth
    num_images = len(cfg["grid_hws"]) if num_images is None else num_images
    taps = ["patch_embed", "rope_freqs_cis", "attention_mask_visible"]
    taps += [f"block_{i:02d}" for i in range(depth)]
    taps += ["norm"]
    taps += [f"merged_{i:02d}" for i in range(num_images)]
    return taps


def locateanything_tap_order() -> list[str]:
    """Ordered taps for LocateAnything local integration fixture."""
    return [
        "projector",
        "inputs_embeds",
        "pbd_block_logits",
        "sampled_tokens",
        "generated_ids",
        "boxes",
        "points",
    ]


def rfdetr_tap_order(
    num_levels: int | None = None,
    num_layers: int | None = None,
    *,
    include_self_attention: bool = False,
) -> list[str]:
    """Ordered taps for RF-DETR detector parity."""
    cfg = RFDETR_FIXTURE_CONFIG
    num_levels = len(cfg["projector_scale_factors"]) if num_levels is None else num_levels
    num_layers = cfg["decoder"]["num_layers"] if num_layers is None else num_layers
    taps = [f"projector.level_{i}" for i in range(num_levels)]
    for i in range(num_layers):
        if include_self_attention:
            taps.append(f"decoder.self_attention_{i}")
        taps.append(f"decoder.deformable_attention_{i}")
    taps += ["decoder.hidden_states", "head.logits", "head.boxes", "result.boxes", "result.scores", "result.class_ids"]
    return taps


def sam3_tap_order(
    *,
    num_levels: int | None = None,
    include_text: bool = False,
    include_geometry: bool = False,
) -> list[str]:
    """Ordered taps for SAM3 image-mode parity."""
    cfg = SAM3_FIXTURE_CONFIG
    num_levels = len(cfg["image"]["neck_scales"]) if num_levels is None else num_levels
    taps: list[str] = []
    if include_text:
        taps += ["text.token_ids", "text.language_features", "text.language_embeds"]
    if include_geometry:
        taps += ["prompt.boxes_cxcywh", "prompt.exemplar_boxes_cxcywh"]
    taps += ["backbone.patch_tokens"]
    taps += [f"neck.level_{i}" for i in range(num_levels)]
    taps += [
        "decoder.hidden_states",
        "head.mask_logits",
        "head.object_scores",
        "head.boxes",
        "result.masks",
        "result.boxes",
        "result.scores",
        "result.class_ids",
    ]
    return taps
