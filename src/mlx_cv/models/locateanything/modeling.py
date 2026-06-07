"""LocateAnything model assembly (MLX) — Stage 3 (not yet implemented).

Contract: an ``nn.Module`` ``Model`` composed of

    vision_tower            -> backbones/vision/moonvit (MoonViT-SO-400M)
    multi_modal_projector   -> LayerNorm(4608) -> Linear -> GELU -> Linear(2048)
    language_model          -> backbones/llm/qwen2 (Qwen2.5-3B)

``get_input_embeddings`` projects MoonViT features and scatters them into the LLM
embedding stream at ``image_token_index`` positions; ``pbd_generate`` drives the
Parallel Box Decoder (Stage 4). Requires the ``mlx`` extra; the v0.0.2 spine ships
mlx-free, so this lands with Stage 2/3.
"""

from __future__ import annotations

__all__: list[str] = []
