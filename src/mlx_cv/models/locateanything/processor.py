"""LocateAnything processor (pre/post) — Stage 3 (not yet implemented).

Contract (implements ``core.base.Processor``):

  preprocess(image, prompt) -> (model_inputs, ctx)
    * MoonViT dynamic resize (bicubic, to patch/merge multiples; mean/std 0.5)
      recorded in a ``SpatialTransform`` ctx
    * build the chat template; expand each ``<image-N>`` into
      ``<img> + <IMG_CONTEXT> * N + </img>`` with N = grid_h*grid_w / (merge^2)

  postprocess(token_ids, ctx) -> Result
    * decode.parse_grounding_tokens -> GroundingItem s
    * tokenizer-decode label ids -> text; coords [0,1000] -> pixels via ctx
    * assemble Detections (boxes, labels; scores=None) + Points

Requires the ``mlx`` extra + the model tokenizer; lands with Stage 3.
"""

from __future__ import annotations

__all__: list[str] = []
