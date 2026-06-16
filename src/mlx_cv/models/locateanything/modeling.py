"""LocateAnything model assembly (MLX).

This module owns the multimodal boundary: MoonViT encodes packed image patches,
the projector maps merged vision features into Qwen2 hidden size, and image-token
positions in the language embedding stream are replaced with those projected
features. Qwen2 and MoonViT remain reusable backbones.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from ...backbones.llm.qwen2.cache import Qwen2KVCache
from ...backbones.llm.qwen2.modeling import Qwen2ForCausalLM
from ...backbones.vision.moonvit.modeling import MoonViTBackbone
from .config import LocateAnythingConfig

__all__ = ["LocateAnythingProjector", "LocateAnythingModel"]


class LocateAnythingProjector(nn.Module):
    """MoonViT merged-feature projector into Qwen2 hidden space."""

    def __init__(self, config: LocateAnythingConfig) -> None:
        super().__init__()
        vision_hidden = config.vision_config.hidden_size
        merge_h, merge_w = config.vision_config.merge_kernel_size
        self.input_dim = int(vision_hidden * merge_h * merge_w)
        self.output_dim = int(config.text_config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.linear_1 = nn.Linear(self.input_dim, self.output_dim)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(self.output_dim, self.output_dim)

    def __call__(self, image_features: Sequence[mx.array] | mx.array) -> mx.array:
        if isinstance(image_features, mx.array):
            features = image_features
        else:
            if not image_features:
                return mx.zeros((0, self.input_dim))
            features = mx.concatenate(list(image_features), axis=0)
        features = features.reshape(-1, self.input_dim)
        hidden = self.layer_norm(features)
        hidden = self.linear_1(hidden)
        hidden = self.act(hidden)
        return self.linear_2(hidden)


class LocateAnythingModel(nn.Module):
    """Assembled LocateAnything VLM: MoonViT + projector + Qwen2."""

    def __init__(self, config: LocateAnythingConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocateAnythingConfig()
        self.vision_tower = MoonViTBackbone(self.config.vision_config)
        self.language_model = Qwen2ForCausalLM(self.config.text_config)
        self.multi_modal_projector = LocateAnythingProjector(self.config)
        self.image_token_index = self.config.image_token_index

    def _project_image_features(
        self,
        *,
        pixel_values: mx.array | None,
        image_grid_hws=None,
        cached_image_features: Sequence[mx.array] | mx.array | None,
    ) -> mx.array | None:
        if cached_image_features is not None:
            return self.multi_modal_projector(cached_image_features)
        if pixel_values is None:
            return None
        if image_grid_hws is None:
            raise ValueError("image_grid_hws is required when pixel_values are provided")
        image_features = self.vision_tower(pixel_values, image_grid_hws)
        return self.multi_modal_projector(image_features)

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        *,
        image_grid_hws=None,
        cached_image_features: Sequence[mx.array] | mx.array | None = None,
        image_token_id: int | None = None,
    ) -> mx.array:
        """Embed text ids and scatter projected image features into image-token slots."""
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        image_features = self._project_image_features(
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            cached_image_features=cached_image_features,
        )
        if image_features is None:
            return inputs_embeds

        token_id = self.image_token_index if image_token_id is None else int(image_token_id)
        image_mask = input_ids == token_id
        token_count = int(mx.sum(image_mask.astype(mx.int32)).item())
        feature_count = int(image_features.shape[0])
        if token_count != feature_count:
            raise ValueError(
                "image token count must match projected image features: "
                f"{token_count} tokens vs {feature_count} features"
            )
        if token_count == 0:
            return inputs_embeds

        mask_flat = image_mask.reshape(-1)
        feature_index = (mx.cumsum(mask_flat.astype(mx.int32)) - 1).reshape(input_ids.shape)
        feature_index = mx.where(image_mask, feature_index, 0)
        gathered = image_features[feature_index]
        return mx.where(image_mask[..., None], gathered, inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        *,
        image_grid_hws=None,
        cached_image_features: Sequence[mx.array] | mx.array | None = None,
        image_token_id: int | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        past_key_values: Qwen2KVCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple:
        inputs_embeds = self.get_input_embeddings(
            input_ids,
            pixel_values,
            image_grid_hws=image_grid_hws,
            cached_image_features=cached_image_features,
            image_token_id=image_token_id,
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    def make_cache(self) -> Qwen2KVCache:
        return Qwen2KVCache(len(self.language_model.model.layers))

    def pbd_generate(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        *,
        image_grid_hws=None,
        cached_image_features: Sequence[mx.array] | mx.array | None = None,
        generation_mode: str = "hybrid",
        max_tokens: int = 2048,
        cache: Qwen2KVCache | None = None,
        n_future_tokens: int | None = None,
    ) -> list[int]:
        from .pbd import PBDDecoder

        inputs_embeds = self.get_input_embeddings(
            input_ids,
            pixel_values,
            image_grid_hws=image_grid_hws,
            cached_image_features=cached_image_features,
        )
        decoder = PBDDecoder(self, generation_mode=generation_mode, n_future_tokens=n_future_tokens)
        return decoder.generate(input_ids, inputs_embeds, cache or self.make_cache(), max_tokens=max_tokens)
