"""Qwen2.5 language-backbone building blocks.

This submodule is the MLX boundary for Qwen2. Package-root/config imports stay
mlx-free; later slices extend this file with attention, decoder layers, and the
registered builder.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ....core.registry import register_backbone
from .cache import Qwen2KVCache
from .config import Qwen2Config
from .masks import (
    make_causal_mask_4d,
    update_causal_mask_for_one_gen_window_2d,
    update_causal_mask_with_pad_non_visible_2d,
)

__all__ = [
    "Qwen2RMSNorm",
    "Qwen2RotaryEmbedding",
    "Qwen2Attention",
    "Qwen2DecoderLayer",
    "Qwen2MLP",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "build_qwen2",
    "rotate_half",
    "apply_rotary_pos_emb",
    "repeat_kv",
]


class Qwen2RMSNorm(nn.Module):
    """Qwen2 RMSNorm, equivalent to the reference Llama/T5-style layer norm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        x = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.astype(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    """1D Qwen2 RoPE table with the reference half-split convention."""

    def __init__(
        self,
        dim: int,
        *,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

    def __call__(self, x: mx.array, seq_len: int | None = None) -> tuple[mx.array, mx.array]:
        seq_len = int(seq_len if seq_len is not None else x.shape[-2])
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


def rotate_half(x: mx.array) -> mx.array:
    """Rotate the last-dimension halves: ``[x1, x2] -> [-x2, x1]``."""
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    position_ids: mx.array,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[mx.array, mx.array]:
    """Apply gathered RoPE tables to query/key tensors."""
    cos = mx.expand_dims(cos[position_ids], axis=unsqueeze_dim)
    sin = mx.expand_dims(sin[position_ids], axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Module):
    """Reference SwiGLU block: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise NotImplementedError(f"Qwen2MLP only supports hidden_act='silu', got {config.hidden_act!r}")
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        return self.down_proj((gate * mx.sigmoid(gate)) * self.up_proj(x))


def repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
    """Expand KV heads from ``(B, kv_heads, T, D)`` to attention heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = mx.expand_dims(hidden_states, axis=2)
    hidden_states = mx.broadcast_to(
        hidden_states,
        (batch, num_key_value_heads, n_rep, slen, head_dim),
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """Manual additive-mask GQA attention for Qwen2."""

    def __init__(self, config: Qwen2Config, layer_idx: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads: {self.hidden_size} vs {self.num_heads}"
            )
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{self.num_heads} vs {self.num_key_value_heads}"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        past_key_value=None,
        *,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[mx.array, mx.array | None, object | None]:
        if use_cache and past_key_value is None:
            raise ValueError("use_cache=True requires a Qwen2KVCache")
        if past_key_value is not None and not isinstance(past_key_value, Qwen2KVCache):
            raise TypeError(f"past_key_value must be Qwen2KVCache, got {type(past_key_value).__name__}")
        if past_key_value is not None and self.layer_idx is None:
            raise ValueError("Qwen2Attention needs layer_idx when using a KV cache")

        batch, q_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = mx.broadcast_to(
                mx.arange(q_len, dtype=mx.int32).reshape(1, q_len),
                (batch, q_len),
            )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = mx.transpose(
            query_states.reshape(batch, q_len, self.num_heads, self.head_dim),
            (0, 2, 1, 3),
        )
        key_states = mx.transpose(
            key_states.reshape(batch, q_len, self.num_key_value_heads, self.head_dim),
            (0, 2, 1, 3),
        )
        value_states = mx.transpose(
            value_states.reshape(batch, q_len, self.num_key_value_heads, self.head_dim),
            (0, 2, 1, 3),
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        rotary_seq_len = max(kv_seq_len, int(mx.max(position_ids)) + 1)
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )
        present_key_value = None
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
            present_key_value = past_key_value

        if attention_mask is not None:
            mask_shape = (batch, 1, q_len, kv_seq_len)
            if attention_mask.shape != mask_shape:
                raise ValueError(f"Attention mask should have shape {mask_shape}, got {attention_mask.shape}")

        if not output_attentions:
            attn_output = mx.fast.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                scale=self.head_dim ** -0.5,
                mask=attention_mask,
            )
            attn_probs = None
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = (query_states @ mx.transpose(key_states, (0, 1, 3, 2))) / math.sqrt(
                self.head_dim
            )
            expected_shape = (batch, self.num_heads, q_len, kv_seq_len)
            if attn_weights.shape != expected_shape:
                raise ValueError(f"Attention weights should have shape {expected_shape}, got {attn_weights.shape}")
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_probs = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(query_states.dtype)
            attn_output = attn_probs @ value_states

        output_shape = (batch, self.num_heads, q_len, self.head_dim)
        if attn_output.shape != output_shape:
            raise ValueError(f"Attention output should have shape {output_shape}, got {attn_output.shape}")

        attn_output = mx.transpose(attn_output, (0, 2, 1, 3)).reshape(batch, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_probs, present_key_value


class Qwen2DecoderLayer(nn.Module):
    """One Qwen2 decoder block with reference residual ordering."""

    def __init__(self, config: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        past_key_value=None,
        *,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs: tuple = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen2Model(nn.Module):
    """Bare Qwen2 decoder stack."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.causal_attn = config.causal_attn
        self.text_mask_token_id = config.text_mask_token_id

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def _embed_with_visual_features(
        self,
        input_ids: mx.array,
        visual_features: mx.array | None,
        image_token_index: int | None,
    ) -> mx.array:
        if visual_features is None:
            return self.embed_tokens(input_ids)
        del image_token_index
        raise NotImplementedError("visual feature insertion is owned by the later LocateAnything VLM assembly")

    def _prepare_attention_mask(
        self,
        *,
        input_ids: mx.array | None,
        inputs_embeds: mx.array,
        attention_mask: mx.array | None,
        position_ids: mx.array,
        past_key_values_length: int = 0,
        use_cache: bool = False,
    ) -> mx.array:
        del position_ids
        batch_size, seq_length, _ = inputs_embeds.shape
        key_value_length = seq_length + past_key_values_length
        if attention_mask is None:
            mask = make_causal_mask_4d(
                batch_size,
                seq_length,
                key_value_length=key_value_length,
                dtype=inputs_embeds.dtype,
            )
        elif attention_mask.ndim == 4:
            mask = attention_mask
            expected = (batch_size, 1, seq_length, key_value_length)
            if mask.shape != expected:
                raise ValueError(f"4D attention_mask must have shape {expected}, got {mask.shape}")
        elif attention_mask.ndim == 2:
            if attention_mask.shape != (batch_size, key_value_length):
                raise ValueError(
                    "2D attention_mask width must match cached key/value length "
                    f"{key_value_length}, got {attention_mask.shape}"
                )
            mask = make_causal_mask_4d(
                batch_size,
                seq_length,
                key_value_length=key_value_length,
                dtype=inputs_embeds.dtype,
            )
            pad = mx.where(
                attention_mask[:, None, None, :] == 0,
                mx.array(-float("inf"), dtype=inputs_embeds.dtype),
                mx.array(0.0, dtype=inputs_embeds.dtype),
            )
            mask = mask + pad
        else:
            raise ValueError(f"attention_mask must be 2D or 4D, got shape {attention_mask.shape}")

        if input_ids is None or seq_length == 1:
            return mask

        rows: list[mx.array] = []
        for b in range(batch_size):
            row = mask[b, 0]
            if int(input_ids[b, -1]) == self.text_mask_token_id:
                if use_cache:
                    full = make_causal_mask_4d(
                        1,
                        key_value_length,
                        dtype=inputs_embeds.dtype,
                    )[0, 0]
                    row = update_causal_mask_for_one_gen_window_2d(
                        input_ids[b],
                        full,
                        block_size=self.block_size,
                        use_cache=True,
                        causal_attn=self.causal_attn,
                    )[-seq_length:, :]
                else:
                    row = update_causal_mask_with_pad_non_visible_2d(
                        input_ids[b],
                        row,
                        self.text_mask_token_id,
                        block_size=self.block_size,
                        causal_attn=self.causal_attn,
                    )
            rows.append(mx.expand_dims(row, axis=0))
        return mx.stack(rows, axis=0)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        visual_features: mx.array | None = None,
        image_token_index: int | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        past_key_values=None,
        inputs_embeds: mx.array | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> tuple:
        del return_dict
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, Qwen2KVCache):
            raise TypeError(f"past_key_values must be Qwen2KVCache, got {type(past_key_values).__name__}")
        if use_cache and past_key_values is None:
            past_key_values = Qwen2KVCache(len(self.layers))
        past_key_values_length = past_key_values.get_seq_length(0) if past_key_values is not None else 0
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Specify either input_ids or inputs_embeds")
            inputs_embeds = self._embed_with_visual_features(input_ids, visual_features, image_token_index)

        batch_size, seq_length, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = mx.broadcast_to(
                mx.arange(
                    past_key_values_length,
                    past_key_values_length + seq_length,
                    dtype=mx.int32,
                ).reshape(1, seq_length),
                (batch_size, seq_length),
            )
        else:
            position_ids = position_ids.reshape(batch_size, seq_length).astype(mx.int32)

        attention_mask = self._prepare_attention_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values_length=past_key_values_length,
            use_cache=use_cache,
        )

        hidden_states = inputs_embeds
        all_hidden_states: tuple = () if output_hidden_states else ()
        all_self_attns: tuple = () if output_attentions else ()
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs: tuple = (hidden_states,)
        if use_cache:
            outputs += (past_key_values,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_self_attns,)
        return outputs


class Qwen2ForCausalLM(nn.Module):
    """Qwen2 decoder with tied embedding logits."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.text_mask_token_id = config.text_mask_token_id

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def compute_logits(self, hidden_states: mx.array) -> mx.array:
        return hidden_states @ mx.transpose(self.model.embed_tokens.weight)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        visual_features: mx.array | None = None,
        image_token_index: int | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        past_key_values=None,
        inputs_embeds: mx.array | None = None,
        labels=None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> tuple:
        if labels is not None:
            raise NotImplementedError("Qwen2ForCausalLM loss support is outside this backbone slice")
        outputs = self.model(
            input_ids=input_ids,
            visual_features=visual_features,
            image_token_index=image_token_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.compute_logits(outputs[0])
        return (logits,) + outputs[1:]


@register_backbone("qwen2.5-3b", kind="llm")
def build_qwen2(config) -> Qwen2ForCausalLM:
    cfg = config if isinstance(config, Qwen2Config) else Qwen2Config.from_dict(config)
    return Qwen2ForCausalLM(cfg)
