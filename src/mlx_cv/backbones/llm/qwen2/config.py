"""Qwen2.5 config for LocateAnything's language backbone.

This module is intentionally mlx-free.  ``models.locateanything`` imports this
submodule for Stage-1 config/decode/convert tests, so package-root imports must
not pull in MLX modeling code or register the backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Qwen2Config"]


@dataclass
class Qwen2Config:
    """Qwen2.5-3B-Instruct decoder (GQA), with LocateAnything PBD fields."""

    hidden_size: int = 2048
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 11008
    vocab_size: int = 152681
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    use_cache: bool = False
    pad_token_id: int | None = None
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    use_sliding_window: bool = False
    sliding_window: int = 32768
    max_window_layers: int = 70
    # Local implementation path.  The reference checkpoint config says "magi",
    # but parity fixtures are minted through the comparable SDPA/manual mask path.
    attn_implementation: str = "sdpa"
    # Parallel Box Decoding / mask dispatch fields.
    block_size: int = 6
    causal_attn: bool = False
    text_mask_token_id: int = 151676
    null_token_id: int = 152678
    switch_token_id: int = 152679

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def _attn_implementation(self) -> str:
        """Compatibility with reference/HF config naming."""
        return self.attn_implementation

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Qwen2Config":
        """Build from a reference-style config dict, reconciling Magi to local SDPA."""
        data = dict(d)
        if "text_config" in data:
            data = dict(data["text_config"])
        if "_attn_implementation" in data and "attn_implementation" not in data:
            data["attn_implementation"] = data.pop("_attn_implementation")
        if data.get("attn_implementation") == "magi":
            data["attn_implementation"] = "sdpa"
        allowed = {name for name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})
