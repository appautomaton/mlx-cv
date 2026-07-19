"""Thin adapter around the optional Hugging Face tokenizers runtime."""

from __future__ import annotations

from pathlib import Path

__all__ = ["LocateAnythingTokenizer"]


class LocateAnythingTokenizer:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.unk_token_id = tokenizer.token_to_id("<|endoftext|>")

    @classmethod
    def from_pretrained(cls, package_path: str | Path) -> "LocateAnythingTokenizer":
        path = Path(package_path) / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"LocateAnything package is missing tokenizer.json: {path}"
            )
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "LocateAnything tokenization requires `pip install \"mlx-cv[hub]\"`"
            ) from exc
        return cls(Tokenizer.from_file(str(path)))

    def __call__(self, texts, *, padding: bool = True):
        if isinstance(texts, str):
            texts = [texts]
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=True)
        ids = [encoding.ids for encoding in encodings]
        if padding:
            width = max((len(row) for row in ids), default=0)
            pad_id = self._tokenizer.token_to_id("<|endoftext|>") or 0
            masks = [[1] * len(row) + [0] * (width - len(row)) for row in ids]
            ids = [row + [pad_id] * (width - len(row)) for row in ids]
        else:
            masks = [[1] * len(row) for row in ids]
        return {"input_ids": ids, "attention_mask": masks}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(
            [int(token) for token in token_ids],
            skip_special_tokens=skip_special_tokens,
        )
