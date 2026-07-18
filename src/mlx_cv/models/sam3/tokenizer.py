"""Runtime-light SAM 3.1 CLIP-style byte-BPE tokenizer."""

from __future__ import annotations

import gzip
import html
import re
import string
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable

import numpy as np

__all__ = [
    "DEFAULT_CONTEXT_LENGTH",
    "SAM3Tokenizer",
    "bytes_to_unicode",
    "canonicalize_text",
    "default_bpe_path",
]


DEFAULT_CONTEXT_LENGTH = 32


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte-to-unicode table used by CLIP byte BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs = set()
    prev = word[0]
    for char in word[1:]:
        pairs.add((prev, char))
        prev = char
    return pairs


def canonicalize_text(text: str, *, keep_punctuation_exact_string: str | None = None) -> str:
    """Lowercase text and remove punctuation following the SAM3/CLIP cleaning path."""
    text = html.unescape(html.unescape(text)).replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def default_bpe_path() -> Path:
    """Path to the committed reduced SAM3 BPE merge asset."""
    return Path(resources.files(__package__).joinpath("assets", "bpe_simple_vocab_tiny.txt"))


def _read_merges(path: str | Path) -> list[tuple[str, str]]:
    path = Path(path)
    if path.suffix == ".gz":
        text = gzip.open(path, "rb").read().decode("utf-8")
    else:
        text = path.read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and "#version" in lines[0]:
        lines = lines[1:]
    merges: list[tuple[str, str]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid BPE merge line in {path}: {line!r}")
        merges.append((parts[0], parts[1]))
    # The official OpenAI CLIP/SAM 3 vocabulary intentionally uses only the
    # first 48,894 merges from the larger compressed source asset.
    return merges[: 49152 - 256 - 2]


class SAM3Tokenizer:
    """CLIP-style byte-BPE tokenizer returning padded numpy token ids."""

    def __init__(
        self,
        bpe_path: str | Path | None = None,
        *,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        additional_special_tokens: Iterable[str] | None = None,
        clean: str = "canonicalize",
    ) -> None:
        if context_length <= 1:
            raise ValueError("SAM3Tokenizer context_length must be greater than 1")
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.context_length = int(context_length)
        self.clean = clean

        merges = _read_merges(default_bpe_path() if bpe_path is None else bpe_path)
        vocab = list(bytes_to_unicode().values())
        vocab += [v + "</w>" for v in vocab]
        vocab += ["".join(merge) for merge in merges]
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        if additional_special_tokens is not None:
            special_tokens += list(additional_special_tokens)
        vocab += special_tokens

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {token: token for token in special_tokens}
        special = "|".join(re.escape(token) for token in special_tokens)
        token_pattern = r"'s|'t|'re|'ve|'m|'ll|'d|[A-Za-z]+|\d|[^\sA-Za-z\d]+"
        self.pat = re.compile(f"{special}|{token_pattern}", re.IGNORECASE)
        self.vocab_size = len(self.encoder)
        self.sot_token_id = self.encoder["<start_of_text>"]
        self.eot_token_id = self.encoder["<end_of_text>"]

    def _clean(self, text: str) -> str:
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text).strip()
        if self.clean == "whitespace":
            return text
        if self.clean == "lower":
            return text.lower()
        if self.clean == "canonicalize":
            return canonicalize_text(text)
        raise ValueError(f"unknown SAM3 tokenizer clean mode {self.clean!r}")

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        bpe_tokens: list[int] = []
        for token in re.findall(self.pat, self._clean(text)):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[piece] for piece in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens: Iterable[int]) -> str:
        text = "".join(self.decoder[int(token)] for token in tokens if int(token) in self.decoder)
        return bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace").replace("</w>", " ")

    def __call__(self, texts: str | list[str] | tuple[str, ...], context_length: int | None = None) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        context_length = int(context_length or self.context_length)
        result = np.zeros((len(texts), context_length), dtype=np.int64)
        for row, text in enumerate(texts):
            tokens = [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            result[row, : len(tokens)] = np.asarray(tokens, dtype=np.int64)
        return result
