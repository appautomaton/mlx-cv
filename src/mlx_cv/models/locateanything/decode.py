"""Parse LocateAnything's PBD output into structured boxes / points (§16.2).

Grounded in the reference PBD decoder's frame grammar (``handle_pattern``):

    ref:    ref_start, <label tokens...>, ref_end
    box:    box_start, c1, c2, c3, c4, box_end      (4 coords)
    point:  box_start, c1, c2, box_end              (2 coords)
    empty:  box_start, none_id, box_end             (queried object absent)

A coordinate token's value is ``token_id - coord_start`` in ``[0, coord_range]``
(default 1000). This module is **pure** (token / text space). Mapping label ids ->
text (tokenizer) and normalized coords -> pixels (``SpatialTransform`` ctx) happens
in the processor (Stage 3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["TokenScheme", "GroundingItem", "parse_grounding_tokens", "parse_grounding_text"]


@dataclass
class TokenScheme:
    """The grounding token ids the parser keys on (defaults = verified values)."""

    box_start: int = 151668
    box_end: int = 151669
    coord_start: int = 151677
    coord_end: int = 152677
    ref_start: int = 151672
    ref_end: int = 151673
    none_id: int = 4064
    coord_range: int = 1000

    @classmethod
    def from_config(cls, cfg) -> "TokenScheme":
        return cls(
            box_start=cfg.box_start_token_id,
            box_end=cfg.box_end_token_id,
            coord_start=cfg.coord_start_token_id,
            coord_end=cfg.coord_end_token_id,
            ref_start=cfg.ref_start_token_id,
            ref_end=cfg.ref_end_token_id,
            none_id=cfg.none_token_id,
        )


@dataclass
class GroundingItem:
    """One parsed instance. ``label`` is token ids (token parser) or text (text parser)."""

    kind: str                      # "box" | "point"
    coords: list[int]              # normalized [0, coord_range]; len 4 (box) or 2 (point)
    label: object = None           # list[int] | str | None


def parse_grounding_tokens(token_ids, scheme: TokenScheme | None = None) -> list[GroundingItem]:
    """Parse a PBD token-id stream into ordered :class:`GroundingItem` s.

    A ``ref`` label persists across the boxes that follow it (multi-instance grounding)
    until the next ``ref``.
    """
    s = scheme or TokenScheme()
    toks = list(token_ids)
    n = len(toks)
    items: list[GroundingItem] = []
    label: object = None
    i = 0
    while i < n:
        t = toks[i]
        if t == s.ref_start:
            j = i + 1
            buf: list[int] = []
            while j < n and toks[j] not in (s.ref_end, s.box_start):
                buf.append(toks[j])
                j += 1
            label = buf
            i = j + 1 if (j < n and toks[j] == s.ref_end) else j
            continue
        if t == s.box_start:
            j = i + 1
            inner: list[int] = []
            while j < n and toks[j] != s.box_end:
                inner.append(toks[j])
                j += 1
            i = j + 1 if (j < n and toks[j] == s.box_end) else j
            coords = [c - s.coord_start for c in inner if s.coord_start <= c <= s.coord_end]
            if len(coords) >= 4:
                items.append(GroundingItem("box", coords[:4], label))
            elif len(coords) == 2:
                items.append(GroundingItem("point", coords[:2], label))
            # 0 coords (none_id / absent) or malformed -> skip
            continue
        i += 1
    return items


_REF_OR_BOX = re.compile(r"<ref>(.*?)</ref>|<box>((?:<\d+>)+)</box>", re.S)
_NUM = re.compile(r"<(\d+)>")


def parse_grounding_text(text: str) -> list[GroundingItem]:
    """Parse the model's text form ``<ref>label</ref><box><x1>..</box>`` (HF card form)."""
    items: list[GroundingItem] = []
    label: object = None
    for m in _REF_OR_BOX.finditer(text):
        if m.group(1) is not None:
            label = m.group(1).strip()
        else:
            nums = [int(x) for x in _NUM.findall(m.group(2))]
            if len(nums) >= 4:
                items.append(GroundingItem("box", nums[:4], label))
            elif len(nums) == 2:
                items.append(GroundingItem("point", nums[:2], label))
    return items
