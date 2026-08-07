"""Reusable vision and language backbones.

Two kinds: ``vision/`` encoders (image -> multi-scale features) and ``llm/`` decoders
(embeds -> hidden states + decode loop). Port once, reuse across models.
"""
