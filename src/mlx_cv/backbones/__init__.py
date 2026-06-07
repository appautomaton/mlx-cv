"""Reusable backbones, registered and shared (§5.3, §16).

Two kinds: ``vision/`` encoders (image -> multi-scale features) and ``llm/`` decoders
(embeds -> hidden states + decode loop). Port once, reuse across models.
"""
