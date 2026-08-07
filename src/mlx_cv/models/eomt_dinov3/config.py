"""EoMT-DINOv3 universal-segmentation configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ...backbones.vision.dinov3 import DINOv3Config

__all__ = ["EoMTDINOv3Config"]


@dataclass(frozen=True)
class EoMTDINOv3Config:
    """Architecture and public panoptic contract for EoMT-small at 640 px."""

    backbone: DINOv3Config = field(
        default_factory=lambda: DINOv3Config(
            embed_dim=384,
            depth=12,
            num_heads=6,
            patch_size=16,
            in_chans=3,
            n_storage_tokens=4,
            ffn_ratio=4.0,
            qkv_bias=True,
            layer_norm_eps=1e-5,
            rope_base=100.0,
            layerscale_init=1.0,
        )
    )
    image_size: int = 640
    num_classes: int = 133
    num_queries: int = 200
    num_blocks: int = 3
    num_upscale_blocks: int = 2
    stuff_classes: tuple[int, ...] = tuple(range(80, 133))
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.image_size % self.backbone.patch_size:
            raise ValueError("EoMT image_size must be positive and divisible by the patch size")
        if self.num_classes <= 0 or self.num_queries <= 0:
            raise ValueError("EoMT num_classes and num_queries must be positive")
        if not 1 <= self.num_blocks <= self.backbone.depth:
            raise ValueError("EoMT num_blocks must be between 1 and backbone.depth")
        if self.num_upscale_blocks <= 0:
            raise ValueError("EoMT num_upscale_blocks must be positive")
        if any(class_id < 0 or class_id >= self.num_classes for class_id in self.stuff_classes):
            raise ValueError("EoMT stuff class IDs must be inside the configured class range")
        if self.labels is not None and len(self.labels) != self.num_classes:
            raise ValueError(
                f"EoMT labels has length {len(self.labels)}, expected {self.num_classes}"
            )

    @classmethod
    def coco_panoptic_small_640(cls) -> "EoMTDINOv3Config":
        return cls()

    @classmethod
    def tiny_fixture(cls) -> "EoMTDINOv3Config":
        return cls(
            backbone=DINOv3Config(
                embed_dim=16,
                depth=3,
                num_heads=2,
                patch_size=4,
                in_chans=3,
                n_storage_tokens=2,
                ffn_ratio=2.0,
                layer_norm_eps=1e-5,
                layerscale_init=1.0,
            ),
            image_size=8,
            num_classes=4,
            num_queries=3,
            num_blocks=2,
            num_upscale_blocks=1,
            stuff_classes=(2, 3),
            labels=("zero", "one", "two", "three"),
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "EoMTDINOv3Config":
        """Accept normalized local configs and the official Transformers config."""

        if not isinstance(payload, dict):
            raise TypeError("EoMT config must be a dictionary")
        data = payload.get("config", payload)
        if not isinstance(data, dict):
            raise ValueError("EoMT config.config must be a dictionary")

        architecture_keys = {
            "backbone",
            "embed_dim",
            "hidden_size",
            "depth",
            "num_hidden_layers",
        }
        if not architecture_keys.intersection(data):
            base = cls.coco_panoptic_small_640()
            return cls(
                backbone=base.backbone,
                image_size=int(data.get("image_size", base.image_size)),
                num_classes=int(data.get("num_classes", base.num_classes)),
                num_queries=int(data.get("num_queries", base.num_queries)),
                num_blocks=int(data.get("num_blocks", base.num_blocks)),
                num_upscale_blocks=int(
                    data.get("num_upscale_blocks", base.num_upscale_blocks)
                ),
                stuff_classes=tuple(data.get("stuff_classes", base.stuff_classes)),
                labels=None
                if data.get("labels") is None
                else tuple(str(label) for label in data["labels"]),
            )

        backbone_payload = data.get("backbone")
        if backbone_payload is None:
            backbone_payload = data
        backbone = (
            backbone_payload
            if isinstance(backbone_payload, DINOv3Config)
            else DINOv3Config.from_dict(backbone_payload)
        )

        id2label = data.get("id2label")
        labels = data.get("labels")
        if labels is None and isinstance(id2label, dict) and id2label:
            labels = tuple(
                str(id2label[str(i)] if str(i) in id2label else id2label[i])
                for i in range(len(id2label))
            )
        elif labels is not None:
            labels = tuple(str(label) for label in labels)

        num_classes = int(
            data.get(
                "num_classes",
                data.get("num_labels", len(labels) if labels is not None else 133),
            )
        )
        default_stuff = tuple(range(80, num_classes)) if num_classes >= 133 else ()
        return cls(
            backbone=backbone,
            image_size=int(data.get("image_size", 640)),
            num_classes=num_classes,
            num_queries=int(data.get("num_queries", 200)),
            num_blocks=int(data.get("num_blocks", 3)),
            num_upscale_blocks=int(data.get("num_upscale_blocks", 2)),
            stuff_classes=tuple(int(i) for i in data.get("stuff_classes", default_stuff)),
            labels=labels,
        )

    def to_dict(self) -> dict:
        """Return the normalized JSON-serializable architecture configuration."""

        return asdict(self)
