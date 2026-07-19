"""Self-contained LocateAnything model, processor, and tokenizer pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from ...hub import resolve_pretrained
from .config import LocateAnythingConfig
from .convert import load_locateanything_weights
from .modeling import LocateAnythingModel
from .processor import LocateAnythingProcessor
from .tokenizer import LocateAnythingTokenizer

__all__ = ["LocateAnythingPipeline"]


class LocateAnythingPipeline:
    def __init__(self, model, processor, tokenizer, *, package_path: Path) -> None:
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.package_path = package_path

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool | None = None,
        token: str | bool | None = None,
    ) -> "LocateAnythingPipeline":
        package = resolve_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        if not package.is_dir():
            raise ValueError("LocateAnything from_pretrained requires a package directory")
        config_path = package / "config.json"
        weights_path = package / "model.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(f"LocateAnything package is missing config.json: {package}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"LocateAnything package is missing model.safetensors: {package}")
        config = LocateAnythingConfig.from_dict(json.loads(config_path.read_text()))
        tokenizer = LocateAnythingTokenizer.from_pretrained(package)
        model = load_locateanything_weights(LocateAnythingModel(config), weights_path)
        processor = LocateAnythingProcessor(config, tokenizer=tokenizer)
        return cls(model, processor, tokenizer, package_path=package)

    def predict(self, image, prompt: str, **kwargs):
        return self.model.predict(
            image,
            self.format_prompt(prompt),
            processor=self.processor,
            **kwargs,
        )

    @staticmethod
    def format_prompt(prompt: str) -> str:
        """Apply the upstream single-image chat template deterministically."""

        question = prompt.replace("<image-0>", "").replace("<image-1>", "").strip()
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<image-1>"
            f"{question}<|im_end|>\n<|im_start|>assistant\n"
        )
