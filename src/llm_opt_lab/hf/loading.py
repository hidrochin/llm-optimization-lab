"""Load processors and models (BF16, NF4, with or without a LoRA adapter) from a config."""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from llm_opt_lab.config import ExperimentConfig, MethodConfig
from llm_opt_lab.paths import ModelSource

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def is_text_only(config: ExperimentConfig) -> bool:
    return config.model.architecture == "causal_lm"


def _revision(config: ExperimentConfig, model: str) -> str | None:
    # The revision pins the hub base model only, never a local merged checkpoint.
    return config.model.revision if model == config.model.id else None


def load_processor(config: ExperimentConfig, model: str | None = None) -> Any:
    """AutoProcessor for image-text models, AutoTokenizer for text-only models."""
    model = model or config.model.id
    cls = AutoTokenizer if is_text_only(config) else AutoProcessor
    return cls.from_pretrained(model, revision=_revision(config, model))


def bnb_config(method: MethodConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=method.quantization,
        bnb_4bit_use_double_quant=method.double_quant,
        bnb_4bit_compute_dtype=DTYPES[method.compute_dtype],
    )


def load_model(
    config: ExperimentConfig,
    source: ModelSource,
    *,
    quantize: bool = False,
    device: str | None = None,
) -> Any:
    """Load `source.model` on one device; NF4 when `quantize`; adapter on top if given."""
    cls = AutoModelForCausalLM if is_text_only(config) else AutoModelForImageTextToText
    kwargs: dict[str, Any] = {
        "dtype": DTYPES[config.model.dtype],
        "device_map": {"": device or default_device()},
        "revision": _revision(config, source.model),
    }
    if config.model.attn_implementation:
        kwargs["attn_implementation"] = config.model.attn_implementation
    if quantize:
        if config.method.name != "qlora":
            raise ValueError(
                "quantize=True needs method.name 'qlora' and its quantization settings"
            )
        kwargs["quantization_config"] = bnb_config(config.method)
    model = cls.from_pretrained(source.model, **kwargs)
    if source.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(source.adapter))
    return model
