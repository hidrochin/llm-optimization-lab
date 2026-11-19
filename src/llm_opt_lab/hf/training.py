"""Helpers shared by SFT (`scripts/train.py`) and RL (`scripts/train_grpo.py`) training."""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig as PeftLoraConfig

from llm_opt_lab.config import ExperimentConfig, LoraConfig
from llm_opt_lab.data import Sample, filter_samples, load_samples
from llm_opt_lab.model_parts import NON_LANGUAGE_MODULES, is_non_language
from llm_opt_lab.paths import RunPaths
from llm_opt_lab.results import record_metrics

log = logging.getLogger(__name__)


def peft_config(lora: LoraConfig) -> PeftLoraConfig:
    if not lora.target_modules and not lora.target_parameters:
        raise SystemExit(
            "lora.target_modules is empty: take it from scripts/inspect_model.py and the "
            "gemma-skills template, never guess"
        )
    kwargs: dict[str, Any] = {
        "r": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": lora.target_modules,
    }
    exclude = lora.exclude_modules or (None if lora.train_vision_encoder else NON_LANGUAGE_MODULES)
    if exclude:
        kwargs["exclude_modules"] = exclude
    if lora.target_parameters:
        kwargs["target_parameters"] = lora.target_parameters
    return PeftLoraConfig(**kwargs)


def check_lora_placement(model: Any, lora: LoraConfig) -> None:
    placed = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    log.info("LoRA on %d modules, e.g. %s", len(placed), placed[:3])
    if not placed and not lora.target_parameters:
        raise SystemExit("no module received LoRA: check lora.target_modules")
    outside = [name for name in placed if is_non_language(name)]
    if outside and not lora.train_vision_encoder:
        raise SystemExit(f"LoRA reached non-language modules: {outside[:5]}")


def load_split(config: ExperimentConfig, split: str, limit: int | None) -> list[Sample]:
    samples = filter_samples(
        load_samples(config.data.dir / f"{split}.jsonl"),
        config.data.doc_types,
        config.data.input_types,
    )
    if limit is not None and limit < len(samples):
        samples = random.Random(config.seed).sample(samples, limit)
    return samples


def save_training_run(
    config: ExperimentConfig,
    config_path: Path,
    *,
    model: Any,
    processor: Any,
    log_history: list[dict[str, Any]],
    train_metrics: dict[str, Any],
    updates: dict[str, Any],
) -> None:
    """Save the adapter, processor, config copy and log history; record training metrics."""
    paths = RunPaths.for_config(config)
    model.save_pretrained(paths.adapter)
    processor.save_pretrained(paths.adapter)
    shutil.copy(config_path, paths.adapter / "experiment.yaml")
    paths.logs.mkdir(parents=True, exist_ok=True)
    with open(paths.logs / "train_log.jsonl", "w", encoding="utf-8") as f:
        for entry in log_history:
            f.write(json.dumps(entry) + "\n")

    record_metrics(
        paths.metrics(),
        experiment_id=config.experiment_id,
        run_name=config.run_name,
        stage="train",
        config_path=config_path,
        updates={
            "seed": config.seed,
            "max_seq_length": config.training.max_seq_length,
            "train_peak_vram_gb": (
                torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
            ),
            "train_time_s": train_metrics.get("train_runtime"),
            "train_steps_per_s": train_metrics.get("train_steps_per_second"),
            # Adapter weights only; the processor files saved next to them are not counted.
            "adapter_size_mb": sum(p.stat().st_size for p in paths.adapter.glob("adapter_model*"))
            / 1024**2,
            **updates,
        },
    )
    log.info("adapter saved to %s", paths.adapter)
