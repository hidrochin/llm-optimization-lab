"""Fine-tune with LoRA or QLoRA (`method.name` in the config) and save the adapter.

Usage:
    uv run --group train python scripts/train.py --config configs/dev/cord_e2b_lora_smoke.yaml
    uv run --group train python scripts/train.py --config ... --check-loss 4   # no training

Writes the adapter to artifacts/adapters/<run_id>/, trainer checkpoints to
artifacts/checkpoints/<run_id>/, the log history to results/logs/<run_id>/train_log.jsonl,
and training metrics into results/metrics/<run_id>.json.

Each training sample asks for the fields of a training request: every annotated field
(minus `fields.holdout`), or a random subset per draw with `fields.sampling`
(docs/formats.md §3).

The default loss (`training.loss: label_positions`) applies lm_head only at positions whose
next token is part of the answer, which avoids materializing 32K x 262K logits (README,
Risk 1). `--check-loss N` compares it with the full-logits loss on N training samples.
"""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, set_seed

from llm_opt_lab.config import ExperimentConfig, load_config
from llm_opt_lab.doc_types import load_doc_types
from llm_opt_lab.hf.loading import is_text_only, load_model, load_processor
from llm_opt_lab.hf.sft_data import IGNORE_INDEX, SFTCollator, SFTDataset
from llm_opt_lab.hf.training import (
    check_lora_placement,
    load_split,
    peft_config,
    save_training_run,
)
from llm_opt_lab.paths import ModelSource, RunPaths
from llm_opt_lab.results import save_gpu_snapshot

log = logging.getLogger("train")


def token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over answer tokens, computed in FP32."""
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX
    )


def compute_sft_loss(
    model: Any, inputs: dict[str, torch.Tensor], mode: str
) -> tuple[torch.Tensor, Any]:
    labels = inputs["labels"]
    model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
    if mode == "label_positions":
        # Position t predicts token t+1: keep only positions whose next token is an answer token.
        positions = (labels[:, 1:] != IGNORE_INDEX).any(dim=0).nonzero().squeeze(-1)
        outputs = model(**model_inputs, logits_to_keep=positions)
        return token_loss(outputs.logits, labels[:, positions + 1]), outputs
    outputs = model(**model_inputs)
    return token_loss(outputs.logits[:, :-1], labels[:, 1:]), outputs


class ExtractionTrainer(Trainer):
    def __init__(self, *args: Any, loss_mode: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_mode = loss_mode
        # compute_loss returns a per-batch mean, so Trainer must divide by the accumulation steps.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ANN001
        loss, outputs = compute_sft_loss(model, inputs, self.loss_mode)
        return (loss, outputs) if return_outputs else loss


def check_logits_to_keep(model: Any) -> None:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if "logits_to_keep" not in inspect.signature(base.forward).parameters:
        raise SystemExit(
            f"{type(base).__name__}.forward has no logits_to_keep; set training.loss: full"
        )


def training_arguments(config: ExperimentConfig, paths: RunPaths) -> TrainingArguments:
    tc = config.training
    kwargs: dict[str, Any] = {
        "output_dir": str(paths.checkpoints),
        "per_device_train_batch_size": tc.batch_size,
        "per_device_eval_batch_size": tc.batch_size,
        "gradient_accumulation_steps": tc.gradient_accumulation_steps,
        "num_train_epochs": tc.epochs,
        "max_steps": tc.max_steps or -1,
        "learning_rate": tc.learning_rate,
        "lr_scheduler_type": tc.lr_scheduler,
        "warmup_steps": tc.warmup_ratio,  # transformers 5: a float in [0, 1) is a ratio
        "weight_decay": tc.weight_decay,
        "max_grad_norm": tc.max_grad_norm,
        "bf16": config.model.dtype == "bfloat16",
        "fp16": config.model.dtype == "float16",
        "gradient_checkpointing": tc.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": tc.logging_steps,
        "eval_strategy": "steps" if tc.eval_steps else "no",
        "save_strategy": "steps" if tc.save_steps else "no",
        "prediction_loss_only": True,  # never accumulate eval logits
        "remove_unused_columns": False,
        "report_to": "none",  # private data: no external experiment trackers
        "seed": config.seed,
    }
    if tc.eval_steps:
        kwargs["eval_steps"] = tc.eval_steps
    if tc.save_steps:
        kwargs["save_steps"] = tc.save_steps
    return TrainingArguments(**kwargs)


@torch.no_grad()
def check_loss(model: Any, dataset: SFTDataset, collator: SFTCollator, n: int) -> None:
    """Both loss modes must agree: label_positions only skips logits that are never scored."""
    model.eval()
    for i in range(min(n, len(dataset))):
        batch = {k: v.to(model.device) for k, v in collator([dataset[i]]).items()}
        fast, _ = compute_sft_loss(model, batch, "label_positions")
        full, _ = compute_sft_loss(model, batch, "full")
        print(f"{dataset.samples[i].id}: label_positions={fast.item():.6f} full={full.item():.6f} "
              f"diff={abs(fast.item() - full.item()):.2e}")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="resume from the last checkpoint")
    parser.add_argument("--check-loss", type=int, metavar="N", help="compare losses, do not train")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    for name in ("train", "llm_opt_lab"):  # INFO for our code only, not every HTTP request
        logging.getLogger(name).setLevel(logging.INFO)

    config = load_config(args.config)
    if config.method.name not in ("lora", "qlora"):
        parser.error("method.name must be lora or qlora")
    tc, paths = config.training, RunPaths.for_config(config)
    set_seed(config.seed)

    processor = load_processor(config)
    dataset_kwargs: dict[str, Any] = {
        "doc_types": load_doc_types(config.data.doc_types_dir, config.data.doc_types or None),
        "processor": processor,
        "data_dir": config.data.dir,
        "text_only": is_text_only(config),
        "chat_template_kwargs": config.model.chat_template_kwargs,
        "max_seq_length": tc.max_seq_length,
        "over_length": tc.over_length,
        "seed": config.seed,
    }
    train_set = SFTDataset(
        load_split(config, "train", tc.max_train_samples), fields=config.fields, **dataset_kwargs
    )
    # Eval loss always uses the fixed request (every annotated, non-held-out field).
    fixed = config.fields.model_copy(update={"sampling": None})
    eval_set = (
        SFTDataset(
            load_split(config, "validation", tc.eval_samples), fields=fixed, **dataset_kwargs
        )
        if tc.eval_steps
        else None
    )

    qlora = config.method.name == "qlora"
    model = load_model(config, ModelSource(config.model.id), quantize=qlora)
    if qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=tc.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model = get_peft_model(model, peft_config(config.lora))
    check_lora_placement(model, config.lora)
    if tc.loss == "label_positions":
        check_logits_to_keep(model)
    trainable, total = model.get_nb_trainable_parameters()
    log.info("trainable parameters: %d / %d (%.3f%%)", trainable, total, 100 * trainable / total)

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    collator = SFTCollator(pad_id)
    if args.check_loss:
        check_loss(model, train_set, collator, args.check_loss)
        return 0

    trainer = ExtractionTrainer(
        model=model,
        args=training_arguments(config, paths),
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=collator,
        loss_mode=tc.loss,
    )
    save_gpu_snapshot(paths.logs / "nvidia-smi_train.txt")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)

    save_training_run(
        config,
        args.config,
        model=model,
        processor=processor,
        log_history=trainer.state.log_history,
        train_metrics=result.metrics,
        updates={
            "trainable_params": trainable,
            "n_train_samples": len(train_set),
            "n_dropped_over_length": len(train_set.dropped),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
