"""Merge a LoRA adapter into its base model and save a standalone checkpoint for vLLM.

The base is always loaded in `model.dtype` (BF16), never NF4, also for QLoRA adapters:
merging into quantized weights is not supported.

Usage:
    uv run --group train python scripts/merge_adapter.py --config ... --verify 20
    uv run --group train python scripts/merge_adapter.py --config ... --device cpu   # 26B-A4B

--verify N compares base + adapter with the merged model on N validation samples: the
maximum absolute difference of the next-token logits after the prompt, and the share
of identical greedy outputs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch

from llm_opt_lab.config import ExperimentConfig, load_config
from llm_opt_lab.data import Sample, filter_samples, load_samples
from llm_opt_lab.doc_types import load_doc_types
from llm_opt_lab.hf.loading import load_model, load_processor
from llm_opt_lab.hf.predictor import HFPredictor
from llm_opt_lab.paths import ModelSource, RunPaths, directory_size
from llm_opt_lab.results import record_metrics


@torch.inference_mode()
def snapshot(predictor: HFPredictor, samples: list[Sample]) -> list[tuple[torch.Tensor, str]]:
    """Next-token logits after each prompt, and the greedy output."""
    results = []
    for sample in samples:
        inputs = predictor.build_inputs(sample)
        logits = predictor.model(**inputs, logits_to_keep=1).logits[0, -1].float().cpu()
        results.append((logits, predictor.predict(sample).raw_output or ""))
    return results


def validation_samples(config: ExperimentConfig, n: int) -> list[Sample]:
    samples = filter_samples(
        load_samples(config.data.dir / "validation.jsonl"),
        config.data.doc_types,
        config.data.input_types,
    )
    return random.Random(config.seed).sample(samples, min(n, len(samples)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="defaults to the run's adapter")
    parser.add_argument("--out", type=Path, help="defaults to the run's merged path")
    parser.add_argument("--device", help="cuda:0 (default when available) or cpu")
    parser.add_argument("--verify", type=int, default=0, metavar="N")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = RunPaths.for_config(config)
    adapter, out = args.adapter or paths.adapter, args.out or paths.merged
    if out.exists() and not args.overwrite:
        parser.error(f"{out} exists; pass --overwrite to replace it")

    processor = load_processor(config)
    model = load_model(config, ModelSource(config.model.id, adapter), device=args.device)
    doc_types = load_doc_types(config.data.doc_types_dir)
    samples = validation_samples(config, args.verify) if args.verify else []

    def predictor(m: Any) -> HFPredictor:
        return HFPredictor(m, processor, config, doc_types, config.data.dir)

    before = snapshot(predictor(model), samples)
    merged = model.merge_and_unload()
    after = snapshot(predictor(merged), samples)

    updates: dict[str, Any] = {}
    if samples:
        diffs = [(a[0] - b[0]).abs().max().item() for a, b in zip(before, after, strict=True)]
        same = [a[1] == b[1] for a, b in zip(before, after, strict=True)]
        updates["merge_max_abs_logit_diff"] = max(diffs)
        updates["merge_identical_output_rate"] = sum(same) / len(same)
        print(f"max |logit diff| = {max(diffs):.4g}, identical outputs = {sum(same)}/{len(same)}")

    merged.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    updates["checkpoint_size_gb"] = directory_size(out) / 1024**3
    record_metrics(
        paths.metrics(),
        experiment_id=config.experiment_id,
        run_name=config.run_name,
        stage="merge",
        config_path=args.config,
        updates=updates,
    )
    print(f"merged model saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
