"""W4A16 post-training quantization with LLM Compressor, saved as compressed-tensors.

Only the language model is quantized; `quantization.ignore` keeps the vision/audio
towers and lm_head in BF16. Calibration uses the `calibration` split, built with the
same chat template as training (prompt + answer), with real lengths. Each sample asks for
the inference fields, or with `quantization.calibration_requests: sampled` for a random
training request, so the field-list reading learned in SFT is calibrated too.

Runs in .venv-quant. Written against the LLM Compressor multimodal examples; check it
against the installed version in T-D02.

Usage:
    python quantization/quantize_w4a16.py --config configs/dev/cord_e2b_w4a16.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from llm_opt_lab.config import ExperimentConfig, load_config
from llm_opt_lab.data import filter_samples, load_samples
from llm_opt_lab.doc_types import load_doc_types
from llm_opt_lab.field_requests import inference_request, training_request
from llm_opt_lab.hf.loading import DTYPES, is_text_only, load_processor
from llm_opt_lab.hf.sft_data import build_features
from llm_opt_lab.paths import RunPaths, directory_size, resolve_model_source
from llm_opt_lab.results import record_metrics, save_gpu_snapshot


def calibration_input_types(config: ExperimentConfig) -> list[str]:
    q = config.quantization
    input_types = q.calibration_input_types or config.data.input_types
    if q.pipeline == "sequential" and len(set(input_types)) > 1:
        # The sequential pipeline traces the model once; a sample without pixel_values then
        # fails in the traced vision subgraph.
        raise SystemExit(
            "pipeline 'sequential' cannot mix image and spatial-text calibration samples: "
            "set quantization.calibration_input_types to one input type, or use pipeline 'basic'"
        )
    return input_types


def calibration_dataset(config: ExperimentConfig, processor: Any) -> Dataset:
    q = config.quantization
    samples = filter_samples(
        load_samples(config.data.dir / f"{q.calibration_split}.jsonl"),
        config.data.doc_types,
        calibration_input_types(config),
    )
    random.Random(config.seed).shuffle(samples)
    doc_types = load_doc_types(config.data.doc_types_dir)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        doc_type = doc_types[sample.doc_type]
        if q.calibration_requests == "sampled":
            rng = random.Random(f"{config.seed}:{sample.id}:calibration")
            request = training_request(sample.target, doc_type, config.fields, rng)
        else:
            request = inference_request(doc_type, config.fields)
        features = build_features(
            sample,
            doc_type,
            processor,
            config.data.dir,
            text_only=is_text_only(config),
            chat_template_kwargs=config.model.chat_template_kwargs,
            request=request,
        )
        if features["input_ids"].shape[0] > q.max_seq_length:
            continue
        features.pop("labels")
        rows.append(
            {
                k: v.float().numpy() if v.is_floating_point() else v.numpy()
                for k, v in features.items()
            }
        )
        if len(rows) == q.num_calibration_samples:
            break
    if not rows:
        raise SystemExit("no calibration sample fits quantization.max_seq_length")
    keys = sorted(set().union(*rows))  # text-only rows have no pixel_values
    print(f"{len(rows)} calibration samples")
    return Dataset.from_list([{k: row.get(k) for k in keys} for row in rows]).with_format("torch")


def make_collator(dtype: torch.dtype):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(batch) != 1:
            raise ValueError("calibration runs one sample at a time")
        out = {}
        for key, value in batch[0].items():
            if value is None:
                continue
            tensor = torch.as_tensor(value)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype)  # pixel values
            elif tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)  # token sequences get a batch dimension
            out[key] = tensor
        return out

    return collate


def build_recipe(config: ExperimentConfig) -> Any:
    q = config.quantization
    if q.algorithm == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier

        return GPTQModifier(targets="Linear", scheme=q.scheme, ignore=q.ignore)
    from llmcompressor.modifiers.awq import AWQModifier

    return AWQModifier(targets=["Linear"], scheme=q.scheme, ignore=q.ignore)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="defaults to the run's quantized path")
    args = parser.parse_args(argv)

    try:
        from llmcompressor import oneshot
    except ImportError:  # older releases
        from llmcompressor.transformers import oneshot

    config = load_config(args.config)
    if config.quantization is None:
        parser.error("config has no quantization section")
    q, paths = config.quantization, RunPaths.for_config(config)
    out = args.out or paths.quantized
    source = resolve_model_source(config, q.source)
    if source.adapter is not None:
        parser.error(
            "merge the adapter first (scripts/merge_adapter.py), then quantize the merged model"
        )

    processor = load_processor(config, source.model)
    dataset = calibration_dataset(config, processor)
    cls = AutoModelForCausalLM if is_text_only(config) else AutoModelForImageTextToText
    # basic: the whole model runs forward, so it goes on the GPU. sequential: loaded on CPU,
    # LLM Compressor moves one decoder layer at a time to the GPU.
    on_gpu = q.pipeline == "basic" and torch.cuda.is_available()
    model = cls.from_pretrained(
        source.model,
        dtype=DTYPES[config.model.dtype],
        device_map={"": "cuda:0"} if on_gpu else None,
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "recipe": build_recipe(config),
        "pipeline": q.pipeline,
        "max_seq_length": q.max_seq_length,
        "num_calibration_samples": len(dataset),
        "data_collator": make_collator(DTYPES[config.model.dtype]),
    }
    if q.sequential_targets:
        kwargs["sequential_targets"] = q.sequential_targets
    save_gpu_snapshot(paths.logs / "nvidia-smi_quantize.txt")
    oneshot(**kwargs)

    model.save_pretrained(out, save_compressed=True)
    processor.save_pretrained(out)
    record_metrics(
        paths.metrics(),
        experiment_id=config.experiment_id,
        run_name=config.run_name,
        stage="quantize",
        config_path=args.config,
        updates={"checkpoint_size_gb": directory_size(out) / 1024**3},
    )
    print(f"quantized model saved to {out}; check it with quantization/verify_quantized.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
