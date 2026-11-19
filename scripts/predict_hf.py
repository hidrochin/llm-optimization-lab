"""Generate predictions with a Hugging Face model, one sample at a time.

Usage:
    uv run --group train python scripts/predict_hf.py --config configs/dev/cord_e2b_base.yaml
    uv run --group train python scripts/predict_hf.py --config ... --source adapter
    uv run --group train python scripts/predict_hf.py --config ... --source merged:E1_x

--source: `base` (default), `adapter` (base + this run's adapter), `merged`, or
`<kind>:<run_id>` / a path; see llm_opt_lab.paths.resolve_model_source. With
`method.name: qlora`, the hub base model is loaded in NF4 unless --no-quantize.

Writes artifacts/predictions/<run_id>/<split>.jsonl. The file is appended to, so an
interrupted run resumes where it stopped. HF latency and peak VRAM go to the metrics file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from llm_opt_lab.config import load_config
from llm_opt_lab.data import filter_samples, load_samples
from llm_opt_lab.doc_types import load_doc_types
from llm_opt_lab.hf.loading import load_model, load_processor
from llm_opt_lab.hf.predictor import HFPredictor
from llm_opt_lab.paths import RunPaths, resolve_model_source
from llm_opt_lab.predictions import read_predictions
from llm_opt_lab.profiling import latency_summary
from llm_opt_lab.results import record_metrics, save_gpu_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", help="defaults to evaluation.split")
    parser.add_argument("--source", default="base")
    parser.add_argument("--no-quantize", action="store_true", help="load a QLoRA base in BF16")
    parser.add_argument("--limit", type=int, help="first N samples only (smoke tests)")
    parser.add_argument("--out", type=Path, help="defaults to the run's predictions path")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    split = args.split or config.evaluation.split
    paths = RunPaths.for_config(config)
    out = args.out or paths.predictions(split)

    samples = filter_samples(
        load_samples(config.data.dir / f"{split}.jsonl"),
        config.data.doc_types,
        config.data.input_types,
    )[: args.limit]
    done = read_predictions(out) if out.exists() else {}
    todo = [s for s in samples if s.id not in done]
    print(f"{len(samples)} samples, {len(done)} already predicted, {len(todo)} to go -> {out}")
    if not todo:
        return 0

    source = resolve_model_source(config, args.source)
    quantize = (
        config.method.name == "qlora" and source.model == config.model.id and not args.no_quantize
    )
    processor = load_processor(config, source.model)
    model = load_model(config, source, quantize=quantize)
    predictor = HFPredictor(
        model, processor, config, load_doc_types(config.data.doc_types_dir), config.data.dir
    )

    save_gpu_snapshot(paths.logs / f"nvidia-smi_predict_hf_{split}.txt")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    new = []
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with open(out, "a", encoding="utf-8") as f:
        for i, sample in enumerate(todo, start=1):
            prediction = predictor.predict(sample)
            f.write(prediction.model_dump_json() + "\n")
            f.flush()
            new.append(prediction)
            if i % 20 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - start)
                errors = sum(p.error is not None for p in new)
                print(f"  {i}/{len(todo)}  {rate:.2f} samples/s  errors={errors}")

    timed = [p for p in new if p.latency_ms is not None]
    latency = latency_summary([p.latency_ms for p in timed]) if timed else None
    generated_s = sum(p.latency_ms for p in timed) / 1000
    record_metrics(
        paths.metrics(split),
        experiment_id=config.experiment_id,
        run_name=config.run_name,
        stage="predict_hf",
        config_path=args.config,
        updates={
            "infer_peak_vram_gb": (
                torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
            ),
            "hf_latency_ms_p50": latency["p50"] if latency else None,
            "hf_latency_ms_p95": latency["p95"] if latency else None,
            "hf_tokens_per_s": (
                sum(p.completion_tokens or 0 for p in timed) / generated_s if generated_s else None
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
