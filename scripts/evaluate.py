"""Score a predictions file and record the run's quality metrics.

Usage:
    uv run python scripts/evaluate.py --config configs/gemma4_e2b/baseline.yaml
    uv run python scripts/evaluate.py --config ... --split validation --predictions preds.jsonl

Reads   <data.dir>/<split>.jsonl and artifacts/predictions/<run_id>/<split>.jsonl
Writes  results/metrics/<run_id>.json (merged) and artifacts/eval/<run_id>/<split>/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_opt_lab.config import load_config
from llm_opt_lab.data import filter_samples, load_samples
from llm_opt_lab.doc_types import load_doc_types
from llm_opt_lab.evaluation import EvalReport, evaluate_predictions, write_eval_details
from llm_opt_lab.metrics import QualityStats
from llm_opt_lab.paths import RunPaths
from llm_opt_lab.predictions import read_predictions
from llm_opt_lab.results import record_metrics


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _row(name: str, stats: QualityStats) -> str:
    return (
        f"{name:<24}{stats.n:>7}  {_fmt(stats.exact_match):>8}  {_fmt(stats.field_accuracy):>8}"
        f"  {_fmt(stats.ned_mean):>8}  {_fmt(stats.parse_rate):>8}"
    )


def print_report(title: str, report: EvalReport, met: bool, n_worst: int = 10) -> None:
    print(title)
    print(f"{'':<24}{'n':>7}  {'exact':>8}  {'field':>8}  {'ned':>8}  {'parse':>8}")
    print(_row("overall", report.overall))
    for name, stats in {**report.by_input_type, **report.by_doc_type}.items():
        print(_row(name, stats))
    if report.n_request_errors:
        print(f"request errors: {report.n_request_errors}")
    print(f"quality target: {'MET' if met else 'NOT MET'}")
    print(f"lowest-accuracy fields (of {len(report.per_field)}):")
    for field in report.per_field[:n_worst]:
        print(f"  {field.accuracy:.4f}  {field.doc_type}/{field.field}  (n={field.n})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", help="defaults to evaluation.split in the config")
    parser.add_argument("--predictions", type=Path, help="defaults to the run's standard path")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    split = args.split or config.evaluation.split
    paths = RunPaths.for_config(config)
    predictions_path = args.predictions or paths.predictions(split)

    samples = filter_samples(
        load_samples(config.data.dir / f"{split}.jsonl"),
        config.data.doc_types,
        config.data.input_types,
    )
    if not samples:
        parser.error(f"no samples in split {split!r} match the config filters")
    doc_types = load_doc_types(config.data.doc_types_dir, config.data.doc_types or None)
    report = evaluate_predictions(samples, read_predictions(predictions_path), doc_types)
    target = config.evaluation.quality_target

    write_eval_details(report, paths.eval_details(split))
    record_metrics(
        paths.metrics(split),
        experiment_id=config.experiment_id,
        run_name=config.run_name,
        stage="evaluate",
        config_path=args.config,
        updates={
            **report.metrics_updates(target),
            "split": split,
            "seed": config.seed,
            "max_seq_length": config.training.max_seq_length if config.training else None,
        },
    )
    print_report(f"{config.run_id} · {split} · {predictions_path}", report, report.meets(target))
    print(f"metrics: {paths.metrics(split)}  details: {paths.eval_details(split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
