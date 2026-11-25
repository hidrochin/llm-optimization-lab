"""Collect every metrics file into one CSV and the Markdown tables for the README.

Usage:
    uv run python scripts/aggregate_results.py
    uv run python scripts/aggregate_results.py --include-dev --out-dir /tmp/tables

Reads   results/metrics/*.json
Writes  results/tables/experiments.csv   one row per metrics file, every metric
        results/tables/experiments.md    every run and split, for the experiment matrix
        results/tables/final_table.md    test runs only, the columns of README §13

Dev runs (experiment ID DEV) are left out unless --include-dev. A run whose code was not
committed (`git_dirty` in any stage) is marked "dirty": its numbers cannot be reproduced.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_opt_lab.results import RunMetrics, load_run_metrics

# Nested sections flattened into `<prefix><key>.<metric>` columns.
NESTED = {"by_input_type": "", "by_doc_type": "", "by_concurrency": "c"}


def run_order(run: RunMetrics) -> tuple[Any, ...]:
    """E0, E1, ..., E10, then R0, R1; within an ID by run name, test before other splits."""
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", run.experiment_id)
    key = (match.group(1), int(match.group(2))) if match else (run.experiment_id, -1)
    return (*key, run.run_name, run.split != "test", run.split)


def commits(run: RunMetrics) -> str:
    found = sorted({s.git_commit for s in run.stages.values() if s.git_commit})
    dirty = any(s.git_dirty for s in run.stages.values())
    return " ".join(found) + (" dirty" if dirty else "")


def csv_row(run: RunMetrics) -> dict[str, Any]:
    data = run.model_dump(exclude={"stages", *NESTED})
    row: dict[str, Any] = {"run_id": f"{run.experiment_id}_{run.run_name}", **data}
    for section, prefix in NESTED.items():
        for key, stats in (getattr(run, section) or {}).items():
            for metric, value in stats.model_dump().items():
                row[f"{prefix}{key}.{metric}"] = value
    row["git_commits"] = commits(run)
    return row


def write_csv(path: Path, runs: list[RunMetrics]) -> None:
    rows = [csv_row(run) for run in runs]
    fixed = ["run_id", *RunMetrics.model_fields]
    fixed = [c for c in fixed if c not in ("stages", *NESTED)]
    extra = sorted({key for row in rows for key in row} - {*fixed, "git_commits"})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[*fixed, *extra, "git_commits"])
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float | None) -> str:
    return "" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, unit: str = "", scale: float = 1.0, digits: int = 1) -> str:
    return "" if value is None else f"{value * scale:.{digits}f}{unit}"


def latency_p95_at(run: RunMetrics, concurrency: int) -> float | None:
    stats = (run.by_concurrency or {}).get(str(concurrency))
    return None if stats is None else stats.latency_ms_p95


Column = tuple[str, Callable[[RunMetrics], str]]


def columns(concurrency: int) -> dict[str, Column]:
    return {
        "run": ("Run", lambda r: f"`{r.experiment_id}_{r.run_name}`"),
        "id": ("ID", lambda r: r.experiment_id),
        "split": ("Split", lambda r: r.split),
        "exact": ("Exact match", lambda r: _pct(r.exact_match)),
        "field": ("Field acc.", lambda r: _pct(r.field_accuracy)),
        "ned": ("NED", lambda r: _num(r.ned_mean, digits=4)),
        "parse": ("Parse", lambda r: _pct(r.parse_rate)),
        "pass": ("Pass?", lambda r: {True: "✅", False: "❌", None: ""}[r.meets_target]),
        "train_vram": ("Train VRAM", lambda r: _num(r.train_peak_vram_gb, " GB")),
        "train_time": ("Train time", lambda r: _num(r.train_time_s, " h", 1 / 3600)),
        "infer_vram": ("Inference VRAM", lambda r: _num(r.infer_peak_vram_gb, " GB")),
        "p95": (
            f"Latency p95 at {concurrency} requests",
            lambda r: _num(latency_p95_at(r, concurrency), " s", 1 / 1000),
        ),
        "commit": ("Commit", commits),
    }


def markdown_table(runs: list[RunMetrics], spec: dict[str, Column], keys: list[str]) -> str:
    header = [spec[k][0] for k in keys]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(spec[k][1](run) for k in keys) + " |" for run in runs]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--include-dev", action="store_true", help="also list DEV runs")
    parser.add_argument("--concurrency", type=int, default=60, help="level for the p95 column")
    args = parser.parse_args(argv)

    runs = sorted(load_run_metrics(args.metrics_dir, include_dev=args.include_dev), key=run_order)
    if not runs:
        print(f"no metrics files in {args.metrics_dir}")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = columns(args.concurrency)

    write_csv(args.out_dir / "experiments.csv", runs)
    every_run = ["run", "split", "exact", "field", "ned", "parse", "pass"]
    every_run += ["train_vram", "train_time", "infer_vram", "p95", "commit"]
    (args.out_dir / "experiments.md").write_text(
        markdown_table(runs, spec, every_run), encoding="utf-8"
    )
    test_runs = [run for run in runs if run.split == "test"]
    final = ["run", "id", "exact", "field", "ned", "pass", "infer_vram", "p95"]
    (args.out_dir / "final_table.md").write_text(
        markdown_table(test_runs, spec, final), encoding="utf-8"
    )

    dirty = [f"{r.experiment_id}_{r.run_name}" for r in runs if commits(r).endswith("dirty")]
    print(f"{len(runs)} metrics files ({len(test_runs)} on test) -> {args.out_dir}")
    if dirty:
        print(f"WARNING: uncommitted code in {dirty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
