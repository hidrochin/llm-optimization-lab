"""Plot accuracy against cost for every test run, with the 95% target line.

Usage:
    uv run --group report python scripts/plot_tradeoffs.py

Reads   results/metrics/*.json (test runs; DEV runs only with --include-dev)
Writes  results/figures/accuracy_vs_<cost>.png, one figure per cost with data:
        training VRAM, training time, inference VRAM, latency p95 at --concurrency.

Each figure has two panels, exact match and field accuracy, on the same y scale. Every
point is one run, labelled with its experiment ID (the run ID when an ID has several runs).
The numbers behind the figures are in results/tables/experiments.csv.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from llm_opt_lab.results import RunMetrics, load_run_metrics  # noqa: E402

# Reference palette: one series, recessive grid and target line, text in text tokens.
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"

Cost = tuple[str, str, Callable[[RunMetrics], float | None]]


def costs(concurrency: int) -> list[Cost]:
    def latency(run: RunMetrics) -> float | None:
        stats = (run.by_concurrency or {}).get(str(concurrency))
        return (
            None if stats is None or stats.latency_ms_p95 is None else stats.latency_ms_p95 / 1000
        )

    return [
        ("train_vram", "Training peak VRAM (GB)", lambda r: r.train_peak_vram_gb),
        (
            "train_time",
            "Training time (h)",
            lambda r: None if r.train_time_s is None else r.train_time_s / 3600,
        ),
        ("infer_vram", "Inference peak VRAM (GB)", lambda r: r.infer_peak_vram_gb),
        (f"latency_p95_c{concurrency}", f"Latency p95 at {concurrency} requests (s)", latency),
    ]


def labels(runs: list[RunMetrics]) -> dict[int, str]:
    per_id = Counter(run.experiment_id for run in runs)
    return {
        id(run): run.experiment_id
        if per_id[run.experiment_id] == 1
        else f"{run.experiment_id}_{run.run_name}"
        for run in runs
    }


def plot(
    runs: list[RunMetrics], x_label: str, x: Callable[[RunMetrics], float | None], target: float
) -> Figure | None:
    points = [(run, x(run)) for run in runs if x(run) is not None]
    names = labels([run for run, _ in points])
    panels = [
        ("Exact match", lambda r: r.exact_match),
        ("Field accuracy", lambda r: r.field_accuracy),
    ]
    data = [
        [(xv, y(run), names[id(run)]) for run, xv in points if y(run) is not None]
        for _, y in panels
    ]
    if not any(data):
        return None

    ys = [y for panel in data for _, y, _ in panel]
    y_low = min(min(ys), target) - 0.02
    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    for ax, (title, _), panel in zip(axes, panels, data, strict=True):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        ax.axhline(target, color=TEXT_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            f"{target:.0%} target",
            (1, target),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=TEXT_MUTED,
        )
        if panel:
            xs, ys_panel, texts = zip(*panel, strict=True)
            ax.scatter(xs, ys_panel, s=64, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=3)
            for xv, yv, text in panel:
                ax.annotate(
                    text,
                    (xv, yv),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=9,
                    color=TEXT,
                )
        ax.set_xlabel(x_label)
        ax.set_ylim(y_low, 1.005)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--include-dev", action="store_true", help="also plot DEV runs")
    parser.add_argument("--concurrency", type=int, default=60, help="level for the latency plot")
    parser.add_argument("--target", type=float, default=0.95, help="quality target line")
    args = parser.parse_args(argv)

    runs = load_run_metrics(args.metrics_dir, include_dev=args.include_dev)
    runs = [run for run in runs if run.split == "test"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, x_label, x in costs(args.concurrency):
        fig = plot(runs, x_label, x, args.target)
        if fig is None:
            print(f"skipped {name}: no test run has both accuracy and this cost")
            continue
        path = args.out_dir / f"accuracy_vs_{name}.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)
    print(f"{len(runs)} test runs, {len(written)} figures -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
