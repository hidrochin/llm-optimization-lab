"""Latency and throughput vs concurrency against a running vLLM server.

Replays <data root>/benchmark.jsonl at each level of `serving.concurrency_levels`
(closed loop: N requests in flight), polls /metrics during each level, and records
`by_concurrency` (and the concurrency-1 numbers at top level) in the run's metrics file.
Per-request timings (no model output) go to results/logs/<run_id>/benchmark_c<N>.jsonl.

The prefix cache is reset before each level. Without that, later levels would reuse
whole cached prompts from earlier levels and look faster than they are. Resetting needs
VLLM_SERVER_DEV_MODE=1 on the server (serve_vllm.py sets it).

Usage:
    python inference/benchmark_api.py --config configs/dev/cord_e2b_base_vllm.yaml
    python inference/benchmark_api.py --config ... --levels 1 60 --max-requests 120
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from client import make_client, request_messages, send_chat  # noqa: E402

from llm_opt_lab.config import load_config  # noqa: E402
from llm_opt_lab.data import Sample, filter_samples, load_samples  # noqa: E402
from llm_opt_lab.doc_types import load_doc_types  # noqa: E402
from llm_opt_lab.paths import RunPaths  # noqa: E402
from llm_opt_lab.results import ServingStats, record_metrics, save_gpu_snapshot  # noqa: E402
from llm_opt_lab.serving import (  # noqa: E402
    parse_prometheus,
    run_closed_loop,
    summarize_requests,
    summarize_server_metrics,
)

TOP_LEVEL = (
    "ttft_ms_p50",
    "ttft_ms_p95",
    "tpot_ms_p50",
    "latency_ms_p50",
    "latency_ms_p95",
    "tokens_per_s",
    "requests_per_s",
)
# (header, ServingStats field, format) for the console table
COLUMNS = (
    ("n", "n_requests", "d"),
    ("err", "n_errors", "d"),
    ("ttft p50", "ttft_ms_p50", ".0f"),
    ("ttft p95", "ttft_ms_p95", ".0f"),
    ("lat p50", "latency_ms_p50", ".0f"),
    ("lat p95", "latency_ms_p95", ".0f"),
    ("tok/s", "tokens_per_s", ".0f"),
    ("req/s", "requests_per_s", ".2f"),
    ("kv max", "kv_cache_usage_max", ".2f"),
)


def table_row(first: object, cells: list[str]) -> str:
    return f"{first!s:>5}" + "".join(f"{cell:>10}" for cell in cells)


def stats_row(level: int, stats: ServingStats) -> str:
    values = [getattr(stats, field) for _, field, _ in COLUMNS]
    cells = [
        "-" if v is None else format(v, spec)
        for v, (_, _, spec) in zip(values, COLUMNS, strict=True)
    ]
    return table_row(level, cells)


async def poll_metrics(
    http: httpx.AsyncClient, url: str, stop: asyncio.Event, out: list[dict[str, float]]
) -> None:
    while True:
        with contextlib.suppress(httpx.HTTPError):
            out.append(parse_prometheus((await http.get(url)).text))
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
            return
        except TimeoutError:
            continue


async def reset_prefix_cache(http: httpx.AsyncClient, root: str) -> bool:
    try:
        return (await http.post(f"{root}/reset_prefix_cache")).status_code == 200
    except httpx.HTTPError:
        return False


async def benchmark(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    serving = config.serving
    if serving is None:
        raise SystemExit("config has no serving section")
    paths = RunPaths.for_config(config)
    samples = filter_samples(
        load_samples(config.data.dir / "benchmark.jsonl"),
        config.data.doc_types,
        config.data.input_types,
    )[: args.max_requests]
    doc_types = load_doc_types(config.data.doc_types_dir)
    client = make_client(config)
    root = f"http://{serving.host}:{serving.port}"

    async def send(sample: Sample):
        messages = request_messages(sample, doc_types[sample.doc_type], config.data.dir)
        return await send_chat(client, config, sample.id, messages)

    save_gpu_snapshot(paths.logs / "nvidia-smi_benchmark.txt")
    by_level: dict[str, ServingStats] = {}
    notes: list[str] = []
    async with httpx.AsyncClient(timeout=10) as http:
        warmup = samples[: serving.warmup_requests]
        if warmup:
            await run_closed_loop(warmup, len(warmup), send)
        print(table_row("conc", [header for header, _, _ in COLUMNS]))
        for level in args.levels or serving.concurrency_levels:
            if not await reset_prefix_cache(http, root):
                notes.append(f"c={level}: prefix cache not reset")
                print(
                    f"WARNING c={level}: could not reset the prefix cache; results are optimistic"
                )
            snapshots: list[dict[str, float]] = []
            stop = asyncio.Event()
            poller = asyncio.create_task(poll_metrics(http, f"{root}/metrics", stop, snapshots))
            start = time.perf_counter()
            results = await run_closed_loop(samples, level, send)
            duration = time.perf_counter() - start
            stop.set()
            await poller

            stats = summarize_requests(results, duration).model_copy(
                update=summarize_server_metrics(snapshots)
            )
            by_level[str(level)] = stats
            paths.logs.mkdir(parents=True, exist_ok=True)
            # Timings only: the model output may contain private data and is not logged.
            with open(paths.logs / f"benchmark_c{level}.jsonl", "w", encoding="utf-8") as log:
                for r in results:
                    timing = {k: v for k, v in dataclasses.asdict(r).items() if k != "text"}
                    log.write(json.dumps(timing) + "\n")
            print(stats_row(level, stats))

    updates: dict[str, object] = {"by_concurrency": by_level}
    if "1" in by_level:
        updates.update({k: getattr(by_level["1"], k) for k in TOP_LEVEL})
    if notes:
        updates["stability_notes"] = "; ".join(notes)
    record_metrics(
        paths.metrics(),
        experiment_id=config.experiment_id,
        run_name=config.run_name,
        stage="benchmark",
        config_path=args.config,
        updates=updates,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--levels", type=int, nargs="+", help="defaults to serving.concurrency_levels"
    )
    parser.add_argument("--max-requests", type=int, help="first N benchmark samples per level")
    return asyncio.run(benchmark(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
