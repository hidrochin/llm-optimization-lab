"""Serving helpers without vLLM or HTTP dependencies, so they are testable on CPU.

`inference/` holds the parts that talk to a running vLLM server.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from llm_opt_lab.config import ExperimentConfig
from llm_opt_lab.paths import resolve_model_source
from llm_opt_lab.profiling import percentile
from llm_opt_lab.results import ServingStats

T = TypeVar("T")
R = TypeVar("R")


def build_vllm_command(config: ExperimentConfig) -> list[str]:
    """`vllm serve ...` for the config's `serving` section; the served model name is the run id."""
    serving = config.serving
    if serving is None:
        raise ValueError("config has no serving section")
    source = resolve_model_source(config, serving.source)
    if source.adapter is not None:
        raise ValueError("serving an unmerged adapter is not supported here; merge it first")
    cmd = [
        "vllm", "serve", source.model,
        "--served-model-name", config.run_id,
        "--host", serving.host,
        "--port", str(serving.port),
        "--dtype", config.model.dtype,
        "--max-model-len", str(serving.max_model_len),
        "--max-num-seqs", str(serving.max_num_seqs),
        "--gpu-memory-utilization", str(serving.gpu_memory_utilization),
        "--kv-cache-dtype", serving.kv_cache_dtype,
        "--seed", str(config.seed),
    ]  # fmt: skip
    if source.model == config.model.id and config.model.revision:
        cmd += ["--revision", config.model.revision]
    cmd.append(
        "--enable-prefix-caching" if serving.enable_prefix_caching else "--no-enable-prefix-caching"
    )
    if config.model.architecture == "image_text_to_text":
        cmd += ["--limit-mm-per-prompt", json.dumps({"image": serving.max_images_per_prompt})]
    return cmd + serving.extra_args


@dataclass(frozen=True)
class RequestResult:
    id: str
    text: str | None
    error: str | None
    latency_ms: float
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def tpot_ms(self) -> float | None:
        """Time per output token after the first one."""
        if self.ttft_ms is None or not self.completion_tokens or self.completion_tokens < 2:
            return None
        return (self.latency_ms - self.ttft_ms) / (self.completion_tokens - 1)


async def run_closed_loop(
    items: Sequence[T], concurrency: int, send: Callable[[T], Awaitable[R]]
) -> list[R]:
    """Send every item with at most `concurrency` requests in flight; results keep item order."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: T) -> R:
        async with semaphore:
            return await send(item)

    return list(await asyncio.gather(*(one(item) for item in items)))


def _p(values: list[float], q: float) -> float | None:
    return percentile(values, q) if values else None


def summarize_requests(results: Sequence[RequestResult], duration_s: float) -> ServingStats:
    ok = [r for r in results if r.error is None]
    latencies = [r.latency_ms for r in ok]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    tpots = [t for r in ok if (t := r.tpot_ms) is not None]
    tokens = sum(r.completion_tokens or 0 for r in ok)
    return ServingStats(
        n_requests=len(results),
        n_errors=len(results) - len(ok),
        duration_s=duration_s,
        ttft_ms_p50=_p(ttfts, 50),
        ttft_ms_p95=_p(ttfts, 95),
        tpot_ms_p50=_p(tpots, 50),
        latency_ms_p50=_p(latencies, 50),
        latency_ms_p95=_p(latencies, 95),
        tokens_per_s=tokens / duration_s if duration_s > 0 else None,
        requests_per_s=len(ok) / duration_s if duration_s > 0 else None,
    )


_SAMPLE_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")

# Metric names differ between vLLM versions; the first one present is used.
KV_CACHE_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
NUM_RUNNING = ("vllm:num_requests_running",)
NUM_WAITING = ("vllm:num_requests_waiting",)
PREFIX_HITS = ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total")
PREFIX_QUERIES = ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total")


def parse_prometheus(text: str) -> dict[str, float]:
    """Prometheus text format -> {metric name: value summed over label sets}."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _SAMPLE_LINE.match(line)
        if not match:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        values[match.group(1)] = values.get(match.group(1), 0.0) + value
    return values


def _first(snapshot: dict[str, float], names: Sequence[str]) -> float | None:
    return next((snapshot[n] for n in names if n in snapshot), None)


def summarize_server_metrics(snapshots: Sequence[dict[str, float]]) -> dict[str, float | None]:
    """Peaks of gauges and the prefix cache hit rate over a run, from periodic snapshots."""

    def peak(names: Sequence[str]) -> float | None:
        found = [v for s in snapshots if (v := _first(s, names)) is not None]
        return max(found) if found else None

    hit_rate = None
    if len(snapshots) >= 2:
        hits = [_first(s, PREFIX_HITS) for s in (snapshots[0], snapshots[-1])]
        queries = [_first(s, PREFIX_QUERIES) for s in (snapshots[0], snapshots[-1])]
        if None not in hits and None not in queries and queries[1] > queries[0]:
            hit_rate = (hits[1] - hits[0]) / (queries[1] - queries[0])
    return {
        "kv_cache_usage_max": peak(KV_CACHE_USAGE),
        "num_running_max": peak(NUM_RUNNING),
        "num_waiting_max": peak(NUM_WAITING),
        "prefix_cache_hit_rate": hit_rate,
    }
