import asyncio

import pytest

from llm_opt_lab.config import ExperimentConfig
from llm_opt_lab.serving import (
    RequestResult,
    build_vllm_command,
    parse_prometheus,
    run_closed_loop,
    summarize_requests,
    summarize_server_metrics,
)

CONFIG = {
    "experiment_id": "E4",
    "run_name": "w4a16",
    "model": {"id": "google/gemma-4-E2B-it"},
    "data": {"dir": "/data"},
    "serving": {"source": "google/gemma-4-E2B-it-qat-w4a16-ct", "max_model_len": 34816},
}


def test_build_vllm_command():
    cmd = build_vllm_command(ExperimentConfig.model_validate(CONFIG))
    assert cmd[:3] == ["vllm", "serve", "google/gemma-4-E2B-it-qat-w4a16-ct"]
    args = dict(zip(cmd[3::2], cmd[4::2], strict=False))
    assert args["--served-model-name"] == "E4_w4a16"
    assert args["--max-model-len"] == "34816"
    assert "--enable-prefix-caching" in cmd
    assert cmd[cmd.index("--limit-mm-per-prompt") + 1] == '{"image": 1}'


def test_text_only_model_has_no_image_limit():
    config = dict(CONFIG, model={"id": "Qwen/Qwen3-4B", "architecture": "causal_lm"})
    assert "--limit-mm-per-prompt" not in build_vllm_command(
        ExperimentConfig.model_validate(config)
    )


def test_closed_loop_respects_concurrency():
    in_flight, peak = 0, 0

    async def send(item):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.001)
        in_flight -= 1
        return item * 2

    results = asyncio.run(run_closed_loop(list(range(20)), 4, send))
    assert results == [i * 2 for i in range(20)]
    assert peak == 4


def test_summarize_requests():
    results = [
        RequestResult("a", "x", None, latency_ms=1000, ttft_ms=100, completion_tokens=10),
        RequestResult("b", "x", None, latency_ms=2000, ttft_ms=200, completion_tokens=19),
        RequestResult("c", None, "Timeout", latency_ms=5000),
    ]
    stats = summarize_requests(results, duration_s=2.0)
    assert (stats.n_requests, stats.n_errors) == (3, 1)
    assert stats.latency_ms_p50 == 1500
    assert stats.tpot_ms_p50 == pytest.approx((900 / 9 + 1800 / 18) / 2)
    assert stats.tokens_per_s == pytest.approx(14.5)
    assert stats.requests_per_s == pytest.approx(1.0)


def test_parse_prometheus_and_server_summary():
    text = """# HELP vllm:num_requests_running Number of requests running.
vllm:num_requests_running{model_name="m"} 3.0
vllm:kv_cache_usage_perc{model_name="m"} 0.42
vllm:prefix_cache_queries_total{model_name="m"} 1000.0
vllm:prefix_cache_hits_total{model_name="m"} 100.0
"""
    first = parse_prometheus(text)
    assert first["vllm:num_requests_running"] == 3.0
    last = dict(
        first, **{"vllm:prefix_cache_queries_total": 3000.0, "vllm:prefix_cache_hits_total": 1100.0}
    )
    summary = summarize_server_metrics([first, last])
    assert summary["kv_cache_usage_max"] == 0.42
    assert summary["prefix_cache_hit_rate"] == pytest.approx(0.5)
