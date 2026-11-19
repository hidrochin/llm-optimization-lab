"""Metrics JSON: one file per run, shared schema, filled in by several stages.

Training, evaluation and benchmarking each call `record_metrics` with the fields they
measured; values are merged into the same file and each stage leaves its provenance
(git commit, host, GPU) under `stages`.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from llm_opt_lab.metrics import QualityStats


class StageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    git_commit: str | None
    git_dirty: bool | None
    hostname: str
    cuda_visible_devices: str | None
    gpu_name: str | None
    config: str | None = None


class ServingStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_requests: int | None = None
    n_errors: int | None = None
    duration_s: float | None = None
    ttft_ms_p50: float | None = None
    ttft_ms_p95: float | None = None
    tpot_ms_p50: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    tokens_per_s: float | None = None
    requests_per_s: float | None = None
    # From the vLLM /metrics endpoint during the run
    kv_cache_usage_max: float | None = None
    num_running_max: float | None = None
    num_waiting_max: float | None = None
    prefix_cache_hit_rate: float | None = None


class RunMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    run_name: str
    split: str = "test"
    config: str | None = None
    seed: int | None = None
    max_seq_length: int | None = None
    stages: dict[str, StageInfo] = {}

    # Quality (evaluate)
    n_samples: int | None = None
    n_fields: int | None = None
    exact_match: float | None = None
    field_accuracy: float | None = None
    ned_mean: float | None = None
    parse_rate: float | None = None
    meets_target: bool | None = None
    by_input_type: dict[str, QualityStats] | None = None
    by_doc_type: dict[str, QualityStats] | None = None

    # Training
    train_peak_vram_gb: float | None = None
    train_time_s: float | None = None
    train_steps_per_s: float | None = None
    trainable_params: int | None = None
    n_train_samples: int | None = None
    n_dropped_over_length: int | None = None

    # Merge check (base + adapter vs merged)
    merge_max_abs_logit_diff: float | None = None
    merge_identical_output_rate: float | None = None

    # HF inference (predict_hf, one request at a time); not comparable with vLLM serving
    hf_latency_ms_p50: float | None = None
    hf_latency_ms_p95: float | None = None
    hf_tokens_per_s: float | None = None

    # vLLM serving; top-level values are at concurrency 1, the sweep goes in by_concurrency
    infer_peak_vram_gb: float | None = None
    kv_cache_gb: float | None = None
    ttft_ms_p50: float | None = None
    ttft_ms_p95: float | None = None
    tpot_ms_p50: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    tokens_per_s: float | None = None
    requests_per_s: float | None = None
    by_concurrency: dict[str, ServingStats] | None = None

    # Size and stability
    checkpoint_size_gb: float | None = None
    adapter_size_mb: float | None = None
    stability_notes: str = ""


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_name() -> str | None:
    """Name of the first visible GPU, without importing torch."""
    output = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not output:
        return None
    names = output.splitlines()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",")[0].strip()
        if first.isdigit() and int(first) < len(names):
            return names[int(first)].strip()
    return names[0].strip()


def stage_info(config_path: str | Path | None = None) -> StageInfo:
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return StageInfo(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        git_commit=_run(["git", "rev-parse", "--short", "HEAD"]),
        git_dirty=None if status is None else bool(status),
        hostname=socket.gethostname(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_name=gpu_name(),
        config=str(config_path) if config_path else None,
    )


def save_gpu_snapshot(path: str | Path) -> Path:
    """Save full `nvidia-smi` output next to a measurement, to prove the GPU was idle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_run(["nvidia-smi"]) or "nvidia-smi not available\n", encoding="utf-8")
    return path


def record_metrics(
    path: str | Path,
    *,
    experiment_id: str,
    run_name: str,
    stage: str,
    updates: dict[str, Any],
    config_path: str | Path | None = None,
) -> RunMetrics:
    """Merge `updates` into the run's metrics file and stamp `stages[stage]`.

    None values in `updates` do not overwrite existing values.
    """
    path = Path(path)
    if path.exists():
        current = RunMetrics.model_validate_json(path.read_text(encoding="utf-8"))
        if (current.experiment_id, current.run_name) != (experiment_id, run_name):
            raise ValueError(f"{path} belongs to {current.experiment_id}_{current.run_name}")
        data = current.model_dump()
    else:
        data = {"experiment_id": experiment_id, "run_name": run_name}
    data.update({k: v for k, v in updates.items() if v is not None})
    if config_path is not None:
        data["config"] = str(config_path)
    data.setdefault("stages", {})[stage] = stage_info(config_path).model_dump()
    metrics = RunMetrics.model_validate(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics.model_dump(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics
