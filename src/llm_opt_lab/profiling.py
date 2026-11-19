"""Timing and memory helpers shared by training, inference and benchmarking."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from types import TracebackType


class Timer:
    """`with Timer() as t: ...` then read `t.elapsed_s`."""

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        self.elapsed_s = 0.0
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_s = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile with linear interpolation (same as numpy's default), q in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def latency_summary(values_ms: Sequence[float]) -> dict[str, float]:
    return {
        "p50": percentile(values_ms, 50),
        "p95": percentile(values_ms, 95),
        "mean": sum(values_ms) / len(values_ms),
    }


def reset_peak_vram(device: int = 0) -> None:
    import torch

    torch.cuda.reset_peak_memory_stats(device)


def peak_vram_gb(device: int = 0) -> float:
    """Peak memory allocated by PyTorch on `device` since the last reset.

    This misses memory outside the PyTorch allocator (vLLM, CUDA context); use
    `nvidia-smi` for whole-process numbers.
    """
    import torch

    return torch.cuda.max_memory_allocated(device) / 1024**3
