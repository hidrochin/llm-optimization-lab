"""Standard locations for everything a run produces.

`results/` is committed (metrics, logs, tables). `artifacts/` is not: it holds
checkpoints and anything containing private data (predictions, per-sample eval).
Set `LLMOPT_ARTIFACTS_DIR` to put artifacts on another disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from llm_opt_lab.config import ExperimentConfig

RESULTS_DIR = Path("results")


def artifacts_dir() -> Path:
    return Path(os.environ.get("LLMOPT_ARTIFACTS_DIR", "artifacts"))


def directory_size(path: str | Path) -> int:
    """Total size in bytes of the files under `path`."""
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


@dataclass(frozen=True)
class RunPaths:
    run_id: str

    @classmethod
    def for_config(cls, config: ExperimentConfig) -> RunPaths:
        return cls(config.run_id)

    def metrics(self, split: str = "test") -> Path:
        """Test metrics are the headline result; other splits get a suffix."""
        suffix = "" if split == "test" else f"__{split}"
        return RESULTS_DIR / "metrics" / f"{self.run_id}{suffix}.json"

    @property
    def logs(self) -> Path:
        return RESULTS_DIR / "logs" / self.run_id

    def predictions(self, split: str = "test") -> Path:
        return artifacts_dir() / "predictions" / self.run_id / f"{split}.jsonl"

    def eval_details(self, split: str = "test") -> Path:
        return artifacts_dir() / "eval" / self.run_id / split

    @property
    def checkpoints(self) -> Path:
        """Intermediate trainer checkpoints; the final adapter goes to `adapter`."""
        return artifacts_dir() / "checkpoints" / self.run_id

    @property
    def adapter(self) -> Path:
        return artifacts_dir() / "adapters" / self.run_id

    @property
    def merged(self) -> Path:
        return artifacts_dir() / "merged" / self.run_id

    @property
    def quantized(self) -> Path:
        return artifacts_dir() / "quantized" / self.run_id


@dataclass(frozen=True)
class ModelSource:
    model: str  # hub id or local directory
    adapter: Path | None = None  # LoRA adapter to load on top of `model`


_RUN_ARTIFACTS = ("merged", "quantized", "adapter")


def resolve_model_source(config: ExperimentConfig, source: str) -> ModelSource:
    """Turn a `source` string from a config or CLI into what to load.

    - `base`: `model.id`
    - `merged`, `quantized`, `adapter`: this run's artifact (`adapter` = base + adapter)
    - `<kind>:<run_id>`: another run's artifact, e.g. `merged:E1_gemma4_e2b_lora_r16`
      for E4, which quantizes the E3 merged model
    - anything else: a local path or hub id, used as is
    """
    kind, _, run_id = source.partition(":")
    if source == "base":
        return ModelSource(config.model.id)
    if kind in _RUN_ARTIFACTS:
        paths = RunPaths(run_id) if run_id else RunPaths.for_config(config)
        if kind == "adapter":
            return ModelSource(config.model.id, paths.adapter)
        return ModelSource(str(getattr(paths, kind)))
    return ModelSource(os.path.expandvars(source))
