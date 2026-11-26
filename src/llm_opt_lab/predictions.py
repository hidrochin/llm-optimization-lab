"""Predictions file: the contract between every inference path and evaluation.

HF inference (scripts/predict_hf.py), vLLM (inference/predict_vllm.py) and any other
backend write the same JSONL, so
`scripts/evaluate.py` scores all of them the same way.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import Field, ValidationError

from llm_opt_lab.config import StrictModel


class Prediction(StrictModel):
    id: str
    raw_output: str | None = None
    error: str | None = Field(default=None, description="Set when the request itself failed")
    # Fields the prompt asked for (generic paths, decoys included); evaluation scores these.
    # None: the config's inference request.
    fields: list[str] | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(prediction.model_dump_json() + "\n")


def read_predictions(path: str | Path) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                prediction = Prediction.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as e:
                raise ValueError(f"{path}:{line_no}: invalid prediction: {e}") from e
            if prediction.id in predictions:
                raise ValueError(f"{path}:{line_no}: duplicate prediction id {prediction.id!r}")
            predictions[prediction.id] = prediction
    return predictions
