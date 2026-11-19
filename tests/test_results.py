import json

import pytest

from llm_opt_lab.profiling import latency_summary, percentile
from llm_opt_lab.results import record_metrics

RUN = {"experiment_id": "E1", "run_name": "r16"}


def test_record_metrics_merges_stages(tmp_path):
    path = tmp_path / "E1_r16.json"
    record_metrics(path, **RUN, stage="train", updates={"train_time_s": 120.0})
    record_metrics(
        path, **RUN, stage="evaluate", updates={"exact_match": 0.9, "train_time_s": None}
    )
    data = json.loads(path.read_text())
    assert data["train_time_s"] == 120.0
    assert data["exact_match"] == 0.9
    assert set(data["stages"]) == {"train", "evaluate"}
    assert data["stages"]["train"]["hostname"]


def test_record_metrics_rejects_other_run(tmp_path):
    path = tmp_path / "E1_r16.json"
    record_metrics(path, **RUN, stage="train", updates={})
    with pytest.raises(ValueError, match="belongs to"):
        record_metrics(path, experiment_id="E2", run_name="r16", stage="train", updates={})


def test_record_metrics_rejects_unknown_field(tmp_path):
    with pytest.raises(ValueError, match="latency_p99"):
        record_metrics(tmp_path / "m.json", **RUN, stage="bench", updates={"latency_p99": 1.0})


def test_percentile_matches_numpy_linear():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 95) == pytest.approx(48.0)
    assert latency_summary([1, 2, 3])["mean"] == 2
