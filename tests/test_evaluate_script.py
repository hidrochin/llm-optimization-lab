"""End to end: config + split + predictions -> metrics file and eval details."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from conftest import DATA_DIR, DOC_TYPES_DIR
from llm_opt_lab.predictions import Prediction, write_predictions

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate.py"


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_end_to_end(tmp_path, monkeypatch, samples):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLMOPT_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    config = {
        "experiment_id": "E0",
        "run_name": "fixture",
        "model": {"id": "google/gemma-4-E2B-it"},
        "data": {"dir": str(DATA_DIR), "doc_types_dir": str(DOC_TYPES_DIR)},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))

    perfect = dict(
        samples[0].target, store={"name": "  CỬA HÀNG  AN", "address": "12 Lê Lợi, Quận 1"}
    )
    one_wrong = dict(samples[1].target, total=60000)
    write_predictions(
        tmp_path / "artifacts/predictions/E0_fixture/test.jsonl",
        [
            Prediction(id=samples[0].id, raw_output=json.dumps(perfect, ensure_ascii=False)),
            Prediction(id=samples[1].id, raw_output=f"```json\n{json.dumps(one_wrong)}\n```"),
            Prediction(id=samples[2].id, raw_output='{"store": '),
        ],
    )

    assert load_script().main(["--config", "config.yaml"]) == 0

    metrics = json.loads((tmp_path / "results/metrics/E0_fixture.json").read_text())
    assert metrics["n_samples"] == 3 and metrics["n_fields"] == 27
    assert metrics["exact_match"] == pytest.approx(1 / 3)
    assert metrics["field_accuracy"] == pytest.approx(19 / 27)
    assert metrics["parse_rate"] == pytest.approx(2 / 3)
    assert metrics["meets_target"] is False
    assert metrics["by_input_type"]["image"]["exact_match"] == 1.0
    assert "evaluate" in metrics["stages"]

    details_dir = tmp_path / "artifacts/eval/E0_fixture/test"
    details = [json.loads(line) for line in (details_dir / "samples.jsonl").open()]
    assert details[2]["parse_error"].startswith("invalid JSON")
    worst_field = json.loads((details_dir / "fields.json").read_text())[0]
    assert (worst_field["field"], worst_field["n"]) == ("total", 3)


def test_missing_prediction_is_an_error(tmp_path, monkeypatch, samples):
    monkeypatch.chdir(tmp_path)
    config = {
        "experiment_id": "E0",
        "run_name": "fixture",
        "model": {"id": "m"},
        "data": {"dir": str(DATA_DIR), "doc_types_dir": str(DOC_TYPES_DIR)},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    write_predictions(tmp_path / "p.jsonl", [Prediction(id=samples[0].id, raw_output="{}")])
    with pytest.raises(ValueError, match="2 samples have no prediction"):
        load_script().main(["--config", "config.yaml", "--predictions", "p.jsonl"])
