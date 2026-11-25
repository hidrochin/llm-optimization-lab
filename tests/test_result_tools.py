"""Result tables, trade-off figures and error analysis (T-L07)."""

import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from llm_opt_lab.evaluation import evaluate_predictions, write_eval_details
from llm_opt_lab.paths import RunPaths
from llm_opt_lab.predictions import Prediction
from llm_opt_lab.results import load_run_metrics

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage(commit="abc1234", dirty=False):
    return {
        "timestamp": "2026-11-25T10:00:00",
        "git_commit": commit,
        "git_dirty": dirty,
        "hostname": "gpu-01",
        "cuda_visible_devices": "0",
        "gpu_name": "NVIDIA L40",
    }


def quality(em, fa):
    return {
        "n": 10,
        "n_fields": 90,
        "exact_match": em,
        "field_accuracy": fa,
        "ned_mean": 0.01,
        "parse_rate": 1.0,
    }


def write_metrics(metrics_dir, name, **fields):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{name}.json").write_text(json.dumps(fields), encoding="utf-8")


@pytest.fixture
def metrics_dir(tmp_path):
    d = tmp_path / "metrics"
    serving = {"60": {"n_requests": 300, "latency_ms_p95": 9100.0}}
    write_metrics(
        d,
        "E1_lora_r16",
        experiment_id="E1",
        run_name="lora_r16",
        stages={"train": stage(), "evaluate": stage()},
        exact_match=0.96,
        field_accuracy=0.991,
        ned_mean=0.004,
        parse_rate=1.0,
        meets_target=True,
        by_input_type={"image": quality(0.95, 0.99), "spatial_text": quality(0.97, 0.992)},
        train_peak_vram_gb=41.2,
        train_time_s=15120,
        infer_peak_vram_gb=14.8,
        by_concurrency=serving,
    )
    write_metrics(
        d,
        "E1_lora_r16__validation",
        experiment_id="E1",
        run_name="lora_r16",
        split="validation",
        exact_match=0.97,
        field_accuracy=0.993,
        meets_target=True,
    )
    write_metrics(
        d,
        "E10_moe_experts",
        experiment_id="E10",
        run_name="moe_experts",
        exact_match=0.9,
        field_accuracy=0.97,
        meets_target=False,
        train_peak_vram_gb=46.0,
    )
    write_metrics(
        d,
        "E2_qlora_r16",
        experiment_id="E2",
        run_name="qlora_r16",
        stages={"train": stage("def5678", dirty=True)},
        exact_match=0.94,
        field_accuracy=0.985,
        meets_target=False,
        train_peak_vram_gb=22.5,
    )
    write_metrics(d, "DEV_cord_smoke", experiment_id="DEV", run_name="cord_smoke", exact_match=0.5)
    return d


def test_load_run_metrics_leaves_out_dev(metrics_dir):
    assert {r.experiment_id for r in load_run_metrics(metrics_dir)} == {"E1", "E2", "E10"}
    assert len(load_run_metrics(metrics_dir, include_dev=True)) == 5


def test_aggregate_results(tmp_path, metrics_dir, capsys):
    out = tmp_path / "tables"
    script = load_script("aggregate_results")
    assert script.main(["--metrics-dir", str(metrics_dir), "--out-dir", str(out)]) == 0
    assert "uncommitted code in ['E2_qlora_r16']" in capsys.readouterr().out

    with open(out / "experiments.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # E1 test before its validation run; E2 before E10 (numeric order); DEV left out
    assert [(r["run_id"], r["split"]) for r in rows] == [
        ("E1_lora_r16", "test"),
        ("E1_lora_r16", "validation"),
        ("E2_qlora_r16", "test"),
        ("E10_moe_experts", "test"),
    ]
    assert rows[0]["image.exact_match"] == "0.95"
    assert rows[0]["c60.latency_ms_p95"] == "9100.0"
    assert rows[2]["git_commits"] == "def5678 dirty"

    final = (out / "final_table.md").read_text(encoding="utf-8").splitlines()
    assert final[0].startswith("| Run | ID | Exact match | Field acc. | NED | Pass?")
    assert len(final) == 2 + 3  # header, separator, three test runs
    assert final[2] == "| `E1_lora_r16` | E1 | 96.0% | 99.1% | 0.0040 | ✅ | 14.8 GB | 9.1 s |"
    experiments = (out / "experiments.md").read_text(encoding="utf-8")
    assert "| `E1_lora_r16` | validation | 97.0% |" in experiments
    assert "| 4.2 h |" in experiments


def test_aggregate_results_without_metrics(tmp_path):
    script = load_script("aggregate_results")
    assert script.main(["--metrics-dir", str(tmp_path / "none"), "--out-dir", str(tmp_path)]) == 0


def test_plot_tradeoffs(tmp_path, metrics_dir, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "figures"
    script = load_script("plot_tradeoffs")
    assert script.main(["--metrics-dir", str(metrics_dir), "--out-dir", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "accuracy_vs_infer_vram.png",
        "accuracy_vs_latency_p95_c60.png",
        "accuracy_vs_train_time.png",
        "accuracy_vs_train_vram.png",
    ]
    capsys.readouterr()
    write_metrics(metrics_dir, "E3_merge", experiment_id="E3", run_name="merge", exact_match=0.9)
    assert (
        script.main(
            ["--metrics-dir", str(metrics_dir), "--out-dir", str(out), "--concurrency", "8"]
        )
        == 0
    )
    assert "skipped latency_p95_c8" in capsys.readouterr().out


def run_eval(run_id, samples, doc_types, outputs):
    """Score `outputs` (target dicts or raw strings) as `run_id` on test, like evaluate.py."""
    predictions = {
        s.id: Prediction(id=s.id, raw_output=o if isinstance(o, str) else json.dumps(o))
        for s, o in zip(samples, outputs, strict=True)
    }
    report = evaluate_predictions(samples, predictions, doc_types)
    write_eval_details(report, RunPaths(run_id).eval_details("test"))


@pytest.fixture
def two_runs(tmp_path, monkeypatch, samples, doc_types):
    monkeypatch.setenv("LLMOPT_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    targets = [copy.deepcopy(s.target) for s in samples]

    # A: items reordered on sample 0, sample 1 right, total off by one digit on sample 2
    a = copy.deepcopy(targets)
    a[0]["items"].reverse()
    a[2]["total"] = 15001
    run_eval("E2_a", samples, doc_types, a)

    # B: sample 0 right, total missing and an extra field on sample 1, sample 2 unparsed
    b = copy.deepcopy(targets)
    del b[1]["total"]
    b[1]["note"] = "x"
    b[2] = '{"store": '
    run_eval("E1_b", samples, doc_types, b)


def test_error_analysis_report(two_runs):
    script = load_script("error_analysis")
    assert script.main(["--run", "E2_a"]) == 0
    text = (RunPaths("E2_a").eval_details("test") / "errors.md").read_text(encoding="utf-8")
    # total: one near miss (65000 -> 65001 has NED 0.2) out of three
    assert "| receipt | `total` | 3 | 66.7% | 0 | 0 | 0 | 1 | 0 |" in text
    assert "| 15000 | 15001 | 1 |" in text
    # sample 0 has the right items in another order: 1 of 3 item lists
    assert "| `items` | 3 | 0.0% | 0.0% | 33.3% |" in text

    assert script.main(["--run", "E1_b"]) == 0
    text = (RunPaths("E1_b").eval_details("test") / "errors.md").read_text(encoding="utf-8")
    assert "1 of 3 samples." in text and "- invalid JSON: 1" in text
    assert "- `note`: 1" in text
    assert "| receipt | `total` | 3 | 33.3% | 1 | 0 | 0 | 0 | 1 |" in text


def test_error_analysis_compare(two_runs):
    script = load_script("error_analysis")
    assert script.main(["--run", "E2_a", "--compare", "E1_b"]) == 0
    text = (RunPaths("E2_a").eval_details("test") / "vs_E1_b.md").read_text(encoding="utf-8")
    assert "| fixed | 1 (33.3%) |" in text and "| broken | 1 (33.3%) |" in text
    assert "Broken: receipt-001-img" in text and "Fixed: receipt-001-txt" in text
    # items are compared by position, so the reordered sample loses both items' names
    assert "| receipt | `items[].name` | 80.0% | 60.0% | -20.0 pt |" in text
    assert "| receipt | `total` | 33.3% | 66.7% | +33.3 pt |" in text


def test_error_analysis_needs_the_same_samples(two_runs, samples, doc_types):
    run_eval("E3_c", samples[:2], doc_types, [s.target for s in samples[:2]])
    with pytest.raises(SystemExit, match="different samples"):
        load_script("error_analysis").main(["--run", "E3_c", "--compare", "E1_b"])
