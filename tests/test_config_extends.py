from pathlib import Path

import pytest
import yaml

from llm_opt_lab.config import load_config, load_raw_config
from llm_opt_lab.paths import ModelSource, RunPaths, resolve_model_source

BASE = {
    "experiment_id": "E1",
    "run_name": "reference",
    "model": {"id": "google/gemma-4-E2B-it"},
    "data": {"dir": "/data", "input_types": ["image", "spatial_text"]},
    "method": {"name": "lora"},
    "lora": {"rank": 16, "alpha": 32, "target_modules": ["q_proj"]},
    "training": {"learning_rate": 2e-4, "max_seq_length": 32768, "epochs": 2},
}


def write(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


def test_extends_overrides_one_variable(tmp_path):
    write(tmp_path / "reference.yaml", BASE)
    child = write(
        tmp_path / "e2.yaml",
        {
            "extends": "reference.yaml",
            "experiment_id": "E2",
            "method": {"name": "qlora", "quantization": "nf4"},
            "data": {"input_types": ["image"]},
        },
    )
    config = load_config(child)
    assert config.run_id == "E2_reference"
    assert config.method.quantization == "nf4"
    assert config.lora.rank == 16 and config.training.epochs == 2  # inherited
    assert config.data.input_types == ["image"]  # lists replace, they do not merge
    assert str(config.data.dir) == "/data"  # nested mappings merge


def test_circular_extends(tmp_path):
    write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    write(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ValueError, match="circular"):
        load_raw_config(tmp_path / "a.yaml")


def test_every_committed_config_loads(monkeypatch):
    monkeypatch.setenv("LLMOPT_DATA_DIR", "/srv/data")
    paths = sorted((Path(__file__).parents[1] / "configs").rglob("*.yaml"))
    assert paths
    for path in paths:
        load_config(path)


def test_resolve_model_source(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMOPT_ARTIFACTS_DIR", str(tmp_path))
    config = load_config(write(tmp_path / "c.yaml", BASE))
    own = RunPaths.for_config(config)
    assert resolve_model_source(config, "base") == ModelSource("google/gemma-4-E2B-it")
    assert resolve_model_source(config, "merged") == ModelSource(str(own.merged))
    assert resolve_model_source(config, "adapter") == ModelSource(config.model.id, own.adapter)
    other = resolve_model_source(config, "merged:E1_gemma4_e2b_lora_r16")
    assert other.model == str(tmp_path / "merged" / "E1_gemma4_e2b_lora_r16")
    assert resolve_model_source(config, "google/x").model == "google/x"
