import pytest
import yaml
from pydantic import ValidationError

from llm_opt_lab.config import ExperimentConfig, expand_path, load_config

BASE = {
    "experiment_id": "E1",
    "run_name": "gemma4_e2b_lora_r16",
    "model": {"id": "google/gemma-4-E2B-it"},
    "data": {"dir": "/data"},
    "method": {"name": "lora"},
    "lora": {"rank": 16, "alpha": 32},
    "training": {"learning_rate": 2e-4, "max_seq_length": 32768},
}


def test_load_config_and_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(BASE))
    config = load_config(path)
    assert config.run_id == "E1_gemma4_e2b_lora_r16"
    assert config.data.input_types == ["image", "spatial_text"]
    assert config.evaluation.quality_target.exact_match == 0.95


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError, match="lerning_rate"):
        ExperimentConfig.model_validate(
            {**BASE, "training": {**BASE["training"], "lerning_rate": 1e-4}}
        )


def test_lora_requires_lora_and_training_sections():
    with pytest.raises(ValidationError, match="requires sections"):
        ExperimentConfig.model_validate({k: v for k, v in BASE.items() if k != "training"})


def test_qlora_requires_quantization():
    with pytest.raises(ValidationError, match="quantization is required"):
        ExperimentConfig.model_validate({**BASE, "method": {"name": "qlora"}})


def test_rl_section_defaults_and_checks():
    rl = {"init_adapter": "adapter:E1_gemma4_e2b_lora_r16", "num_generations": 4}
    training = {**BASE["training"], "gradient_accumulation_steps": 8}
    config = ExperimentConfig.model_validate({**BASE, "training": training, "rl": rl})
    assert config.rl.loss_type == "dapo" and config.rl.reward_weights.exact_match == 1.0

    with pytest.raises(ValidationError, match="method.name must be lora or qlora"):
        ExperimentConfig.model_validate({**BASE, "method": {"name": "none"}, "rl": rl})
    with pytest.raises(ValidationError, match="whole prompt groups"):
        ExperimentConfig.model_validate({**BASE, "rl": rl})  # 1 x 1 completions per step
    with pytest.raises(ValidationError, match="adapter:<run id>"):
        ExperimentConfig.model_validate(
            {**BASE, "training": training, "rl": {**rl, "init_adapter": "merged:E3"}}
        )


def test_expand_path(monkeypatch):
    monkeypatch.setenv("LLMOPT_DATA_DIR", "/srv/data")
    assert str(expand_path("${LLMOPT_DATA_DIR}/v1")) == "/srv/data/v1"
    monkeypatch.delenv("LLMOPT_DATA_DIR")
    with pytest.raises(ValueError, match="unresolved"):
        expand_path("${LLMOPT_DATA_DIR}/v1")
