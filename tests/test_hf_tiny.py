"""HF code on tiny random Gemma 4, Qwen3.5 and Qwen3.6 MoE models, on CPU.

Skipped unless LLMOPT_RUN_TINY=1 (downloads ~200 MB) and the train group is installed:
    LLMOPT_RUN_TINY=1 uv run --group train pytest tests/test_hf_tiny.py
"""

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

if os.environ.get("LLMOPT_RUN_TINY") != "1":
    pytest.skip("set LLMOPT_RUN_TINY=1 to run", allow_module_level=True)
torch = pytest.importorskip("torch")

from conftest import DATA_DIR, DOC_TYPES_DIR  # noqa: E402
from llm_opt_lab.config import ExperimentConfig, FieldSampling, FieldsConfig  # noqa: E402
from llm_opt_lab.data import load_samples  # noqa: E402
from llm_opt_lab.doc_types import load_doc_types  # noqa: E402
from llm_opt_lab.hf.loading import load_model, load_processor  # noqa: E402
from llm_opt_lab.hf.predictor import HFPredictor  # noqa: E402
from llm_opt_lab.hf.sft_data import (  # noqa: E402
    IGNORE_INDEX,
    SFTCollator,
    SFTDataset,
    build_features,
)
from llm_opt_lab.metrics import flatten  # noqa: E402
from llm_opt_lab.parsing import parse_json_output  # noqa: E402
from llm_opt_lab.paths import ModelSource  # noqa: E402
from llm_opt_lab.prompting import serialize_target  # noqa: E402

ROOT = Path(__file__).parents[1]
# The "Think" Qwen variants use the real Qwen3.5/3.6 chat template; production turns
# thinking off, as the configs do.
MODELS = {
    "gemma4": ("trl-internal-testing/tiny-Gemma4ForConditionalGeneration", {}),
    "qwen3_5": (
        "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-Think",
        {"enable_thinking": False},
    ),
    "qwen3_6_moe": (
        "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6",
        {"enable_thinking": False},
    ),
}


def load_script(relative: str):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=list(MODELS))
def env(request, tmp_path_factory):
    from PIL import Image

    model_id, chat_template_kwargs = MODELS[request.param]
    data_dir = tmp_path_factory.mktemp("data")
    for split in ("train", "test"):
        shutil.copy(DATA_DIR / "test.jsonl", data_dir / f"{split}.jsonl")
    (data_dir / "images").mkdir()
    Image.new("RGB", (224, 224), "white").save(data_dir / "images" / "receipt-001.png")
    config = ExperimentConfig.model_validate(
        {
            "experiment_id": "T",
            "run_name": "tiny",
            "model": {
                "id": model_id,
                "dtype": "float32",
                "chat_template_kwargs": chat_template_kwargs,
            },
            "data": {"dir": str(data_dir), "doc_types_dir": str(DOC_TYPES_DIR)},
            "evaluation": {"max_new_tokens": 8},
        }
    )
    processor = load_processor(config)
    model = load_model(config, ModelSource(model_id), device="cpu")
    return {
        "config": config,
        "processor": processor,
        "model": model,
        "samples": load_samples(data_dir / "test.jsonl"),
        "doc_types": load_doc_types(DOC_TYPES_DIR),
        "data_dir": data_dir,
    }


def features(env, sample):
    return build_features(
        sample,
        env["doc_types"][sample.doc_type],
        env["processor"],
        env["data_dir"],
        text_only=False,
        chat_template_kwargs=env["config"].model.chat_template_kwargs,
    )


def test_only_the_answer_is_trained(env):
    tokenizer = env["processor"].tokenizer
    for sample in env["samples"]:
        f = features(env, sample)
        answer = tokenizer.decode(f["labels"][f["labels"] != IGNORE_INDEX])
        assert answer.startswith(serialize_target(sample.target))
        assert ("pixel_values" in f) == (sample.input_type == "image")


def test_sampled_requests_train_on_the_requested_fields(env):
    """Each draw asks for another field subset; the answer holds exactly those fields."""
    fields = FieldsConfig(sampling=FieldSampling(full_rate=0.0, decoy_rate=1.0))
    dataset = SFTDataset(
        env["samples"][:1],
        env["doc_types"],
        env["processor"],
        env["data_dir"],
        text_only=False,
        chat_template_kwargs=env["config"].model.chat_template_kwargs,
        max_seq_length=8192,
        fields=fields,
        seed=1,
    )
    tokenizer = env["processor"].tokenizer
    target = flatten(env["samples"][0].target)
    answers = []
    for _ in range(4):
        labels = dataset[0]["labels"]  # the prefix check inside passes for every request
        answer, _ = parse_json_output(tokenizer.decode(labels[labels != IGNORE_INDEX]))
        leaves = flatten(answer)
        decoys = [path for path in leaves if path not in target]
        assert decoys and all(leaves[path] is None for path in decoys)
        assert all(leaves[path] == target[path] for path in leaves if path in target)
        answers.append(answer)
    assert len({json.dumps(a, sort_keys=True) for a in answers}) > 1


def test_label_position_loss_equals_full_loss(env):
    train = load_script("scripts/train.py")
    collator = SFTCollator(env["processor"].tokenizer.pad_token_id)
    with torch.no_grad():
        for sample in env["samples"]:
            batch = collator([features(env, sample)])
            fast, _ = train.compute_sft_loss(env["model"], batch, "label_positions")
            full, _ = train.compute_sft_loss(env["model"], batch, "full")
            assert fast.item() == pytest.approx(full.item(), rel=1e-5)


def test_predictor_writes_a_prediction(env):
    predictor = HFPredictor(
        env["model"], env["processor"], env["config"], env["doc_types"], env["data_dir"]
    )
    for sample in env["samples"]:
        prediction = predictor.predict(sample)
        assert prediction.id == sample.id and prediction.error is None
        assert 0 < prediction.completion_tokens <= 8


def test_grpo_prompt_matches_the_sft_prompt(env):
    """TRL rebuilds the prompt from our dataset row; the tokens must be the ones SFT used."""
    from datasets import Dataset
    from PIL import Image
    from trl.data_utils import prepare_multimodal_messages

    from llm_opt_lab.hf.sft_data import apply_template, chat_messages

    train_grpo = load_script("scripts/train_grpo.py")
    kwargs = env["config"].model.chat_template_kwargs
    for sample in env["samples"]:
        doc_type = env["doc_types"][sample.doc_type]
        prompt, images = train_grpo.rl_prompt(sample, doc_type, env["data_dir"], text_only=False)
        stored = Dataset.from_list([{"prompt": prompt}])[0]["prompt"]  # as the trainer reads it
        rebuilt = prepare_multimodal_messages(stored, images=[Image.open(p) for p in images])
        ours = chat_messages(sample, doc_type, env["data_dir"], text_only=False)
        trl_ids, our_ids = (
            apply_template(
                env["processor"], m, add_generation_prompt=True, chat_template_kwargs=kwargs
            )["input_ids"]
            for m in (rebuilt, ours)
        )
        assert torch.equal(trl_ids, our_ids)


def run_config(script: str, config: dict, directory: Path) -> int:
    path = directory / f"{config['run_name']}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_script(script).main(["--config", str(path)])


def test_grpo_continues_the_sft_adapter(env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # results/ is relative to the working directory
    monkeypatch.setenv("LLMOPT_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    model = env["config"].model
    sft = {
        "experiment_id": "T",
        "run_name": "sft",
        "model": {"id": model.id, "dtype": "float32",
                  "chat_template_kwargs": model.chat_template_kwargs},
        "data": {"dir": str(env["data_dir"]), "doc_types_dir": str(DOC_TYPES_DIR)},
        "fields": {"sampling": {"decoy_rate": 0.5}},  # SFT and GRPO on sampled requests
        "method": {"name": "lora"},
        "lora": {"rank": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        "training": {
            "learning_rate": 1.0e-4,
            "batch_size": 1,  # SFT batches cannot mix image and spatial-text samples
            "gradient_accumulation_steps": 2,
            "max_steps": 2,
            "max_seq_length": 4096,
            "logging_steps": 1,
            "gradient_checkpointing": False,
        },
        "evaluation": {"max_new_tokens": 8},
    }  # fmt: skip
    assert run_config("scripts/train.py", sft, tmp_path) == 0
    grpo = {
        **sft,
        "run_name": "grpo",
        "training": {**sft["training"], "max_steps": 3},  # one prompt per step: all 3 samples
        "rl": {"init_adapter": "adapter:T_sft", "num_generations": 2},
    }
    assert run_config("scripts/train_grpo.py", grpo, tmp_path) == 0

    adapter = tmp_path / "artifacts" / "adapters" / "T_grpo"
    assert (adapter / "adapter_model.safetensors").is_file()
    log_lines = (tmp_path / "results/logs/T_grpo/train_log.jsonl").read_text().splitlines()
    rewards = [e for e in map(json.loads, log_lines) if "rewards/exact_match/mean" in e]
    assert rewards and all(0 <= e["rewards/field_accuracy/mean"] <= 1 for e in rewards)
    metrics = json.loads((tmp_path / "results/metrics/T_grpo.json").read_text())
    assert metrics["trainable_params"] > 0 and "train" in metrics["stages"]
