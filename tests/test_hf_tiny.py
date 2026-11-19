"""HF code on a tiny random Gemma 4 model, on CPU.

Skipped unless LLMOPT_RUN_TINY=1 (downloads ~70 MB) and the train group is installed:
    LLMOPT_RUN_TINY=1 uv run --group train pytest tests/test_hf_tiny.py
"""

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

if os.environ.get("LLMOPT_RUN_TINY") != "1":
    pytest.skip("set LLMOPT_RUN_TINY=1 to run", allow_module_level=True)
torch = pytest.importorskip("torch")

from conftest import DATA_DIR, DOC_TYPES_DIR  # noqa: E402
from llm_opt_lab.config import ExperimentConfig  # noqa: E402
from llm_opt_lab.data import load_samples  # noqa: E402
from llm_opt_lab.doc_types import load_doc_types  # noqa: E402
from llm_opt_lab.hf.loading import load_model, load_processor  # noqa: E402
from llm_opt_lab.hf.predictor import HFPredictor  # noqa: E402
from llm_opt_lab.hf.sft_data import IGNORE_INDEX, SFTCollator, build_features  # noqa: E402
from llm_opt_lab.paths import ModelSource  # noqa: E402
from llm_opt_lab.prompting import serialize_target  # noqa: E402

TINY = "trl-internal-testing/tiny-Gemma4ForConditionalGeneration"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from PIL import Image

    data_dir = tmp_path_factory.mktemp("data")
    shutil.copy(DATA_DIR / "test.jsonl", data_dir / "test.jsonl")
    (data_dir / "images").mkdir()
    Image.new("RGB", (224, 224), "white").save(data_dir / "images" / "receipt-001.png")
    config = ExperimentConfig.model_validate(
        {
            "experiment_id": "T",
            "run_name": "tiny",
            "model": {"id": TINY, "dtype": "float32"},
            "data": {"dir": str(data_dir), "doc_types_dir": str(DOC_TYPES_DIR)},
            "evaluation": {"max_new_tokens": 8},
        }
    )
    processor = load_processor(config)
    model = load_model(config, ModelSource(TINY), device="cpu")
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
        chat_template_kwargs={},
    )


def test_only_the_answer_is_trained(env):
    tokenizer = env["processor"].tokenizer
    for sample in env["samples"]:
        f = features(env, sample)
        answer = tokenizer.decode(f["labels"][f["labels"] != IGNORE_INDEX])
        assert answer.startswith(serialize_target(sample.target))
        assert ("pixel_values" in f) == (sample.input_type == "image")


def test_label_position_loss_equals_full_loss(env):
    spec = importlib.util.spec_from_file_location(
        "train", Path(__file__).parents[1] / "scripts" / "train.py"
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
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
