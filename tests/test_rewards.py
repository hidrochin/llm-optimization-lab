import json

import pytest

from conftest import DOC_TYPES_DIR
from llm_opt_lab.doc_types import load_doc_types
from llm_opt_lab.rewards import (
    REWARD_COMPONENTS,
    completion_text,
    make_reward_funcs,
    reward_components,
    score_completion,
)

DOC_TYPES = load_doc_types(DOC_TYPES_DIR)
TARGET = {
    "store": {"name": "Tiệm Bình", "address": None},
    "date": "2026-01-06",
    "items": [{"name": "Trà đá", "quantity": 3, "price": 15000}],
    "total": 15000,
}


def rewards(prediction: str) -> dict[str, float]:
    return reward_components(score_completion(prediction, TARGET, DOC_TYPES["receipt"]))


def test_perfect_prediction_gets_every_reward():
    assert rewards(json.dumps(TARGET, ensure_ascii=False)) == {
        "field_accuracy": 1.0,
        "exact_match": 1.0,
        "similarity": 1.0,
        "parse": 1.0,
        "value_recall": 1.0,
        "null_accuracy": 1.0,
    }


def test_normalization_is_the_evaluation_one():
    # Case and spacing differences are normalized away exactly as in scripts/evaluate.py.
    prediction = {**TARGET, "store": {"name": "  tiệm   bình ", "address": None}}
    assert rewards(json.dumps(prediction, ensure_ascii=False))["exact_match"] == 1.0


def test_one_wrong_field_gives_partial_credit():
    prediction = {**TARGET, "total": 15001}
    r = rewards(json.dumps(prediction, ensure_ascii=False))
    assert r["exact_match"] == 0.0
    assert r["field_accuracy"] == pytest.approx(6 / 7)  # 7 leaves, one wrong
    assert 6 / 7 < r["similarity"] < 1.0  # "15001" is one edit from "15000"


def test_extra_field_breaks_exact_match_only():
    prediction = {**TARGET, "cashier": "An"}
    r = rewards(json.dumps(prediction, ensure_ascii=False))
    assert r["field_accuracy"] == 1.0 and r["exact_match"] == 0.0


def test_unparsed_output_gets_nothing():
    assert rewards("I cannot read this receipt.") == {
        "field_accuracy": 0.0,
        "exact_match": 0.0,
        "similarity": 0.0,
        "parse": 0.0,
        "value_recall": 0.0,
        "null_accuracy": 0.0,
    }


def test_missed_and_invented_values_have_their_own_components():
    # Answering null everywhere scores well on field accuracy when most fields are null;
    # value_recall separates it. Inventing a value lowers null_accuracy.
    lazy = {
        "store": {"name": None, "address": None},
        "date": None,
        "items": [{"name": None, "quantity": None, "price": None}],
        "total": None,
    }
    r = rewards(json.dumps(lazy))
    assert r["value_recall"] == 0.0 and r["null_accuracy"] == 1.0
    inventing = {**TARGET, "store": {"name": "Tiệm Bình", "address": "1 Lê Lợi"}}
    r = rewards(json.dumps(inventing, ensure_ascii=False))
    assert r["value_recall"] == 1.0 and r["null_accuracy"] == 0.0


def test_completion_text_accepts_trl_formats():
    assert completion_text('{"a": 1}') == '{"a": 1}'
    assert completion_text([{"role": "assistant", "content": '{"a": 1}'}]) == '{"a": 1}'
    parts = [{"type": "text", "text": '{"a": '}, {"type": "text", "text": "1}"}]
    assert completion_text([{"role": "assistant", "content": parts}]) == '{"a": 1}'


def test_reward_funcs_follow_the_trl_calling_convention():
    funcs = make_reward_funcs(DOC_TYPES)
    assert [f.__name__ for f in funcs] == list(REWARD_COMPONENTS)
    good = json.dumps(TARGET, ensure_ascii=False)
    completions = [
        [{"role": "assistant", "content": good}],
        [{"role": "assistant", "content": "x"}],
    ]
    kwargs = {
        "prompts": [[], []],
        "target_json": [good, good],
        "doc_type": ["receipt", "receipt"],
        "sample_id": ["a", "a"],  # other dataset columns and TRL extras are ignored
        "trainer_state": None,
    }
    by_name = {f.__name__: f(completions=completions, **kwargs) for f in funcs}
    assert by_name["exact_match"] == [1.0, 0.0]
    assert by_name["parse"] == [1.0, 0.0]
