import random

import pytest

from llm_opt_lab.config import FieldSampling, FieldsConfig
from llm_opt_lab.field_requests import (
    annotated_paths,
    inference_request,
    longest_training_request,
    project,
    scored_target,
    skeleton,
    training_request,
)

TARGET = {
    "store": {"name": "Tiệm Bình", "address": None},
    "date": "2026-01-06",
    "items": [{"name": "Trà đá", "quantity": 3, "price": 15000}],
    "total": 15000,
}
HOLDOUT = FieldsConfig(holdout={"receipt": ["date", "items[].price"]})


def test_a_present_key_is_annotated_and_a_missing_key_is_not(doc_types):
    receipt = doc_types["receipt"]
    partial = {"store": {"name": "An"}, "items": [{"name": "x"}], "total": None}
    assert annotated_paths(partial, receipt) == {"store.name", "items[].name", "total"}
    # An empty list answers every field of its entries.
    assert {"items[].name", "items[].price"} <= annotated_paths({"items": []}, receipt)


def test_project_keeps_request_order_and_fills_gaps():
    record = {"a": {"b": 1, "c": 2}, "items": [{"x": 1, "y": 2}, {"x": 3}]}
    assert project(record, ["items[].y", "a.c", "missing", "nothing[].z"]) == {
        "items": [{"y": 2}, {"y": None}],
        "a": {"c": 2},
        "missing": None,
        "nothing": [],
    }
    assert skeleton(["a.c", "items[].sub[].y"]) == {"a": {"c": ""}, "items": [{"sub": [{"y": ""}]}]}


def test_inference_request_modes(doc_types):
    receipt = doc_types["receipt"]
    full = inference_request(receipt, FieldsConfig()).paths
    assert full == [f.path for f in receipt.extract_fields]  # registry order, no derive field
    seen = inference_request(receipt, HOLDOUT.model_copy(update={"inference": "seen"})).paths
    assert "date" not in seen and "items[].price" not in seen and "total" in seen
    held_out = HOLDOUT.model_copy(update={"inference": "held_out", "decoys": 1})
    assert inference_request(receipt, held_out).paths == [
        "date",
        "items[].price",
        "container_number",
    ]


def test_holdout_must_name_extract_fields(doc_types):
    with pytest.raises(ValueError, match="loyalty_points"):
        inference_request(
            doc_types["receipt"], FieldsConfig(holdout={"receipt": ["loyalty_points"]})
        )
    with pytest.raises(ValueError, match="needs fields.holdout"):
        FieldsConfig(inference="seen")


def test_fixed_training_request_skips_holdout_and_unannotated_fields(doc_types):
    receipt = doc_types["receipt"]
    target = {**TARGET}
    del target["total"]  # not annotated in this record
    paths = training_request(target, receipt, HOLDOUT).paths
    assert paths == ["store.name", "store.address", "items[].name", "items[].quantity"]


def test_sampled_requests_vary_and_stay_inside_the_annotated_fields(doc_types):
    receipt = doc_types["receipt"]
    config = FieldsConfig(sampling=FieldSampling(full_rate=0.2, decoy_rate=0.5, max_decoys=2))
    full = training_request(TARGET, receipt, config).paths
    allowed = set(full) | {d.path for d in receipt.decoys}
    drawn = [training_request(TARGET, receipt, config, random.Random(i)).paths for i in range(200)]
    assert all(set(paths) <= allowed and paths for paths in drawn)
    assert len({tuple(p) for p in drawn}) > 50  # subsets, orders and decoys differ
    assert sum(p == full for p in drawn) >= 20  # full_rate keeps the fixed prompt in the mix
    assert any("container_number" in p for p in drawn)
    assert {len(p) for p in drawn} >= {1, len(full)}
    longest = longest_training_request(TARGET, receipt, config).paths
    assert max(len(p) for p in drawn) <= len(longest)


def test_held_out_decoys_are_kept_for_evaluation(doc_types):
    receipt = doc_types["receipt"]
    config = FieldsConfig(
        holdout={"receipt": ["container_number"]},
        decoys=1,
        sampling=FieldSampling(full_rate=0.0, decoy_rate=1.0),
    )
    drawn = {
        path
        for i in range(100)
        for path in training_request(TARGET, receipt, config, random.Random(i)).paths
    }
    assert "vessel_name" in drawn and "container_number" not in drawn
    assert inference_request(receipt, config).paths[-1] == "container_number"


def test_same_seed_same_request(doc_types):
    config = FieldsConfig(sampling=FieldSampling())
    draw = [
        training_request(TARGET, doc_types["receipt"], config, random.Random("42:id:0")).paths
        for _ in range(2)
    ]
    assert draw[0] == draw[1]


def test_scored_target_skips_unannotated_and_nulls_decoys(doc_types):
    receipt = doc_types["receipt"]
    target = {"store": {"name": "An"}, "total": 5}
    scored = scored_target(target, receipt, ["total", "date", "container_number"])
    assert scored.gold == {"total": 5, "container_number": None}
    assert scored.skipped == {"date"}
