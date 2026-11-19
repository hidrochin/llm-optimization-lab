import json

import pytest

from llm_opt_lab.data import filter_samples, load_samples, write_samples


def test_load_samples(samples):
    assert [s.input_type for s in samples] == ["image", "spatial_text", "spatial_text"]
    assert samples[0].group == "receipt-001"


def test_roundtrip(tmp_path, samples):
    write_samples(tmp_path / "out.jsonl", samples)
    assert load_samples(tmp_path / "out.jsonl") == samples


def test_missing_input_field_is_rejected(tmp_path):
    record = {"id": "x", "doc_type": "receipt", "input_type": "image", "target": {}}
    (tmp_path / "bad.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_samples(tmp_path / "bad.jsonl")


def test_duplicate_ids_are_rejected(tmp_path, samples):
    write_samples(tmp_path / "dup.jsonl", [samples[0], samples[0]])
    with pytest.raises(ValueError, match="duplicate"):
        load_samples(tmp_path / "dup.jsonl")


def test_filter_samples(samples):
    assert len(filter_samples(samples, input_types=["spatial_text"])) == 2
    assert len(filter_samples(samples, doc_types=["invoice"])) == 0
    assert filter_samples(samples) == samples
