import pytest

from llm_opt_lab.data import Sample
from llm_opt_lab.split import _allocate, split_samples


def make_samples(n_docs: int, doc_type: str) -> list[Sample]:
    samples = []
    for i in range(n_docs):
        group = f"{doc_type}-{i}"
        samples.append(
            Sample(
                id=f"{group}-img",
                doc_type=doc_type,
                input_type="image",
                group_id=group,
                image=f"{group}.png",
                target={},
            )
        )
        samples.append(
            Sample(
                id=f"{group}-txt",
                doc_type=doc_type,
                input_type="spatial_text",
                group_id=group,
                spatial_text="text",
                target={},
            )
        )
    return samples


SAMPLES = make_samples(100, "invoice") + make_samples(40, "receipt")


def test_allocate_sums_to_n():
    assert _allocate(7, {"a": 0.5, "b": 0.3, "c": 0.2}) == {"a": 4, "b": 2, "c": 1}
    assert sum(_allocate(41, {"a": 0.8, "b": 0.05, "c": 0.1, "d": 0.05}).values()) == 41


def test_split_is_deterministic_and_complete():
    first, second = split_samples(SAMPLES, seed=1), split_samples(SAMPLES, seed=1)
    assert first == second
    assert split_samples(SAMPLES, seed=2) != first
    assert sorted(s.id for part in first.values() for s in part) == sorted(s.id for s in SAMPLES)


def test_groups_stay_together_and_doc_types_are_stratified():
    splits = split_samples(SAMPLES)
    owner = {}
    for name, part in splits.items():
        for s in part:
            assert owner.setdefault(s.group, name) == name
    test = splits["test"]
    assert sum(s.doc_type == "invoice" for s in test) == 2 * 10
    assert sum(s.doc_type == "receipt" for s in test) == 2 * 4


def test_bad_ratios():
    with pytest.raises(ValueError, match="sum to 1"):
        split_samples(SAMPLES, {"train": 0.9, "test": 0.2})
