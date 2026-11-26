"""Exact match, field-level accuracy, NED, parse rate and key presence over nested JSON.

Both sides are normalized first (see postprocess). Nested values are flattened to leaf
paths such as `store.name` or `items[0].price`; list items are compared by position.
The ground truth holds only the requested fields (see field_requests).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from rapidfuzz.distance import Levenshtein

_INDEX = re.compile(r"\[\d+\]")


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON to `{leaf_path: value}`. Empty dicts and lists become None leaves."""
    if isinstance(value, dict) and value:
        items: Iterable[tuple[str, Any]] = (
            (f"{prefix}.{k}" if prefix else k, v) for k, v in value.items()
        )
    elif isinstance(value, list) and value:
        items = ((f"{prefix}[{i}]", v) for i, v in enumerate(value))
    else:
        return {prefix: None if isinstance(value, (dict, list)) else value}
    flat: dict[str, Any] = {}
    for path, sub in items:
        flat.update(flatten(sub, path))
    return flat


def generic_path(path: str) -> str:
    """`items[3].price` -> `items[].price`, to aggregate a field across list positions."""
    return _INDEX.sub("[]", path)


def normalized_edit_distance(gold: str | None, pred: str | None) -> float:
    """Levenshtein distance divided by the longer length; 0 means identical."""
    return Levenshtein.normalized_distance(gold or "", pred or "")


@dataclass(frozen=True)
class FieldResult:
    path: str
    gold: str | None
    pred: str | None
    correct: bool
    ned: float
    present: bool = False  # the output has this path (a requested key it did not drop)


@dataclass(frozen=True)
class SampleScore:
    id: str
    doc_type: str
    input_type: str
    parsed: bool
    exact_match: bool
    fields: list[FieldResult]
    # Non-null predicted leaves that do not exist in the ground truth. They break exact
    # match but are not in the field-accuracy denominator (expected fields only).
    extra_fields: list[str]


def score_sample(
    sample_id: str,
    doc_type: str,
    input_type: str,
    gold: dict[str, Any],
    pred: dict[str, Any] | None,
    ignore: frozenset[str] = frozenset(),
) -> SampleScore:
    """Score one normalized prediction; `pred=None` means unparseable: every field is wrong.

    `gold` holds only the requested fields. Predicted leaves under `ignore` (generic paths
    of requested fields the record does not annotate) are neither scored nor extra.
    """
    gold_flat = flatten(gold)
    if pred is None:
        fields = [FieldResult(p, g, None, False, 1.0) for p, g in gold_flat.items()]
        return SampleScore(sample_id, doc_type, input_type, False, False, fields, [])
    pred_flat = flatten(pred)
    fields = []
    for path, gold_value in gold_flat.items():
        pred_value = pred_flat.get(path)
        ned = normalized_edit_distance(gold_value, pred_value)
        fields.append(
            FieldResult(
                path, gold_value, pred_value, pred_value == gold_value, ned, path in pred_flat
            )
        )
    extra = sorted(
        p
        for p, v in pred_flat.items()
        if p not in gold_flat and v is not None and generic_path(p) not in ignore
    )
    exact = all(f.correct for f in fields) and not extra
    return SampleScore(sample_id, doc_type, input_type, True, exact, fields, extra)


class QualityStats(BaseModel):
    n: int
    n_fields: int
    exact_match: float | None
    field_accuracy: float | None
    ned_mean: float | None
    parse_rate: float | None
    # Share of expected leaf paths present in the output: a dropped or renamed key (or a
    # missing list entry) lowers it even when the expected value is null.
    key_presence: float | None = None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(scores: list[SampleScore]) -> QualityStats:
    fields = [f for s in scores for f in s.fields]
    return QualityStats(
        n=len(scores),
        n_fields=len(fields),
        exact_match=_mean([float(s.exact_match) for s in scores]),
        field_accuracy=_mean([float(f.correct) for f in fields]),
        ned_mean=_mean([f.ned for f in fields]),
        parse_rate=_mean([float(s.parsed) for s in scores]),
        key_presence=_mean([float(f.present) for f in fields]),
    )


def summarize_by(scores: list[SampleScore], key: str) -> dict[str, QualityStats]:
    groups: dict[str, list[SampleScore]] = defaultdict(list)
    for score in scores:
        groups[getattr(score, key)].append(score)
    return {name: summarize(group) for name, group in sorted(groups.items())}


class FieldStats(BaseModel):
    doc_type: str
    field: str  # generic path
    n: int
    accuracy: float
    ned_mean: float


def per_field_stats(scores: list[SampleScore]) -> list[FieldStats]:
    """Accuracy per (doc type, field), lowest accuracy first, to see what drags scores down."""
    groups: dict[tuple[str, str], list[FieldResult]] = defaultdict(list)
    for score in scores:
        for f in score.fields:
            groups[(score.doc_type, generic_path(f.path))].append(f)
    stats = [
        FieldStats(
            doc_type=doc_type,
            field=field,
            n=len(results),
            accuracy=sum(r.correct for r in results) / len(results),
            ned_mean=sum(r.ned for r in results) / len(results),
        )
        for (doc_type, field), results in groups.items()
    ]
    return sorted(stats, key=lambda s: (s.accuracy, s.doc_type, s.field))


class FieldGroupStats(BaseModel):
    """Field-level scores over one group of fields (e.g. seen, held out, decoy)."""

    n_fields: int
    field_accuracy: float | None
    ned_mean: float | None
    key_presence: float | None
    # Expected non-null fields predicted null (a missed value); expected-null fields
    # predicted non-null (an invented value, the decoy false-positive rate).
    missed_value_rate: float | None
    invented_value_rate: float | None


def summarize_field_group(results: list[FieldResult]) -> FieldGroupStats:
    with_value = [r for r in results if r.gold is not None]
    without_value = [r for r in results if r.gold is None]
    return FieldGroupStats(
        n_fields=len(results),
        field_accuracy=_mean([float(r.correct) for r in results]),
        ned_mean=_mean([r.ned for r in results]),
        key_presence=_mean([float(r.present) for r in results]),
        missed_value_rate=_mean([float(r.pred is None) for r in with_value]),
        invented_value_rate=_mean([float(r.pred is not None) for r in without_value]),
    )
