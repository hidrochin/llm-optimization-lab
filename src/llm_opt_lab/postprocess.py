"""Normalization applied to both prediction and ground truth before comparison.

The same functions are meant to run in production, so metrics reflect real output.

Only the generic rules are implemented here. Field-type rules (dates, numbers, ...)
must match the company's production postprocess and are ported in T-L05; until then
`date` and `number` fields fall back to the generic text rules.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

from llm_opt_lab.doc_types import DocType

Normalizer = Callable[[str], str | None]

# TODO(T-L05): align with production: which strings count as "empty"?
EMPTY_VALUES = frozenset({"", "null", "none"})

_WHITESPACE = re.compile(r"\s+")
_NORMALIZERS: dict[str, Normalizer] = {}


def register_normalizer(field_type: str) -> Callable[[Normalizer], Normalizer]:
    """Register the type-specific step that runs after `normalize_text`."""

    def decorator(fn: Normalizer) -> Normalizer:
        _NORMALIZERS[field_type] = fn
        return fn

    return decorator


def known_field_types() -> frozenset[str]:
    return frozenset(_NORMALIZERS)


def normalize_text(value: Any) -> str | None:
    """Generic rules for every field: NFC, whitespace, case, empty -> None."""
    if value is None:
        return None
    if isinstance(value, bool):
        value = "true" if value else "false"
    text = unicodedata.normalize("NFC", str(value))
    text = _WHITESPACE.sub(" ", text).strip().casefold()
    return None if text in EMPTY_VALUES else text


@register_normalizer("text")
def _text(value: str) -> str | None:
    return value


@register_normalizer("date")
def _date(value: str) -> str | None:
    # TODO(T-L05): convert to ISO 8601 YYYY-MM-DD using the production rules.
    return value


@register_normalizer("number")
def _number(value: str) -> str | None:
    # TODO(T-L05): strip thousands separators and unify the decimal mark per field.
    return value


def normalize_value(value: Any, field_type: str = "text") -> str | None:
    try:
        normalizer = _NORMALIZERS[field_type]
    except KeyError:
        raise ValueError(f"unknown field type {field_type!r}") from None
    text = normalize_text(value)
    return None if text is None else normalizer(text)


def normalize_record(record: Any, doc_type: DocType | None = None, _path: str = "") -> Any:
    """Normalize every leaf of a nested JSON value, keeping its structure.

    Leaves are looked up in the doc type schema by generic path (`items[].price`);
    fields missing from the schema use the text rules.
    """
    if isinstance(record, dict):
        return {
            key: normalize_record(value, doc_type, f"{_path}.{key}" if _path else key)
            for key, value in record.items()
        }
    if isinstance(record, list):
        return [normalize_record(value, doc_type, f"{_path}[]") for value in record]
    field_type = doc_type.field_type(_path) if doc_type else "text"
    return normalize_value(record, field_type)


def check_field_types(doc_type: DocType) -> None:
    unknown = sorted(set(doc_type.field_types.values()) - known_field_types())
    if unknown:
        raise ValueError(f"doc type {doc_type.name!r} uses unknown field types: {unknown}")
