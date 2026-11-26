import unicodedata

import pytest

from llm_opt_lab.doc_types import DocType
from llm_opt_lab.postprocess import (
    check_field_types,
    normalize_record,
    normalize_text,
    normalize_value,
)


def test_unicode_is_nfc():
    decomposed = unicodedata.normalize("NFD", "Lê Lợi")
    assert decomposed != "Lê Lợi"
    assert normalize_text(decomposed) == normalize_text("Lê Lợi")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  Cửa   hàng\nAn ", "cửa hàng an"),
        ("", None),
        ("NULL", None),
        (None, None),
        (65000, "65000"),
        (True, "true"),
    ],
)
def test_normalize_text(value, expected):
    assert normalize_text(value) == expected


def test_unknown_field_type():
    with pytest.raises(ValueError, match="unknown field type"):
        normalize_value("x", "currency")
    with pytest.raises(ValueError, match="currency"):
        check_field_types(DocType("bad", "instructions", {}, field_types={"total": "currency"}))


def test_normalize_record_keeps_structure(doc_types):
    record = {"store": {"name": " An "}, "items": [{"price": 40000}], "extra": ""}
    assert normalize_record(record, doc_types["receipt"]) == {
        "store": {"name": "an"},
        "items": [{"price": "40000"}],
        "extra": None,
    }
