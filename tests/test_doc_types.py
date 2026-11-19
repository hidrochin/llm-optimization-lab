import pytest

from conftest import DOC_TYPES_DIR
from llm_opt_lab.doc_types import load_doc_types


def test_field_types_from_schema(doc_types):
    receipt = doc_types["receipt"]
    assert receipt.field_types == {
        "store.name": "text",
        "store.address": "text",
        "date": "date",
        "items[].name": "text",
        "items[].quantity": "number",
        "items[].price": "number",
        "total": "number",
    }
    assert receipt.field_type("not.in.schema") == "text"
    assert receipt.system_prompt.startswith("You extract information")


def test_missing_doc_type_is_an_error():
    with pytest.raises(ValueError, match="invoice"):
        load_doc_types(DOC_TYPES_DIR, ["receipt", "invoice"])
